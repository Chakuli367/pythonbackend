# Standard library
import os
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from flask_cors import cross_origin
from openai import OpenAI
import traceback 
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import json
import re


# Third-party
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from pydantic import BaseModel, Field

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # CORS for all originsg

# Load Firebase config from environment variable
firebase_config_json = os.environ.get("FIREBASE_CONFIG")
if not firebase_config_json:
    raise EnvironmentError("FIREBASE_CONFIG environment variable not set")

try:
    firebase_json = json.loads(firebase_config_json)
except json.JSONDecodeError:
    raise ValueError("FIREBASE_CONFIG is not valid JSON")

# Initialize Firebase app
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_json)
    initialize_app(cred)

# Firestore client
db = firestore.client()

def save_to_firebase(user_id, category, doc_id, data):
    """
    Save a document under users/{user_id}/{category}/{doc_id}.
    """
    if not user_id:
        return
    try:
        doc_ref = db.collection("users").document(user_id).collection(category).document(doc_id)
        doc_ref.set(data)
    except Exception as e:
        print(f"[FIREBASE ERROR] {e}")


client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

LOGS_FILE = "logs.json"
REWARD_FILE = "user_rewards.json"

def load_prompt(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def read_logs():
    if not os.path.exists(LOGS_FILE):
        return []
    with open(LOGS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def write_logs(logs):
    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

def read_rewards():
    if not os.path.exists(REWARD_FILE):
        return {}
    with open(REWARD_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def safe_format(template, **kwargs):
    """Safely format template with default values for missing keys"""
    class SafeDict(defaultdict):
        def __missing__(self, key):
            return f"{{{key}}}"
    
    safe_dict = SafeDict(str)
    safe_dict.update(kwargs)
    return template.format_map(safe_dict)

def normalize_places(places):
    """Normalize place names to title case to avoid duplicates"""
    return [place.strip().title() for place in places if place.strip()]

def merge_places(existing, new):
    """Merge place lists avoiding duplicates (case-insensitive)"""
    # Normalize both lists
    normalized_existing = normalize_places(existing)
    normalized_new = normalize_places(new)
    
    # Create a set for case-insensitive comparison
    existing_lower = {p.lower() for p in normalized_existing}
    merged = normalized_existing.copy()
    
    for place in normalized_new:
        if place.lower() not in existing_lower:
            merged.append(place)
            existing_lower.add(place.lower())
    
    return merged

def call_llm_with_retry(messages, temperature=0.6, max_tokens=500, max_retries=3):
    """Call LLM API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"API call attempt {attempt + 1} failed: {e}")
            continue
    return None

def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown code blocks"""
    try:
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {text}")
        return None

def load_prompt_file(filename, default_content=""):
    """Load prompt file with fallback"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found, using default")
        return default_content
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return default_content

def truncate_chat_history(chat_history, max_messages=20):
    """Truncate chat history to prevent token limit issues"""
    if len(chat_history) <= max_messages:
        return chat_history
    
    # Keep first message (usually intro) and last N messages
    return [chat_history[0]] + chat_history[-(max_messages-1):]

def create_initial_chat(user_id, goal_name="", user_interests=None):
    """Create initial chat document for user"""
    if user_interests is None:
        user_interests = []
    
    initial_message = {
        "role": "assistant",
        "content": f"Hi! I'm here to help you with {goal_name if goal_name else 'your goals'}. Tell me about yourself - what places do you like to visit? What are your interests?"
    }
    
    chat_doc = {
        "day": 1,
        "chat": [initial_message],
        "created_at": firestore.SERVER_TIMESTAMP,
        "goal_name": goal_name,
        "user_interests": user_interests
    }
    
    return chat_doc

def write_rewards(data):
    with open(REWARD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def parse_story_analysis(analysis_text):
    """
    Parse LLM response into structured story analysis format.
    Expected format from LLM should be JSON or structured text.
    """
    try:
        # Try to parse as JSON first
        import re
        
        # Look for JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', analysis_text)
        if json_match:
            analysis_json = json.loads(json_match.group(0))
            return analysis_json
        
        # If no JSON found, try to parse structured text manually
        # This is a fallback parser
        lines = analysis_text.strip().split('\n')
        
        analysis = {
            "overallScore": 0,
            "mechanics": {},
            "strengths": [],
            "improvements": [],
            "rewrittenVersion": ""
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Parse overall score
            if "overall score" in line.lower() or "overall:" in line.lower():
                score_match = re.search(r'(\d+)', line)
                if score_match:
                    analysis["overallScore"] = int(score_match.group(1))
            
            # Parse mechanics
            elif "hook:" in line.lower():
                current_section = "hook"
                analysis["mechanics"]["hook"] = {"score": 0, "feedback": ""}
            elif "emotion:" in line.lower() or "relatable emotion:" in line.lower():
                current_section = "emotion"
                analysis["mechanics"]["emotion"] = {"score": 0, "feedback": ""}
            elif "details:" in line.lower() or "specific details:" in line.lower():
                current_section = "details"
                analysis["mechanics"]["details"] = {"score": 0, "feedback": ""}
            elif "stakes:" in line.lower():
                current_section = "stakes"
                analysis["mechanics"]["stakes"] = {"score": 0, "feedback": ""}
            elif "resolution:" in line.lower():
                current_section = "resolution"
                analysis["mechanics"]["resolution"] = {"score": 0, "feedback": ""}
            elif "bridge:" in line.lower():
                current_section = "bridge"
                analysis["mechanics"]["bridge"] = {"score": 0, "feedback": ""}
            
            # Parse strengths
            elif "strengths:" in line.lower():
                current_section = "strengths"
            elif "improvements:" in line.lower() or "areas to improve:" in line.lower():
                current_section = "improvements"
            elif "rewritten" in line.lower() or "improved version:" in line.lower():
                current_section = "rewritten"
            
            # Parse content based on current section
            elif current_section in ["hook", "emotion", "details", "stakes", "resolution", "bridge"]:
                if line:
                    score_match = re.search(r'(\d+)/100', line)
                    if score_match:
                        analysis["mechanics"][current_section]["score"] = int(score_match.group(1))
                    if "feedback:" in line.lower():
                        feedback = line.split("feedback:", 1)[1].strip()
                        analysis["mechanics"][current_section]["feedback"] = feedback
                    elif analysis["mechanics"][current_section]["feedback"] == "":
                        analysis["mechanics"][current_section]["feedback"] = line
            
            elif current_section == "strengths" and line and line.startswith(("-", "•", "*", "✓")):
                analysis["strengths"].append(line.lstrip("-•*✓ ").strip())
            
            elif current_section == "improvements" and line and line.startswith(("-", "•", "*", "→")):
                analysis["improvements"].append(line.lstrip("-•*→ ").strip())
            
            elif current_section == "rewritten" and line:
                analysis["rewrittenVersion"] += line + " "
        
        # Clean up rewritten version
        analysis["rewrittenVersion"] = analysis["rewrittenVersion"].strip().strip('"').strip("'")
        
        # Ensure all mechanics have default values if missing
        for mechanic in ["hook", "emotion", "details", "stakes", "resolution", "bridge"]:
            if mechanic not in analysis["mechanics"]:
                analysis["mechanics"][mechanic] = {"score": 50, "feedback": "No feedback available"}
        
        return analysis
        
    except Exception as e:
        print(f"Error parsing story analysis: {str(e)}")
        # Return default structure on parse failure
        return {
            "overallScore": 50,
            "mechanics": {
                "hook": {"score": 50, "feedback": "Unable to analyze"},
                "emotion": {"score": 50, "feedback": "Unable to analyze"},
                "details": {"score": 50, "feedback": "Unable to analyze"},
                "stakes": {"score": 50, "feedback": "Unable to analyze"},
                "resolution": {"score": 50, "feedback": "Unable to analyze"},
                "bridge": {"score": 50, "feedback": "Unable to analyze"}
            },
            "strengths": ["Analysis error occurred"],
            "improvements": ["Please try again"],
            "rewrittenVersion": story_text
        }






# ================== PHASE CONFIG ==================
PHASE_REQUIREMENTS = {
    1: ["problem", "context", "emotion", "impact"],
    2: ["skill_gaps", "tips"],
    3: ["locations", "schedule", "anxiety_issues"],
    4: ["analyzed_schedule", "optimal_times", "energy_map"]
}

sessions = {}

# ================== ENHANCED AGENT PROMPTS ==================

PHASE_1_PROMPT = """
# AGENT IDENTITY
You are Jordan, a 28-year-old mentor who went from socially awkward engineering student to confident communicator. You struggled with eye contact, small talk, and reading social cues. You understand the anxiety, shame, and loneliness that comes with social struggles.

# YOUR MISSION (PHASE 1: ROOT CAUSE DISCOVERY)
The user has submitted their intake form data. Your job is to:
1. Acknowledge their struggle with empathy and personal stories
2. Extract key information from their responses
3. Validate their feelings
4. Keep responses SHORT and conversational (3-4 sentences max)

# RESPONSE FORMAT
You MUST respond with valid JSON:
```json
{
  "message": "Your empathetic response acknowledging their specific problem",
  "extracted_data": {
    "problem": "brief summary of their main challenge",
    "context": "where this happens most",
    "emotion": "their emotional state",
    "impact": "what this has cost them"
  },
  "ready_for_next_phase": true
}
```

# CONVERSATION RULES
1. Be casual - "Hey", "I get it", "That sucks"
2. Share brief personal failures
3. No generic advice yet
4. Always set ready_for_next_phase to true after processing form data
"""

PHASE_2_PROMPT = """
# AGENT IDENTITY
You are Jordan, now you know their struggle. Time to get tactical.

# YOUR MISSION (PHASE 2: SKILL GAP ANALYSIS)
The user submitted their skills assessment. Based on their form data:
1. Validate their past attempts (if any)
2. Identify their weakest skill area
3. Provide ONE concrete, testable technique
4. Extract structured skill data

# RESPONSE FORMAT
```json
{
  "message": "Your response with ONE technique (4-5 sentences max)",
  "extracted_data": {
    "skill_gaps": "their weakest area identified",
    "tips": "one concrete technique they can use",
    "past_attempts": "what they've tried before"
  },
  "ready_for_next_phase": true
}
```

Keep it SHORT - 4-5 sentences max.
"""

PHASE_3_PROMPT = """
# AGENT IDENTITY
You are Jordan, accountability partner mode.

# YOUR MISSION (PHASE 3: LOGISTICS)
User submitted their logistics form. Your job:
1. Acknowledge their commitment level
2. Address their top anxiety trigger
3. Confirm their practice locations make sense
4. Extract and structure the data

# RESPONSE FORMAT
```json
{
  "message": "Your tactical validation (3-4 sentences)",
  "extracted_data": {
    "locations": ["location1", "location2"],
    "schedule": "when they can practice",
    "anxiety_issues": "their main anxiety trigger",
    "commitment_level": 8
  },
  "ready_for_next_phase": true
}
```
"""

PHASE_4_ANALYSIS_PROMPT = """
# AGENT IDENTITY
You are Jordan, now analyzing their schedule like a coach planning training sessions.

# YOUR MISSION (PHASE 4: SCHEDULE OPTIMIZATION)
User submitted their weekly schedule. You need to:

1. **ANALYZE THEIR SCHEDULE** for optimal practice windows:
    - Identify low-stress times when they have energy
    - Find existing social touchpoints they can leverage
    - Avoid their stress peaks
    - Match practice difficulty to energy levels

2. **EXTRACT STRUCTURED DATA** - Return this JSON:
```json
{
  "message": "Brief analysis in your casual Jordan voice (2-3 sentences)",
  "extracted_data": {
    "analyzed_schedule": {
      "best_practice_windows": ["Monday 7-8am at Starbucks", "Wednesday 6pm at gym"],
      "existing_social_opportunities": ["Daily standup meetings", "Friday team lunch"],
      "avoid_times": ["Monday mornings (stressed)", "Friday afternoons (deadline crunch)"],
      "energy_peaks": ["Mid-morning (9am-12pm)", "Early evening (5-7pm)"]
    },
    "optimal_times": [
      {
        "day": "Monday",
        "time": "9:30am",
        "location": "Coffee shop on commute",
        "energy_level": "high",
        "difficulty": "easy",
        "task": "Smile at barista + comment on coffee"
      }
    ],
    "energy_map": {
      "high_energy_windows": ["Mon/Wed/Fri 9am-12pm"],
      "medium_energy_windows": ["Tue/Thu afternoons"],
      "low_energy_windows": ["After 8pm weekdays"]
    }
  },
  "ready_for_next_phase": true
}
```

3. **BE STRATEGIC**:
    - Start with high-energy + existing social touchpoints
    - Progressive difficulty: morning coffee small talk → gym longer convos → work networking
    - Avoid stacking practices on their worst days
    - Leverage their routine (don't add extra trips)
"""

PHASE_5_CONFIRMATION_PROMPT = """
# AGENT IDENTITY
You are Jordan, doing final confirmation before generating the plan.

# YOUR MISSION (PHASE 5: FINAL CONFIRMATION)
Review ALL collected data and provide a summary for the user to confirm.

# RESPONSE FORMAT
```json
{
  "message": "Here's what we've got. [Summary in 3-4 sentences]. Sound right?",
  "confirmation_summary": {
    "problem": "Brief statement",
    "skill_focus": "Main skill gap",
    "locations": ["Location 1", "Location 2"],
    "schedule": "When they'll practice",
    "commitment": "X/10"
  },
  "ready_to_generate_plan": false
}
```

When user confirms with "yes", "looks good", "let's do it":
```json
{
  "message": "Let's fucking go. Generating your 5-day plan now...",
  "ready_to_generate_plan": true
}
```
"""

PHASE_PROMPTS = {
    1: PHASE_1_PROMPT,
    2: PHASE_2_PROMPT,
    3: PHASE_3_PROMPT,
    4: PHASE_4_ANALYSIS_PROMPT,
    5: PHASE_5_CONFIRMATION_PROMPT
}

# ================== HELPER FUNCTIONS ==================

def phase_complete(session_state):
    """Check if current phase has all required data"""
    phase = session_state["phase"]
    if phase > 4:
        return True, []
    
    required = PHASE_REQUIREMENTS.get(phase, [])
    data = session_state["phase_data"].get(f"phase_{phase}", {})
    missing = [f for f in required if not data.get(f)]
    return len(missing) == 0, missing

def store_extracted(session_state, extracted):
    """Store extracted data into the current phase"""
    phase = session_state["phase"]
    phase_key = f"phase_{phase}"
    
    if phase_key not in session_state["phase_data"]:
        session_state["phase_data"][phase_key] = {}
    
    for k, v in extracted.items():
        if v and v != "null" and v != None:
            session_state["phase_data"][phase_key][k] = v

def generate_5_day_plan(session_state):
    """Generate 5-day task plan from collected data"""
    phase1 = session_state["phase_data"].get("phase_1", {})
    phase2 = session_state["phase_data"].get("phase_2", {})
    phase3 = session_state["phase_data"].get("phase_3", {})
    phase4 = session_state["phase_data"].get("phase_4", {})
    
    problem = phase1.get("problem", "social difficulties")
    context = phase1.get("context", "various settings")
    skill = phase2.get("skill_gaps", "conversation skills")
    tip = phase2.get("tips", "Start with observation-based comments")
    locations = phase3.get("locations", ["coffee shop", "gym"])
    schedule = phase3.get("schedule", "weekday mornings")
    anxiety = phase3.get("anxiety_issues", "fear of judgment")
    
    # Use Phase 4 optimal times if available
    optimal_times = phase4.get("optimal_times", [])
    analyzed_schedule = phase4.get("analyzed_schedule", {})
    
    days = []
    tasks = []
    start_date = datetime.now()

    # Progressive difficulty tasks
    difficulty_progression = [
        ("Warm-up", "Smile and make eye contact with 3 people", "5 minutes"),
        ("Ice-breaker", "Give one genuine compliment to a stranger", "10 minutes"),
        ("Mini-conversation", "Ask someone a simple question (time, directions, recommendation)", "15 minutes"),
        ("Extended chat", f"Have a 2-minute conversation using the technique: {tip}", "20 minutes"),
        ("Full practice", "Initiate and maintain a 5+ minute conversation", "30 minutes")
    ]

    for d in range(1, 6):
        day_date = start_date + timedelta(days=d-1)
        day_level, day_task_template, est_time = difficulty_progression[d-1]
        
        # Use optimal time if available, otherwise default
        if d <= len(optimal_times):
            optimal = optimal_times[d-1]
            location = optimal.get("location", locations[0] if locations else "coffee shop")
            scheduled_time = optimal.get("time", "09:00")
        else:
            location = locations[0] if locations else "coffee shop"
            scheduled_time = "09:00"
        
        day_tasks = []
        task_text = day_task_template
        full_description = f"📍 {location} | 🎯 {task_text} | 💡 Remember: {tip} | 😰 If anxious: {anxiety}"
        
        task_id = f"day{d}_task"
        tasks.append({
            "id": task_id,
            "title": f"Day {d} - {day_level}",
            "description": full_description,
            "done": False,
            "xp": d * 10,
            "scheduled_time": scheduled_time,
            "type": "social_practice",
            "location": location,
            "estimatedTime": est_time,
            "comfortLevel": "challenging" if d >= 4 else "moderate" if d >= 2 else "easy",
            "contextAnchor": context,
            "skill_focus": skill,
            "difficulty": d
        })
        
        day_tasks.append({
            "task_number": 1, 
            "description": full_description, 
            "done": False
        })
        
        days.append({
            "day": d,
            "date": day_date.strftime("%Y-%m-%d"),
            "title": f"Day {d}: {day_level}",
            "tasks": day_tasks,
            "completed": False,
            "difficulty": d,
            "focus": day_level
        })
    
    return {
        "days": days, 
        "tasks": tasks,
        "user_context": {
            "problem": problem,
            "skill_gaps": skill,
            "practice_locations": locations,
            "schedule": schedule,
            "primary_anxiety": anxiety,
            "analyzed_schedule": analyzed_schedule
        }
    }

def write_to_firebase(session_state):
    """Save completed session data to Firebase"""
    if not db:
        # Mock behavior if db is not initialized
        return "mock_doc_id", generate_5_day_plan(session_state)
    
    user_id = session_state["user_id"]
    created_at = datetime.utcnow().isoformat()
    task_overview = generate_5_day_plan(session_state)

    # Note: Using the non-private path for this example, assuming data sharing/public dashboard might be desired
    doc_ref = db.collection(f"artifacts/{app.config.get('APP_ID', 'default-app-id')}/public/data/courses").document()
    
    doc_ref.set({
        "user_id": user_id,
        "created_at": created_at,
        "phase_data": session_state["phase_data"],
        "task_overview": task_overview,
        "status": "active",
        "completion_rate": 0
    })
    return doc_ref.id, task_overview

def extract_json_from_response(text):
    """Extract JSON from LLM response that might have markdown or extra text"""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None

# ================== ENDPOINTS ==================

@app.route("/init-session", methods=["POST"])
def init_session():
    """Initialize a new session"""
    data = request.json
    session_id = data.get("session_id")
    user_id = data.get("user_id", "anonymous")
    
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    # Initialize session
    sessions[session_id] = {
        "phase": 1,
        "user_id": user_id,
        "phase_data": {f"phase_{i}": {} for i in range(1, 6)},
        "messages": [],
        "forms_completed": []
    }
    
    welcome_msg = "Hey! I'm Jordan. I used to be that person who'd rehearse conversations in the shower, then freeze when actually talking to people. Took me years to figure this out. Ready to start?"
    
    return jsonify({
        "success": True,
        "message": welcome_msg,
        "phase": 1,
        "session_id": session_id
    })

@app.route("/submit-phase-data", methods=["POST"])
def submit_phase_data():
    """Process frontend form data with AI analysis and extraction"""
    data = request.json
    session_id = data.get("session_id")
    phase = data.get("phase")
    form_data = data.get("form_data")
    api_key = data.get("api_key")
    
    # 1. Input Validation
    if not session_id or session_id not in sessions:
        return jsonify({"error": "invalid session_id"}), 400
    
    if not api_key:
        return jsonify({"error": "api_key required"}), 400
    
    if not form_data:
        return jsonify({"error": "form_data required"}), 400
    
    session_state = sessions[session_id]
    
    try:
        # 2. Get the appropriate prompt and prepare data
        prompt_template_text = PHASE_PROMPTS.get(phase, PHASE_1_PROMPT)
        
        # Prepare the dynamic data as JSON strings (no need for manual double-brace escaping)
        form_data_str = json.dumps(form_data, indent=2)
        prev_data_str = json.dumps(session_state["phase_data"], indent=2)
        
        # 3. Define the LLM Context Template
        # We use template placeholders {{variable_name}} for dynamic data
        # to cleanly separate them from the LLM's required JSON output.
        context_template = f"""
{{prompt_text}}

USER'S FORM SUBMISSION FOR PHASE {{phase_num}}:
{{form_data_str}}

PREVIOUSLY COLLECTED DATA:
{{prev_data_str}}

Analyze the form data, extract the required information according to your phase instructions, and respond in the specified JSON format.
"""
        
        # 4. Create the ChatPromptTemplate
        # The template itself is a single System Message
        prompt = ChatPromptTemplate.from_messages([
            ("system", context_template)
        ])
        
        # 5. Invoke LLM with ALL required variables filled in
        # We pass the template text itself as a variable
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=api_key
        )
        
        llm_output = llm.invoke(prompt.format_messages(
            prompt_text=prompt_template_text, # Inject the phase-specific instructions here
            phase_num=phase,
            form_data_str=form_data_str,
            prev_data_str=prev_data_str
        ))
        
        # 6. Parse and Process Response
        parsed = extract_json_from_response(llm_output.content)
        
        if not parsed:
            return jsonify({
                "error": "Failed to parse AI response",
                "raw_response": llm_output.content
            }), 500
        
        # Extract and store data
        extracted_data = parsed.get("extracted_data", {})
        if extracted_data:
            store_extracted(session_state, extracted_data)
        
        # Store the form completion
        session_state["forms_completed"].append(phase)
        
        # Get response message
        response_msg = parsed.get("message", "Data received.")
        
        # Check if ready to advance
        ready_for_next = parsed.get("ready_for_next_phase", False)
        
        if ready_for_next and phase < 5:
            session_state["phase"] = phase + 1
            next_phase = phase + 1
        else:
            next_phase = phase
        
        # 7. Success Response
        return jsonify({
            "success": True,
            "response": response_msg,
            "phase": next_phase,
            "phase_data": session_state["phase_data"],
            "extracted_data": extracted_data,
            "ready_for_next_phase": ready_for_next
        })
    
    except Exception as e:
        # 8. Error Handling
        full_traceback = traceback.format_exc()
        print("--- /submit-phase-data FULL TRACEBACK ---")
        print(full_traceback)
        print("------------------------------------------")
        
        return jsonify({
            "error": "Backend processing failed: An unexpected error occurred.",
            "details": str(e),
            "traceback": full_traceback
        }), 500
        

@app.route("/chat", methods=["POST"])
def chat():
    """Main conversational endpoint with Jordan AI"""
    data = request.json
    session_id = data.get("session_id")
    user_message = data.get("message", "")
    api_key = data.get("api_key")
    user_id = data.get("user_id", "anonymous")

    if not session_id or not api_key:
        return jsonify({"error": "session_id and api_key required"}), 400

    # Initialize session if new
    if session_id not in sessions:
        sessions[session_id] = {
            "phase": 1,
            "user_id": user_id,
            "phase_data": {f"phase_{i}": {} for i in range(1, 6)},
            "messages": [],
            "forms_completed": []
        }

    session_state = sessions[session_id]
    phase = session_state["phase"]
    
    # Phase 5: Confirmation conversation
    if phase == 5:
        # Check if user is confirming the plan
        if any(word in user_message.lower() for word in ["yes", "looks good", "let's do it", "confirm", "correct", "yep", "yeah"]):
            try:
                doc_id, task_overview = write_to_firebase(session_state)
                session_state["phase"] = 6  # Complete
                
                return jsonify({
                    "response": "🎉 Let's fucking go! Your 5-day plan is locked in. I'll be checking in on you. Day 1 starts tomorrow - no backing out now. You've got this.",
                    "phase": 6,
                    "plan_generated": True,
                    "course_id": doc_id,
                    "task_overview": task_overview,
                    "complete": True
                })
            
            except Exception as e:
                # 🚨 UPDATED ERROR HANDLING BLOCK 🚨
                full_traceback = traceback.format_exc()
                print("--- /chat (Phase 5) FULL TRACEBACK ---")
                print(full_traceback)
                print("------------------------------------------")
                
                return jsonify({
                    "error": "Failed to generate plan",
                    "details": str(e),
                    "traceback": full_traceback
                }), 500
        
        # User wants to modify something
        else:
            return jsonify({
                "response": "No worries. What do you want to change? Your schedule, locations, or the skill we're focusing on?",
                "phase": 5,
                "phase_data": session_state["phase_data"],
                "awaiting_modification": True
            })
    
    # Phase 6: Plan already generated
    if phase == 6:
        return jsonify({
            "response": "Your plan is already generated! Check your dashboard to see your 5-day roadmap.",
            "phase": 6,
            "complete": True
        })
    
    # Phases 1-4: Regular chat with AI agent
    try:
        # Store user message
        session_state["messages"].append({"role": "user", "content": user_message})
        
        # Get the appropriate prompt for current phase
        prompt_text = PHASE_PROMPTS.get(phase, PHASE_1_PROMPT)
        
        # Build context with collected data
        context_data = {
            "phase": phase,
            "collected_data": session_state["phase_data"],
            "conversation_history": session_state["messages"][-5:],  # Last 5 messages
            "user_message": user_message
        }

        # --- FIX: Escape curly braces in JSON output to prevent KeyErrors in f-string ---
        collected_data_str = json.dumps(session_state["phase_data"], indent=2).replace('{', '{{').replace('}', '}}')
        conversation_history_str = json.dumps(context_data["conversation_history"], indent=2).replace('{', '{{').replace('}', '}}')
        # --- END FIX ---

        context = f"""
CURRENT PHASE: {phase}

COLLECTED DATA SO FAR:
{collected_data_str}

RECENT CONVERSATION:
{conversation_history_str}

USER'S LATEST MESSAGE: {user_message}

Respond according to your phase instructions.
"""
        
        # Call LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=api_key
        )
        
        full_prompt = f"{prompt_text}\n\n{context}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", full_prompt)
        ])
        
        llm_output = llm.invoke(prompt.format_messages())
        parsed = extract_json_from_response(llm_output.content)
        
        if not parsed:
            # Fallback if JSON parsing fails
            return jsonify({
                "response": llm_output.content,
                "phase": phase,
                "phase_data": session_state["phase_data"]
            })
        
        # Store AI response
        session_state["messages"].append({
            "role": "assistant", 
            "content": parsed.get("message", llm_output.content)
        })
        
        # Check if phase is ready to advance
        if parsed.get("ready_for_next_phase"):
            session_state["phase"] = phase + 1
            return jsonify({
                "response": parsed.get("message"),
                "phase": session_state["phase"],
                "phase_data": session_state["phase_data"],
                "phase_complete": True
            })
        
        # Continue current phase
        return jsonify({
            "response": parsed.get("message"),
            "phase": phase,
            "phase_data": session_state["phase_data"],
            "needs_more_info": parsed.get("needs_more_info", True)
        })
    
    except Exception as e:
        # 🚨 UPDATED ERROR HANDLING BLOCK 🚨
        full_traceback = traceback.format_exc()
        print("--- /chat FULL TRACEBACK ---")
        print(full_traceback)
        print("------------------------------------------")
        
        return jsonify({
            "error": "AI processing failed",
            "details": str(e),
            "traceback": full_traceback, # <-- THIS IS THE KEY DETAIL TO FIND THE BUG
            "fallback_response": "Sorry, I hit a snag. Can you rephrase that?"
        }), 500
        
@app.route("/get-session-status", methods=["POST"])
def get_session_status():
    """Get current session status"""
    data = request.json
    session_id = data.get("session_id")
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "invalid session_id"}), 404
    
    session_state = sessions[session_id]
    
    return jsonify({
        "session_id": session_id,
        "phase": session_state["phase"],
        "phase_data": session_state["phase_data"],
        "forms_completed": session_state["forms_completed"],
        "message_count": len(session_state["messages"])
    })

@app.route("/get-plan/<course_id>", methods=["GET"])
def get_plan(course_id):
    """Retrieve a generated plan from Firebase"""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        
        if not db:
            return jsonify({"error": "Firebase not initialized"}), 500
        
        doc_ref = db.collection("users").document(user_id).collection("courses").document(course_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Plan not found"}), 404
        
        return jsonify(doc.to_dict())
    
    except Exception as e:
        return jsonify({
            "error": "Failed to retrieve plan",
            "details": str(e)
        }), 500

@app.route("/update-task", methods=["POST"])
def update_task():
    """Mark a task as complete and update progress"""
    data = request.json
    user_id = data.get("user_id")
    course_id = data.get("course_id")
    task_id = data.get("task_id")
    completed = data.get("completed", True)
    
    if not all([user_id, course_id, task_id]):
        return jsonify({"error": "user_id, course_id, task_id required"}), 400
    
    try:
        if not db:
            return jsonify({"error": "Firebase not initialized"}), 500
        
        doc_ref = db.collection("users").document(user_id).collection("courses").document(course_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Course not found"}), 404
        
        course_data = doc.to_dict()
        task_overview = course_data.get("task_overview", {})
        tasks = task_overview.get("tasks", [])
        
        # Update task status
        for task in tasks:
            if task.get("id") == task_id:
                task["done"] = completed
                break
        
        # Calculate completion rate
        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t.get("done"))
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Update Firebase
        doc_ref.update({
            "task_overview.tasks": tasks,
            "completion_rate": completion_rate,
            "last_updated": datetime.utcnow().isoformat()
        })
        
        return jsonify({
            "success": True,
            "completion_rate": completion_rate,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks
        })
    
    except Exception as e:
        return jsonify({
            "error": "Failed to update task",
            "details": str(e)
        }), 500

@app.route("/reset-session", methods=["POST"])
def reset_session():
    """Reset a session (for testing or user restart)"""
    data = request.json
    session_id = data.get("session_id")
    
    if session_id and session_id in sessions:
        del sessions[session_id]
        return jsonify({"success": True, "message": "Session reset"})
    
    return jsonify({"error": "Session not found"}), 404

if __name__ == "__main__":
    # Ensure APP_ID is set if not already done via environment config
    app.config['APP_ID'] = os.environ.get('APP_ID', 'default-app-id')
    app.run(debug=True, port=5000)
    


@app.route('/api/judge-story', methods=['POST', 'OPTIONS'])
def judge_story():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    story_text = data.get("storyText", "").strip()
    scenario = data.get("scenario", "").strip()
    scenario_context = data.get("scenarioContext", "").strip()
    
    # Validation
    if not user_id:
        return jsonify({"error": "Missing required field: user_id"}), 400
    
    if not story_text or len(story_text) < 50:
        return jsonify({"error": "Story must be at least 50 characters"}), 400
    
    if not scenario:
        return jsonify({"error": "Missing required field: scenario"}), 400
    
    try:
        # Load prompt template for story judging
        try:
            with open("prompt_story_judge.txt", "r") as f:
                judge_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_story_judge.txt not found"}), 500
        
        # Build system prompt
        system_prompt = judge_prompt_template.format(
            scenario=scenario,
            scenario_context=scenario_context,
            story_text=story_text
        )
        
        # Call LLM to analyze the story
        messages = [{"role": "system", "content": system_prompt}]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=2500
        )
        
        analysis_text = response.choices[0].message.content.strip()
        
        # Parse the response into structured format
        analysis_data = parse_story_analysis(analysis_text)
        
        # Validate that we got proper analysis
        if not analysis_data or "overallScore" not in analysis_data:
            return jsonify({"error": "Failed to parse AI analysis"}), 500
        
        # Save analysis to Firestore
        db.collection("users").document(user_id).collection("storyJudgments").add({
            "story_text": story_text,
            "scenario": scenario,
            "scenario_context": scenario_context,
            "analysis": analysis_data,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            "success": True,
            "analysis": analysis_data
        }), 200
        
    except Exception as e:
        print(f"Error in judge_story: {str(e)}")
        return jsonify({"error": str(e)}), 500
        


@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        user_message = data.get("message", "").strip()
        chat_step = data.get("chatStep", 0)
        conversation_id = data.get("conversationId", "")
        skill_name = data.get("skill_name", "genuine-appreciation")  # Default skill
        
        # Validation
        if not user_id or not user_message:
            return jsonify({"error": "Missing user_id or message"}), 400
        
        # Get API key from Authorization header
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[len("Bearer "):].strip()
        
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401
        
        # Initialize client with provided API key
        client.api_key = api_key
        
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = f"conv_{user_id}_{int(time.time())}"
        
        # Load conversation history from Firebase
        doc_ref = db.collection("chat_conversations").document(conversation_id)
        doc = doc_ref.get()
        
        if doc.exists:
            history = doc.to_dict().get("messages", [])
        else:
            # First time: load the appreciation coach prompt
            prompt_template = load_prompt("prompt_appreciation_coach.txt")
            if not prompt_template:
                return jsonify({"error": "prompt_appreciation_coach.txt not found"}), 500
            
            # Inject skill context into the prompt
            system_prompt = prompt_template.format(
                skill_name=skill_name,
                user_name=data.get("userName", "there")
            )
            history = [{"role": "system", "content": system_prompt}]
        
        # Add context reminder based on chat step
        step_context = get_step_context(chat_step, skill_name)
        context_message = {
            "role": "system",
            "content": f"Current step: {chat_step}. {step_context}"
        }
        
        # Build full message list for the AI
        messages_for_model = [history[0], context_message] + history[1:]
        messages_for_model.append({"role": "user", "content": user_message})
        
        # Call the LLaMA / Groq model
        response = client.chat.completions.create(
            model="groq/compound",
            messages=messages_for_model,
            temperature=0.7,
            max_tokens=300
        )
        
        ai_message = response.choices[0].message.content.strip()
        
        # Determine next step and flow control
        next_step = chat_step
        should_continue_chat = True
        ready_for_scenarios = False
        
        # Check for transition signals in AI response
        if "ready to practice" in ai_message.lower() or "real scenario" in ai_message.lower():
            next_step = 3
            should_continue_chat = False
            ready_for_scenarios = True
        elif chat_step < 3:
            next_step = chat_step + 1
        
        # Append user + AI message to history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": ai_message})
        
        # Save updated conversation to Firebase
        doc_ref.set({
            "messages": history,
            "user_id": user_id,
            "skill_name": skill_name,
            "last_updated": firestore.SERVER_TIMESTAMP,
            "chat_step": next_step
        })
        
        # Return structured response
        return jsonify({
            "success": True,
            "data": {
                "reply": ai_message,
                "nextStep": next_step,
                "conversationId": conversation_id,
                "shouldContinueChat": should_continue_chat,
                "readyForScenarios": ready_for_scenarios,
                "timestamp": datetime.now().isoformat(),
                "promptType": get_prompt_type(chat_step),
                "metadata": {
                    "messageId": f"msg_{int(time.time())}",
                    "aiModel": "groq/compound",
                    "tokensUsed": response.usage.total_tokens if hasattr(response, 'usage') else None
                }
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": f"Unexpected error: {str(e)}",
                "retryable": True
            }
        }), 500


# Helper function: Get step-specific context
def get_step_context(chat_step, skill_name):
    """Returns context based on current chat step"""
    contexts = {
        0: f"User is sharing an initial example about {skill_name}. Ask them to identify specific qualities or actions.",
        1: "User has shared qualities/actions. Now ask how they could express this genuinely.",
        2: "User has practiced expression. Provide encouraging feedback and transition to scenarios.",
        3: "User is ready for scenario practice. Wrap up the conversation warmly."
    }
    return contexts.get(chat_step, "Continue the coaching conversation naturally.")


# Helper function: Get prompt type for frontend
def get_prompt_type(chat_step):
    """Maps chat step to prompt type"""
    types = {
        0: "greeting",
        1: "dig_deeper",
        2: "practice_expression",
        3: "transition_to_scenarios"
    }
    return types.get(chat_step, "general")


# Helper function: Load prompt file
def load_prompt(filename):
    """Load prompt template from file"""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', filename)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

@app.route('/api/generate-briefing', methods=['POST', 'OPTIONS'])
def generate_briefing():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    location = data.get("location", "").strip()
    time = data.get("time", "").strip()
    energy_level = data.get("energy_level", 3)
    confidence_level = data.get("confidence_level", 3)
    user_history = data.get("user_history", {})
    
    # Validation
    if not user_id or not location or not time:
        return jsonify({"error": "Missing required fields: user_id, location, time"}), 400
    
    try:
        # Fetch user's condensed profile for personalization
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        condensed_profile = user_doc.to_dict().get("condensed_profile", "")
        
        # Load prompt template
        try:
            with open("prompt_mission_briefing.txt", "r") as f:
                briefing_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_mission_briefing.txt not found"}), 500
        
        # Build system prompt with user context
        system_prompt = briefing_prompt_template.format(
            location=location,
            time=time,
            energy_level=energy_level,
            confidence_level=confidence_level,
            condensed_profile=condensed_profile,
            user_history=json.dumps(user_history)
        )
        
        # Call LLM to generate briefing
        messages = [{"role": "system", "content": system_prompt}]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        briefing_text = response.choices[0].message.content.strip()
        
        # Parse the response into structured format
        briefing_data = parse_briefing_response(briefing_text)
        
        # Save briefing to user's Firestore document
        db.collection("users").document(user_id).set(
            {
                "last_briefing": {
                    "location": location,
                    "time": time,
                    "energy_level": energy_level,
                    "confidence_level": confidence_level,
                    "briefing_data": briefing_data,
                    "created_at": firestore.SERVER_TIMESTAMP
                }
            },
            merge=True
        )
        
        return jsonify(briefing_data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENDPOINT 2: Regenerate Openers Only
# ============================================================================

@app.route('/api/regenerate-openers', methods=['POST', 'OPTIONS'])
def regenerate_openers():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    location = data.get("location", "").strip()
    confidence_level = data.get("confidence_level", 3)
    previous_openers = data.get("previous_openers", [])
    
    if not user_id or not location:
        return jsonify({"error": "Missing required fields: user_id, location"}), 400
    
    try:
        # Fetch user profile
        user_doc = db.collection("users").document(user_id).get()
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404
        
        condensed_profile = user_doc.to_dict().get("condensed_profile", "")
        
        # Load openers prompt
        try:
            with open("prompt_openers.txt", "r") as f:
                openers_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_openers.txt not found"}), 500
        
        system_prompt = openers_prompt_template.format(
            location=location,
            confidence_level=confidence_level,
            condensed_profile=condensed_profile,
            previous_opener_ids=",".join(previous_openers)
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.8,
            max_tokens=1200
        )
        
        openers_text = response.choices[0].message.content.strip()
        openers = parse_openers_response(openers_text)
        
        return jsonify({"openers": openers}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ENDPOINT 3: Save Favorite Opener
# ============================================================================

@app.route('/api/save-favorite-opener', methods=['POST', 'OPTIONS'])
def save_favorite_opener():
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    opener_id = data.get("opener_id")
    
    if not user_id or not opener_id:
        return jsonify({"error": "Missing required fields: user_id, opener_id"}), 400
    
    try:
        # Add opener to user's favorite_openers array
        db.collection("users").document(user_id).set(
            {
                "favorite_openers": firestore.ArrayUnion([opener_id]),
                "last_favorite_saved": firestore.SERVER_TIMESTAMP
            },
            merge=True
        )
        
        return jsonify({
            "success": True,
            "message": "Opener saved to favorites"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# HELPER FUNCTIONS: Parsing LLM Responses
# ============================================================================

def parse_briefing_response(text):
    """
    Parse the LLM response into structured briefing data.
    The prompt should instruct the LLM to return JSON.
    """
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback: return raw text in a structured format
            return {
                "venue_intel": {"raw_analysis": text},
                "openers": [],
                "scenarios": [],
                "conversation_flows": [],
                "cheat_sheet": text
            }
    except Exception as e:
        return {
            "error": "Failed to parse briefing",
            "raw_response": text
        }


def parse_openers_response(text):
    """
    Parse opener data from LLM response into structured format.
    """
    try:
        import re
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback: return empty list
            return []
    except Exception as e:
        return []



# ============================================================================
# OPTIONAL: Save Briefing Session for Analytics
# ============================================================================

@app.route('/api/save-briefing-session', methods=['POST', 'OPTIONS'])
def save_briefing_session():
    """
    Save user's briefing session for future learning and improvement.
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    session_data = data.get("session_data")  # outcomes, what worked, etc.
    
    if not user_id or not session_data:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        db.collection("users").document(user_id).collection("briefing_history").add({
            "session_data": session_data,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        
        return jsonify({
            "success": True,
            "message": "Session saved for future insights"
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "✅ Groq LLaMA 4 Scout Backend is running."

@app.route('/anxiety-chat', methods=['POST', 'OPTIONS'])
def anxiety_chat():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    try:
        data = request.get_json()
        user_id = data.get("user_id")
        conversation_id = data.get("conversation_id")
        message_type = data.get("message_type")
        context = data.get("context", {})
        user_input = context.get("user_input", "")
        
        if not user_id or not conversation_id or not message_type:
            return jsonify({"error": "Missing required fields"}), 400

        # Get API key from Authorization header
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[len("Bearer "):].strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401

        # Initialize client with provided API key
        client.api_key = api_key

        # Load conversation history from Firebase
        doc_ref = db.collection("anxiety_conversations").document(conversation_id)
        doc = doc_ref.get()

        if doc.exists:
            history = doc.to_dict().get("messages", [])
        else:
            # First time: load the anxiety reduction prompt
            try:
                with open("prompt_anxiety_reduction.txt", "r") as f:
                    system_prompt = f.read()
            except FileNotFoundError:
                return jsonify({"error": "prompt_anxiety_reduction.txt not found"}), 500
            
            history = [{"role": "system", "content": system_prompt}]

        # Build context-aware message based on message_type
        if message_type == "greeting":
            user_message = f"I'm about to have a {context.get('task', {}).get('type', 'social')} interaction. I'm feeling anxious."
        
        elif message_type == "exercise_recommendation":
            user_state = context.get('user_state', {})
            user_message = f"""Based on my current state:
- Anxiety level: {user_state.get('anxietyLevel', 3)}/5
- Energy level: {user_state.get('energyLevel', 3)}/5
- Main worry: {user_state.get('worry', 'unknown')}
- Interaction type: {context.get('task', {}).get('type', 'unknown')}

What exercises should I do to prepare? Respond with a supportive message and suggest exercises from: grounding, breathing, ai-chat, self-talk, physical."""
        
        elif message_type == "motivation":
            exercises_completed = context.get('exercise_history', [])
            user_message = f"I just completed {len(exercises_completed)} exercise(s): {', '.join(exercises_completed)}. Give me encouraging feedback!"
        
        elif message_type == "self_talk_generation":
            user_state = context.get('user_state', {})
            user_message = f"""Generate 4 personalized positive affirmations for someone who:
- Has anxiety level {user_state.get('anxietyLevel', 3)}/5
- Main worry: {user_state.get('worry', 'unknown')}
- About to have a {context.get('task', {}).get('type', 'social')} interaction

Format: Return ONLY a JSON array of 4 strings, nothing else."""
        
        elif message_type == "reflection_prompt":
            user_message = "I've completed my preparation exercises. Help me reflect on what I accomplished."
        
        elif message_type == "reflection_analysis":
            reflection = context.get('reflection', {})
            user_message = f"""I just reflected on my preparation:
- Anxiety before: {context.get('user_state', {}).get('anxietyLevel', 3)}/5
- Anxiety after: {reflection.get('finalAnxiety', 3)}/5
- Confidence: {reflection.get('finalConfidence', 3)}/5
- Exercises helped: {reflection.get('exercisesHelped', 'unknown')}

Give me encouraging analysis of my progress!"""
        
        elif message_type == "emergency_followup":
            user_message = "I just did a 60-second emergency breathing reset. Check in on me."
        
        elif message_type == "user_message":
            user_message = user_input
        
        else:
            user_message = user_input or "Help me with my anxiety."

        # Append user message to history
        history.append({"role": "user", "content": user_message})

        # Call the AI model
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.7 if message_type == "user_message" else 0.6,
            max_tokens=500 if message_type == "user_message" else 300
        )

        ai_reply = response.choices[0].message.content.strip()

        # Handle self-talk generation specially (extract JSON)
        suggestions = None
        if message_type == "self_talk_generation":
            try:
                import json
                # Try to extract JSON array from response
                if "[" in ai_reply and "]" in ai_reply:
                    json_start = ai_reply.index("[")
                    json_end = ai_reply.rindex("]") + 1
                    suggestions = json.loads(ai_reply[json_start:json_end])
                else:
                    # Fallback: split by newlines or bullets
                    suggestions = [line.strip("- •") for line in ai_reply.split("\n") if line.strip()][:4]
            except:
                suggestions = [
                    "I am capable and prepared.",
                    "It's okay to feel nervous.",
                    "I've handled situations like this before.",
                    "One step at a time is enough."
                ]

        # Append AI response to history
        history.append({"role": "assistant", "content": ai_reply})

        # Save updated conversation to Firebase
        doc_ref.set({
            "messages": history,
            "user_id": user_id,
            "last_updated": firestore.SERVER_TIMESTAMP
        }, merge=True)

        # Return response
        response_data = {"response": ai_reply}
        if suggestions:
            response_data["suggestions"] = suggestions

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ============ LIVE ACTION SUPPORT ENDPOINT ============
# ============ LIVE ACTION SUPPORT ENDPOINT ============
@app.route("/live-action-support", methods=['POST'])
def live_action_support():
    # ========== STEP 1: Parse Request ==========
    data = request.get_json()
    task_name = data.get("task_name", "").strip()
    user_id = data.get("user_id", "").strip()
    user_context = data.get("user_context", {})
    
    if not task_name or not user_id:
        return jsonify({"error": "Missing task_name or user_id"}), 400
    
    # Extract user context
    anxiety_level = user_context.get("anxiety_level", "moderate")
    experience = user_context.get("experience", "beginner")
    specific_challenges = user_context.get("specific_challenges", [])
    category = data.get("category", "General Social")
    difficulty = data.get("difficulty", "Medium")
    
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not api_key:
        return jsonify({"error": "Missing API key in Authorization header"}), 401
    client.api_key = api_key
    
    # ========== STEP 2: Load User Profile for Personalization ==========
    user_profile = None
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if user_doc.exists:
            user_profile = user_doc.to_dict()
            print(f"✅ Loaded user profile for personalization")
    except Exception as e:
        print(f"⚠️ Could not load user profile: {e}")
        user_profile = {}
    
    # ========== STEP 3: Load Prompt Template ==========
    # NOTE: Assuming load_prompt and other dependencies (db, client, jsonify, request, json, datetime) are defined elsewhere
    prompt_file = "prompt_live_action_task.txt"
    prompt_template = load_prompt(prompt_file) 
    if not prompt_template:
        return jsonify({"error": f"{prompt_file} not found"}), 404
    
    # Format challenges for prompt
    formatted_challenges = "\n".join([f"- {c}" for c in specific_challenges]) if specific_challenges else "- General social anxiety"
    
    # Replace placeholders
    prompt = (prompt_template
              .replace("<<task_name>>", task_name)
              .replace("<<anxiety_level>>", anxiety_level)
              .replace("<<experience>>", experience)
              .replace("<<specific_challenges>>", formatted_challenges)
              .replace("<<category>>", category)
              .replace("<<difficulty>>", difficulty))
    
    # Add user profile context if available
    if user_profile:
        user_stats = {
            "success_rate": user_profile.get("success_rate", 0),
            "completed_tasks": user_profile.get("completed_tasks", 0),
            "preferred_time": user_profile.get("preferred_time", "morning")
        }
        # Assuming 'json' module is available for dumping stats
        prompt += f"\n\nUser Statistics:\n{json.dumps(user_stats, indent=2)}"
    
    # ========== STEP 4: Generate AI Task Structure (FIX APPLIED HERE) ==========
    result = "" # Initialize result for scope outside try block
    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=6000
        )
        result = response.choices[0].message.content.strip()

        # 🔥 FIX: Remove Markdown code fences before parsing JSON
        if result.startswith("```json"):
            # Remove the leading "```json\n" and the trailing "\n```" (or just "```")
            result = result.replace("```json\n", "", 1).strip().rstrip("`")
        elif result.startswith("```"):
            # Handle cases where the language tag is missing (e.g., just "```")
            result = result.replace("```\n", "", 1).strip().rstrip("`")
        
        # Ensure only the JSON object remains
        if result.endswith('```'):
            result = result.rstrip('`').strip()

        parsed_task = json.loads(result)
        print(f"✅ Live action task structure generated from AI")
    except json.JSONDecodeError:
        # Include the cleaned 'result' string for better debugging if the clean failed
        return jsonify({"error": "Failed to parse task structure as JSON", "raw_response": response.choices[0].message.content.strip(), "cleaned_result": result}), 500
    except Exception as e:
        return jsonify({"error": f"API request failed", "exception": str(e)}), 500
    
    # ========== STEP 5: Transform to App Structure ==========
    expected_keys = {
        "title": ["title", "task_title", "name"],
        "category": ["category", "type"],
        "difficulty": ["difficulty", "level"],
        "description": ["description", "overview"],
        "totalSteps": ["totalSteps", "total_steps", "step_count"],
        "estimatedTime": ["estimatedTime", "estimated_time", "duration"],
        "xpReward": ["xpReward", "xp_reward", "xp"],
        "prerequisites": ["prerequisites", "required_tasks"],
        "tags": ["tags", "keywords"],
        "steps": ["steps", "step_list"],
        "relatedTasks": ["relatedTasks", "related_tasks"],
        "aiMetadata": ["aiMetadata", "ai_metadata", "metadata"]
    }
    
    task_data = {}
    for key, alternatives in expected_keys.items():
        value = None
        for alt in alternatives:
            if alt in parsed_task:
                value = parsed_task[alt]
                break
        
        # Provide sensible defaults
        if value is None:
            if key == "steps":
                value = []
            elif key == "prerequisites" or key == "tags" or key == "relatedTasks":
                value = []
            elif key == "xpReward":
                value = 150
            elif key == "totalSteps":
                value = 5
            elif key == "estimatedTime":
                value = "15 min"
            elif key == "difficulty":
                value = difficulty
            elif key == "category":
                value = category
            elif key == "aiMetadata":
                value = {
                    "anxietyLevel": anxiety_level,
                    "skillsTargeted": [],
                    "commonChallenges": specific_challenges,
                    "recommendedTimeOfDay": []
                }
            else:
                value = ""
        task_data[key] = value
    
    # ========== STEP 6: Process and Validate Steps ==========
    raw_steps = task_data.get("steps", [])
    formatted_steps = []
    
    for idx, step in enumerate(raw_steps):
        if isinstance(step, dict):
            formatted_step = {
                "id": idx + 1,
                "title": step.get("title", f"Step {idx + 1}"),
                "description": step.get("description", ""),
                "tips": step.get("tips", []),
                "examples": step.get("examples", []),
                "aiCoaching": step.get("aiCoaching", step.get("ai_coaching", "")),
                "xp": step.get("xp", 30),
                "media": step.get("media", {
                    "videoUrl": None,
                    "imageUrl": None,
                    "audioUrl": None
                }),
                "successCriteria": step.get("successCriteria", step.get("success_criteria", []))
            }
            formatted_steps.append(formatted_step)
    
    task_data["steps"] = formatted_steps
    task_data["totalSteps"] = len(formatted_steps)
    
    # Calculate total XP if not provided
    if task_data["xpReward"] == 150:  # Default value
        task_data["xpReward"] = sum(step.get("xp", 30) for step in formatted_steps)
    
    # ========== STEP 7: Generate Unique Task ID ==========
    # NOTE: Assuming 'datetime' module is available
    task_id = f"{user_id}_{task_name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
    task_data["id"] = task_id
    task_data["created_at"] = datetime.now().isoformat()
    task_data["user_id"] = user_id
    
    # ========== STEP 8: Save to Firebase ==========
    # NOTE: Assuming 'db' (Firebase client) is available
    try:
        # Save to user's live action tasks collection
        task_ref = db.collection('users').document(user_id).collection('live_action_tasks').document(task_id)
        task_ref.set(task_data)
        print(f"✅ Saved to: users/{user_id}/live_action_tasks/{task_id}")
        
        # Also update user's task library (shared tasks)
        library_ref = db.collection('task_library').document(task_id)
        library_data = task_data.copy()
        library_data["shared"] = False
        library_data["creator_id"] = user_id
        library_ref.set(library_data)
        print(f"✅ Added to task library: task_library/{task_id}")
        
    except Exception as e:
        return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500
    
    # ========== STEP 9: Return Response ==========
    return jsonify({
        "success": True,
        "task_id": task_id,
        "task": task_data,
        "message": f"Live action task '{task_name}' created successfully"
    })


# ============ HELPER FUNCTION FOR DIFFICULTY ==========
def determine_difficulty(task_text):
    """Determine difficulty based on task description"""
    task_lower = task_text.lower()
    
    if any(word in task_lower for word in ['lead', 'present', 'speak to group', 'public']):
        return 'Hard'
    elif any(word in task_lower for word in ['conversation', 'share', 'ask question']):
        return 'Medium'
    else:
        return 'Easy'


# ============ TASK LIST OVERVIEW ENDPOINT ============
@app.route('/create-task-overview', methods=['POST'])
def create_task_overview():
    """
    Creates a high-level overview of tasks from Day 1 to Day 5
    Returns a structured list of all tasks across the 5-day journey
    """
    # ========== STEP 1: Parse Request ==========
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    goal_name = data.get("goal_name", "").strip()
    user_answers = data.get("user_answers", [])
    user_id = data.get("user_id", "").strip()
    join_date_str = data.get("join_date")
    
    if not goal_name or not isinstance(user_answers, list) or not user_id:
        return jsonify({"error": "Missing or invalid goal_name, user_answers, or user_id"}), 400

    try:
        joined_date = datetime.strptime(join_date_str, "%Y-%m-%d") if join_date_str else datetime.now()
    except:
        joined_date = datetime.now()
    
    course_id = goal_name.lower().replace(" ", "_")

    # Escape user inputs
    safe_goal_name = json.dumps(goal_name)[1:-1]
    safe_user_answers = json.dumps(user_answers)
    
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not api_key:
        return jsonify({"error": "Missing API key in Authorization header"}), 401
    client.api_key = api_key

    # ========== STEP 2: Load Task Overview Prompt ==========
    prompt_file = "prompt_task_overview.txt"
    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        return jsonify({"error": f"{prompt_file} not found"}), 404

    # Insert user inputs
    prompt = prompt_template.replace("<<goal_name>>", safe_goal_name)
    prompt = prompt.replace("<<user_answers>>", safe_user_answers)

    # ========== STEP 3: Generate Task Overview from AI ==========
    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=4096
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": "API request failed", "exception": str(e)}), 500

    # Extract JSON
    import re
    def extract_json(text: str):
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    parsed_overview = extract_json(result)
    if not parsed_overview:
        return jsonify({"error": "Failed to parse task overview as valid JSON", "raw_response": result}), 500
    
    print("✅ Task overview generated from AI")

    # ========== STEP 4: Structure and Validate Data ==========
    # Expected structure: {"days": [{"day": 1, "date": "...", "title": "...", "tasks": [...]}, ...]}
    if "days" not in parsed_overview or not isinstance(parsed_overview["days"], list):
        return jsonify({"error": "Invalid response structure - missing 'days' array"}), 500

    # Add dates to each day
    for i, day_data in enumerate(parsed_overview["days"]):
        day_number = day_data.get("day", i + 1)
        day_date = (joined_date + timedelta(days=day_number - 1)).strftime("%Y-%m-%d")
        day_data["date"] = day_date

    # ========== STEP 5: Save to Firebase ==========
    try:
        course_ref = get_course_ref(user_id, course_id)
        
        # Save as a separate document for quick access
        task_overview_data = {
            'goal_name': goal_name,
            'created_at': datetime.now().isoformat(),
            'task_overview': parsed_overview,
            'course_id': course_id
        }
        
        course_ref.set(task_overview_data, merge=True)
        print("✅ Task overview saved to Firebase")
        
    except Exception as e:
        return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500

    # ========== STEP 6: Return Response ==========
    return jsonify({
        "success": True,
        "course_id": course_id,
        "overview": parsed_overview,
        "message": "5-day task overview created successfully"
    })


@app.route('/reply-day-chat-advanced', methods=['POST', 'OPTIONS'])
def reply_day_chat_advanced():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight
    
    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message", "").strip()
    goal_name = data.get("goal_name", "").strip()
    user_interests = data.get("user_interests", [])

    if not user_id or not message:
        return jsonify({"error": "Missing input"}), 400

    # Get API key from Authorization header
    api_key = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[len("Bearer "):].strip()

    if not api_key:
        return jsonify({"error": "Missing API key in Authorization header"}), 401

    # Create a new client instance with the user's API key
    user_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    # ----------------------
    # FETCH EXISTING PLACES FROM FIREBASE
    # ----------------------
    user_doc_ref = db.collection("users").document(user_id)
    user_doc = user_doc_ref.get()
    
    existing_current_places = []
    existing_desired_places = []
    
    if user_doc.exists:
        user_data = user_doc.to_dict()
        existing_current_places = user_data.get("current_places", [])
        existing_desired_places = user_data.get("desired_places", [])

    # ----------------------
    # FETCH OR CREATE CHAT
    # ----------------------
    chats = db.collection("users").document(user_id).collection("custom_day_chat")
    docs = list(chats.order_by("day", direction=firestore.Query.DESCENDING).limit(1).stream())
    
    if not docs:
        # CREATE NEW CHAT AUTOMATICALLY
        new_chat_ref = chats.document()
        new_chat_ref.set({
            "day": firestore.SERVER_TIMESTAMP,
            "chat": []
        })
        chat_history = []
        doc_ref = new_chat_ref
    else:
        doc_ref = docs[0].reference
        chat_data = docs[0].to_dict()
        chat_history = chat_data.get("chat", [])

    # Append user message
    chat_history.append({"role": "user", "content": message})

    # Load chat prompt
    try:
        with open("prompt_DAYONE_COMPONENTONE.txt", "r") as f:
            chat_prompt_template = f.read()
    except FileNotFoundError:
        return jsonify({"error": "prompt_DAYONE_COMPONENTONE.txt not found"}), 500

    # Inject user-specific info into the prompt
    system_prompt = chat_prompt_template.format(
        goal_name=goal_name or "their personal goal",
        user_places=", ".join(existing_current_places) if existing_current_places else "none",
        user_interests=", ".join(user_interests) if user_interests else "none",
        user_desired_places=", ".join(existing_desired_places) if existing_desired_places else "none"
    )

    # Build messages - handle empty chat_history case
    if len(chat_history) > 1:
        # If there's prior history, insert system prompt after first message
        context_message = {"role": "system", "content": system_prompt}
        messages_for_model = [chat_history[0]] + [context_message] + chat_history[1:]
    else:
        # First message - just use system prompt + user message
        messages_for_model = [
            {"role": "system", "content": system_prompt},
            chat_history[0]
        ]

    try:
        # Generate AI chat reply
        response = user_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages_for_model,
            temperature=0.6,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()

        # Append AI response
        chat_history.append({"role": "assistant", "content": reply})
        doc_ref.update({"chat": chat_history})

        # ----------------------
        # EXTRACT PLACES using extraction prompt file
        # ----------------------
        try:
            with open("prompt_PLACE_EXTRACTION.txt", "r") as f:
                extraction_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_PLACE_EXTRACTION.txt not found"}), 500

        extraction_prompt = extraction_prompt_template.format(
            user_message=message
        )

        extraction_response = user_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": extraction_prompt}],
            temperature=0.2,
            max_tokens=200
        )
        extraction_text = extraction_response.choices[0].message.content.strip()

        # Parse extraction
        newly_extracted_current = []
        newly_extracted_desired = []
        
        try:
            # Clean markdown code blocks
            if "```json" in extraction_text:
                extraction_text = extraction_text.split("```json")[1].split("```")[0].strip()
            elif "```" in extraction_text:
                extraction_text = extraction_text.split("```")[1].split("```")[0].strip()
            
            extraction_data = json.loads(extraction_text)
            newly_extracted_current = extraction_data.get("current_places", [])
            newly_extracted_desired = extraction_data.get("desired_places", [])
            
        except json.JSONDecodeError as e:
            print(f"Extraction parse error: {e}")
            print(f"Raw extraction response: {extraction_text}")

        # ----------------------
        # Merge with existing places (avoid duplicates, case-insensitive)
        # ----------------------
        def merge_places(existing, new):
            existing_lower = [p.lower() for p in existing]
            merged = existing.copy()
            for place in new:
                if place.lower() not in existing_lower:
                    merged.append(place)
            return merged
        
        updated_current_places = merge_places(existing_current_places, newly_extracted_current)
        updated_desired_places = merge_places(existing_desired_places, newly_extracted_desired)

        # ----------------------
        # Generate condensed profile using profile prompt file
        # ----------------------
        try:
            with open("prompt_PROFILE_GENERATION.txt", "r") as f:
                profile_prompt_template = f.read()
        except FileNotFoundError:
            return jsonify({"error": "prompt_PROFILE_GENERATION.txt not found"}), 500

        profile_prompt = profile_prompt_template.format(
            chat_history=json.dumps(chat_history, indent=2)
        )

        profile_response = user_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": profile_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        profile_text = profile_response.choices[0].message.content.strip()

        # Parse profile
        profile_data = {}
        try:
            if "```json" in profile_text:
                profile_text = profile_text.split("```json")[1].split("```")[0].strip()
            elif "```" in profile_text:
                profile_text = profile_text.split("```")[1].split("```")[0].strip()
            
            profile_data = json.loads(profile_text)
        except json.JSONDecodeError as e:
            print(f"Profile parse error: {e}")
            print(f"Raw profile response: {profile_text}")
            profile_data = {"social_habits": "", "interests": [], "personality": ""}

        # ----------------------
        # Save everything to Firebase
        # ----------------------
        user_doc_ref.set({
            "current_places": updated_current_places,
            "desired_places": updated_desired_places,
            "condensed_profile": profile_data,
            "social_habits": profile_data.get("social_habits", ""),
            "interests": profile_data.get("interests", []),
            "personality": profile_data.get("personality", ""),
            "comfort_level": profile_data.get("comfort_level", ""),
            "last_updated": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        return jsonify({
            "reply": reply,
            "extracted_this_turn": {
                "current_places": newly_extracted_current,
                "desired_places": newly_extracted_desired
            },
            "total_places": {
                "current_places": updated_current_places,
                "desired_places": updated_desired_places
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/generate-user-places', methods=['POST', 'OPTIONS'])
def generate_user_places():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight
    
    data = request.get_json()
    user_id = data.get("user_id")
    goal_name = data.get("goal_name", "").strip()
    
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    
    # Get API key from Authorization header
    api_key = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[len("Bearer "):].strip()

    if not api_key:
        return jsonify({"error": "Missing API key in Authorization header"}), 401

    # Create a new client instance with the user's API key
    user_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
    
    # Fetch user data including places and profile
    user_doc = db.collection("users").document(user_id).get()
    
    if not user_doc.exists:
        return jsonify({"error": "User not found or profile not generated yet"}), 404
    
    user_data = user_doc.to_dict()
    
    # CRITICAL: Fetch the places we extracted
    current_places = user_data.get("current_places", [])
    desired_places = user_data.get("desired_places", [])
    condensed_profile = user_data.get("condensed_profile", "")
    
    if not condensed_profile:
        return jsonify({"error": "Condensed profile is empty. User needs to chat first."}), 404
    
    # Check if user has provided enough information
    if not current_places and not desired_places:
        return jsonify({
            "error": "No places extracted yet. User needs to share more about where they go and want to go."
        }), 404
    
    # Load location prompt
    try:
        with open("prompt_location.txt", "r") as f:
            location_prompt_template = f.read()
    except FileNotFoundError:
        return jsonify({"error": "prompt_location.txt not found"}), 500
    
    # Inject user info into location prompt INCLUDING PLACES
    system_prompt = location_prompt_template.format(
        goal_name=goal_name or "their personal goal",
        condensed_profile=json.dumps(condensed_profile) if isinstance(condensed_profile, dict) else condensed_profile,
        user_current_places=", ".join(current_places) if current_places else "none provided",
        user_desired_places=", ".join(desired_places) if desired_places else "none provided"
    )
    
    messages_for_model = [{"role": "system", "content": system_prompt}]
    
    try:
        response = user_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages_for_model,
            temperature=0.7,  # Increased for more creative location suggestions
            max_tokens=1500   # Increased to allow full JSON response with 3 locations
        )
        
        suggested_places = response.choices[0].message.content.strip()
        
        # Save suggested places back to user doc
        db.collection("users").document(user_id).set(
            {
                "suggested_places": suggested_places,
                "places_generated_at": firestore.SERVER_TIMESTAMP
            },
            merge=True
        )
        
        return jsonify({
            "suggested_places": suggested_places,
            "used_data": {
                "current_places": current_places,
                "desired_places": desired_places
            }
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        


CONVERSATION_STATES = [
    "context",       # Context & Current Life Snapshot
    "habits",        # Habits & Daily Patterns
    "social",        # Social Circle & Interactions
    "obstacles",     # Obstacles & Pain Points
    "resources",     # Resources & Support
    "motivation",    # Motivation & Desired Outcome
    "final_goal"     # Goal Confirmation
]

@app.route('/chat12', methods=['POST'])
def chat12_endpoint():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        user_message = data.get("message", "").strip()
        goal_name = data.get("goal_name", "").strip()

        if not user_id or not user_message:
            return jsonify({"error": "Missing user_id or message"}), 400

        # Get API key from Authorization header
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[len("Bearer "):].strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401

        client.api_key = api_key

        # Load conversation from Firebase
        doc_ref = db.collection("conversations").document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            doc_data = doc.to_dict()
            history = doc_data.get("messages", [])
            states = doc_data.get("states", {s: "" for s in CONVERSATION_STATES})
            current_state = doc_data.get("current_state", "context")
        else:
            prompt_template = load_prompt("prompt_setgoal.txt")
            if not prompt_template:
                return jsonify({"error": "prompt_setgoal.txt not found"}), 500
            system_prompt = prompt_template.format(goal_name=goal_name or "their personal goal")
            history = [{"role": "system", "content": system_prompt}]
            states = {s: "" for s in CONVERSATION_STATES}
            current_state = "context"

        # Append user message to history
        history.append({"role": "user", "content": user_message})

        # Send only the conversation history to the AI (last role is user)
        messages_for_model = history

        # Call the AI
        response = client.chat.completions.create(
            model="groq/compound",
            messages=messages_for_model,
            temperature=0.7,
            max_tokens=300
        )

        ai_message = response.choices[0].message.content.strip()

        # Save user's input as paragraph for current state
        states[current_state] = ai_message

        # Progress to next state if current input is sufficient
        current_index = CONVERSATION_STATES.index(current_state)
        if current_index < len(CONVERSATION_STATES) - 1:
            next_state = CONVERSATION_STATES[current_index + 1]
        else:
            next_state = current_state  # final state remains

        # Append AI message to history
        history.append({"role": "assistant", "content": ai_message})

        # Save to Firebase
        doc_ref.set({
            "messages": history,
            "states": states,
            "current_state": next_state
        })

        return jsonify({
            "reply": ai_message,
            "current_state": current_state,
            "next_state": next_state,
            "states": states
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500





@app.route("/mindpal-reward", methods=["POST"])
def mindpal_reward_webhook():
    data = request.get_json()
    user_id = data.get("user_id")
    rewards = data.get("rewards", [])

    if not user_id or not isinstance(rewards, list):
        return jsonify({"error": "Missing user_id or rewards[]"}), 400

    # 🔥 Save to: users/<user_id>/rewards/<auto_id>
    save_to_firebase(user_id, "rewards", {
        "source": "mindpal",
        "rewards": rewards
    })

    # ✅ Optionally also save to local file (if still needed)
    local_data = read_rewards()
    local_data[user_id] = {
        "reward_list": rewards,
        "source": "mindpal"
    }
    write_rewards(local_data)

    return jsonify({"status": "Reward saved successfully"}), 200





@app.route('/create-dated-course', methods=['POST'])
def create_dated_course():
    data = request.get_json()
    print("📥 Received payload:", data)  # Log incoming request

    user_id = data.get("user_id")
    final_plan = data.get("final_plan")
    join_date_str = data.get("join_date")  # Optional: user join date

    if not user_id or not final_plan:
        print("❌ Missing required data")
        return jsonify({"error": "Missing required data"}), 400

    # Parse join date
    try:
        joined_date = datetime.strptime(join_date_str, "%Y-%m-%d") if join_date_str else datetime.now()
        print("📅 Parsed join date:", joined_date)
    except Exception as e:
        print("⚠️ Failed to parse join date, using current date. Error:", e)
        joined_date = datetime.now()

    # Convert final_plan into a dated plan
    dated_plan = {}
    for i, day_key in enumerate(final_plan.get("final_plan", {}), start=0):
        date_str = (joined_date + timedelta(days=i)).strftime("%Y-%m-%d")
        day_data = final_plan["final_plan"][day_key].copy()

        # Convert tasks into toggle-ready objects
        tasks_with_toggle = [{"task": t, "done": False} for t in day_data.get("tasks", [])]
        day_data["tasks"] = tasks_with_toggle

        dated_plan[date_str] = day_data

    print("📝 Dated plan prepared:", dated_plan)

    # Save to Firebase
    try:
        course_id = "social_skills_101"  # You can make this dynamic
        doc_path = f"dated_courses/{user_id}/{course_id}"
        print("📌 Writing to Firestore at:", doc_path)

        db.document(doc_path).set({
            "joined_date": joined_date.strftime("%Y-%m-%d"),
            "lessons_by_date": dated_plan
        })

        print("✅ Write successful")
        return jsonify({"success": True, "dated_plan": dated_plan})

    except Exception as e:
        print("❌ Failed to write to Firestore:", e)
        return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500



@app.route('/toggle-task', methods=['POST'])
def toggle_task():
    data = request.get_json()
    user_id = data.get("user_id")
    day = data.get("day")
    task_index = data.get("task_index")
    completed = data.get("completed")

    if user_id is None or day is None or task_index is None or completed is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Reference to user's task document for the day
    task_doc_ref = db.collection("users").document(user_id).collection("task_status").document(f"day_{day}")
    task_doc = task_doc_ref.get()

    if task_doc.exists:
        task_data = task_doc.to_dict()
        tasks_completed = task_data.get("tasks_completed", [])
    else:
        # Initialize if not exists
        tasks_completed = []

    # Ensure the tasks_completed array has enough slots
    while len(tasks_completed) <= task_index:
        tasks_completed.append(False)

    # Update the specific task's completion
    tasks_completed[task_index] = completed

    # Save back to Firestore
    task_doc_ref.set({
        "tasks_completed": tasks_completed,
        "timestamp": datetime.utcnow()
    })

    # Calculate daily progress
    total_tasks = len(tasks_completed)
    completed_count = sum(1 for t in tasks_completed if t)
    daily_progress = completed_count / total_tasks if total_tasks > 0 else 0

    return jsonify({
        "day": day,
        "task_index": task_index,
        "completed": completed,
        "daily_progress": daily_progress,
        "tasks_completed": tasks_completed
    })

if __name__ == "__main__":
    app.run(debug=True)


@app.route('/support-room-question', methods=['POST'])
def support_room_question():
    data = request.get_json()
    user_id = data.get("user_id")
    task = data.get("task", "").strip()
    question = data.get("question", "").strip()

    if not task or not question:
        return jsonify({"error": "Missing task or question"}), 400

    prompt_template = load_prompt("prompt_support_room.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_support_room.txt not found"}), 500

    prompt = (
        prompt_template
        .replace("<<task>>", task)
        .replace("<<question>>", question)
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        result = response.choices[0].message.content.strip()

        # Optionally: save in Firestore
        save_to_firebase(user_id, "support_room_responses", {
            "task": task,
            "question": question,
            "response": result
        })

        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/rescue-plan-chat-answers', methods=['POST'])
def rescue_plan_chat_answers():
    data = request.get_json()
    user_id = data.get("user_id")
    task = data.get("task")
    answers = data.get("answers")  # list of 7 answers

    # ✅ Basic validation
    if not user_id or not task or not answers or not isinstance(answers, list):
        return jsonify({"error": "Missing or invalid data"}), 400

    try:
        # ✅ Save to Firestore
        save_to_firebase(user_id, "rescue_chat_answers", {
            "task": task,
            "answers": answers
        })

        return jsonify({"status": "success", "message": "Answers saved ✅"}), 200

    except Exception as e:
        print("❌ Error saving rescue chat answers:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/generate-action-level-questions', methods=['POST'])
def generate_action_level_questions():
    data = request.get_json()
    user_id = data.get("user_id", "")

    prompt_template = load_prompt("prompt_action_level_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_action_level_questions.txt not found"}), 500

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.4,
            max_tokens=400
        )
        result = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse questions JSON", "raw": result}), 500

        save_to_firebase(user_id, "action_level_questions", {
            "questions": parsed.get("questions", [])
        })

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


@app.route('/rescue-plan-chat-start', methods=['POST'])
def rescue_plan_chat_start():
    data = request.get_json()
    task = data.get("task", "")
    user_id = data.get("user_id", "")

    if not task:
        return jsonify({"error": "Missing task"}), 400

    prompt_template = load_prompt("prompt_rescue_chat_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_rescue_chat_questions.txt not found"}), 500

    prompt = prompt_template.replace("<<task>>", task)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)

        save_to_firebase(user_id, "rescue_chat_questions", {
            "task": task,
            "questions": parsed.get("questions", [])
        })

        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate-rescue-kit', methods=['POST', 'OPTIONS'])
@cross_origin()
def generate_rescue_kit():
    if request.method == "OPTIONS":
        # Preflight request for CORS
        return '', 200

    try:
        data = request.get_json()
        user_id = data.get("userId")  # ✅ match frontend key (camelCase)
        task = data.get("task", "")
        risks = data.get("risks", [])  # list of strings
        reward = data.get("reward", "")  # optional

        if not task or not risks:
            return jsonify({"error": "Missing task or risks"}), 400

        risks_formatted = "\n".join([f"- {r}" for r in risks])

        prompt_template = load_prompt("prompt_rescue_kit.txt")
        if not prompt_template:
            return jsonify({"error": "prompt_rescue_kit.txt not found"}), 500

        prompt = (
            prompt_template
            .replace("<<task>>", task)
            .replace("<<risks>>", risks_formatted)
            .replace("<<reward>>", reward)
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=700
        )
        result = response.choices[0].message.content.strip()

        parsed = json.loads(result)

        save_to_firebase(user_id, "rescue_kit", {
            "task": task,
            "risks": risks,
            "reward": reward,
            "rescue_plans": parsed.get("plans", [])
        })

        return jsonify(parsed)
    
    except Exception as e:
        print("❌ Backend error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-action-level', methods=['POST'])
def analyze_action_level():
    data = request.get_json()
    user_id = data.get("user_id")
    answers = data.get("answers", [])

    if not user_id or not isinstance(answers, list) or not answers:
        return jsonify({"error": "Missing or invalid user_id or answers"}), 400

    formatted_answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])

    prompt_template = load_prompt("prompt_analyze_action_level.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_analyze_action_level.txt not found"}), 500

    prompt = prompt_template.replace("<<userlevelanswers>>", formatted_answers)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        result = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse JSON", "raw_response": result}), 500

        # Store result in Firebase
        save_to_firebase(user_id, "action_level_analysis", {
            "answers": answers,
            "analysis": parsed
        })

        return jsonify(parsed)

    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


@app.route('/achievement-summary', methods=['POST'])
def achievement_summary():
    data = request.get_json()
    user_id = data.get("user_id")
    plan = data.get("plan")  # The user's plan input (likely a dict)

    if not user_id or not plan:
        return jsonify({"error": "Missing user_id or plan"}), 400

    prompt_template = load_prompt("prompt_achievement_summary.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_achievement_summary.txt not found"}), 500

    # Inject the plan JSON into your prompt template
    prompt = prompt_template.replace("<<plan>>", json.dumps(plan, indent=2))

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600
        )
        achievement_text = response.choices[0].message.content.strip()

        # Optionally save achievement summary to Firebase
        save_to_firebase(user_id, "achievement_summaries", {
            "plan": plan,
            "achievement_summary": achievement_text
        })

        return jsonify({"achievement_summary": achievement_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/start-day-chat', methods=['POST', 'OPTIONS'])
def start_day_chat():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    data = request.get_json()
    user_id = data.get("user_id")
    day_number = data.get("day_number")
    sections = data.get("subsections", [])

    if not user_id or not day_number or not isinstance(sections, list):
        return jsonify({"error": "Invalid input"}), 400

    prompt_template = load_prompt("prompt_customize_day.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_customize_day.txt not found"}), 500

    formatted_sections = "\n".join([f"- {s}" for s in sections])
    prompt = (
        prompt_template
        .replace("<<day_number>>", str(day_number))
        .replace("<<subsections>>", formatted_sections)
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        msg = response.choices[0].message.content.strip()

        chat_data = {
            "day": day_number,
            "sections": sections,
            "chat": [{"role": "assistant", "content": msg}]
        }

        save_to_firebase(user_id, "custom_day_chat", chat_data)
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------- REPLY DAY CHAT ---------

@app.route('/reply-day-chat', methods=['POST', 'OPTIONS'])
def reply_day_chat():
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight

    data = request.get_json()
    user_id = data.get("user_id")
    message = data.get("message")

    if not user_id or not message:
        return jsonify({"error": "Missing input"}), 400

    chats = db.collection("users").document(user_id).collection("custom_day_chat")
    docs = list(chats.order_by("day", direction=firestore.Query.DESCENDING).limit(1).stream())
    if not docs:
        return jsonify({"error": "Chat not started"}), 404

    doc_ref = docs[0].reference
    chat_data = docs[0].to_dict()
    chat_history = chat_data.get("chat", [])

    chat_history.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=chat_history,
            temperature=0.5,
            max_tokens=500
        )
        reply = response.choices[0].message.content.strip()
        chat_history.append({"role": "assistant", "content": reply})

        doc_ref.update({"chat": chat_history})
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/finalize-day-chat', methods=['POST'])
def finalize_day_chat():
    data = request.get_json()
    user_id = data.get("user_id")
    user_data = data.get("user_data")
    ogplan = data.get("ogplan")

    if not user_id or not user_data or not ogplan:
        return jsonify({"error": "Missing required data"}), 400

    chats = db.collection("users").document(user_id).collection("custom_day_chat")
    docs = list(chats.order_by("day", direction=firestore.Query.DESCENDING).limit(1).stream())
    if not docs:
        return jsonify({"error": "No chat session found"}), 404

    chat = docs[0].to_dict()
    chat_history = chat.get("chat", [])
    day_number = chat.get("day")

    finalize_prompt = load_prompt("prompt_customize_day_finalize.txt")
    if not finalize_prompt:
        return jsonify({"error": "prompt_customize_day_finalize.txt not found"}), 500

    final_instruction = (
        finalize_prompt
        .replace("<<user_data>>", json.dumps(user_data, indent=2))
        .replace("<<ogplan>>", json.dumps(ogplan, indent=2))
        .replace("<<day_number>>", str(day_number))
    )

    chat_history.append({"role": "user", "content": final_instruction})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=chat_history,
            temperature=0.4,
            max_tokens=4000
        )
        final_output = response.choices[0].message.content.strip()

        # Remove ```json or ``` wrapping from the AI response
        cleaned_output = re.sub(r"^```(?:json)?|```$", "", final_output.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(cleaned_output)
        except json.JSONDecodeError as json_err:
            return jsonify({
                "error": "Failed to parse final JSON",
                "raw": final_output,
                "cleaned": cleaned_output,
                "details": str(json_err)
            }), 500

        final_data = {
            "day": day_number,
            "final_plan": parsed
        }

        save_to_firebase(user_id, "custom_day_final_plans", final_data)
        return jsonify({"final_plan": parsed})
    
    except Exception as e:
        return jsonify({"error": f"Backend error: {str(e)}"}), 500

@app.route("/get-ogplan", methods=["POST"])
def get_ogplan():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        plans = db.collection("users").document(user_id).collection("plans")
        docs = list(plans.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream())
        if not docs:
            return jsonify({"error": "No plan found"}), 404

        plan_data = docs[0].to_dict().get("ai_plan")
        return jsonify({"ogplan": plan_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask-questions', methods=['POST'])
def ask_questions():
    data = request.get_json()
    goal_name = data.get("goal_name", "").strip()
    user_id = data.get("user_id")

    if not goal_name:
        return jsonify({"error": "Missing goal_name"}), 400

    prompt_template = load_prompt("prompt_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_questions.txt not found"}), 500

    prompt = prompt_template.format(goal_name=goal_name)

    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        result = response.choices[0].message.content.strip()
        save_to_firebase(user_id, "questions", {
            "goal_name": goal_name,
            "questions": result
        })
        return jsonify({"questions": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ============ HELPER FUNCTIONS ============

def load_prompt(filename):
    """Load prompt template from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def get_course_ref(user_id, course_id):
    """Get reference to the course document"""
    return db.collection('users').document(user_id).collection('datedcourses').document(course_id)

def determine_difficulty(task_text):
    """Determine task difficulty based on keywords"""
    lower_task = task_text.lower()
    if any(word in lower_task for word in ['review', 'reflect', 'schedule', 'take a few minutes', 'read']):
        return 'easy'
    elif any(word in lower_task for word in ['practice', 'connect', 'reach out', 'write', 'try']):
        return 'medium'
    else:
        return 'hard'

# ============ MAIN ENDPOINT CREATOR ============
# ============ MAIN ENDPOINT CREATOR (FIXED) ============
def create_day_endpoint(day):
    endpoint_name = f"final_plan_day_{day}"
    route_path = f"/final-plan-day{day}"
    
    @app.route(route_path, methods=['POST'], endpoint=endpoint_name)
    def final_plan_day_func():
        # ========== STEP 1: Parse Request ==========
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        goal_name = data.get("goal_name", "").strip()
        user_answers = data.get("user_answers", [])
        user_id = data.get("user_id", "").strip()
        join_date_str = data.get("join_date")
        
        if not goal_name or not isinstance(user_answers, list) or not user_id:
            return jsonify({"error": "Missing or invalid goal_name, user_answers, or user_id"}), 400

        try:
            joined_date = datetime.strptime(join_date_str, "%Y-%m-%d") if join_date_str else datetime.now()
        except:
            joined_date = datetime.now()
        
        day_date = (joined_date + timedelta(days=day-1)).strftime("%Y-%m-%d")
        course_id = goal_name.lower().replace(" ", "_")

        # Escape user inputs to avoid breaking JSON
        safe_goal_name = json.dumps(goal_name)[1:-1]  # strip surrounding quotes
        safe_user_answers = json.dumps(user_answers)
        
        formatted_answers = "\n".join(
            [f"{i+1}. {answer.strip()}" for i, answer in enumerate(user_answers) if isinstance(answer, str)]
        )
        
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401
        client.api_key = api_key

        # ========== STEP 2: Load Previous Day ==========
        previous_day_lesson = None
        if day > 1:
            try:
                course_ref = get_course_ref(user_id, course_id)
                course_doc = course_ref.get()
                if course_doc.exists:
                    course_data = course_doc.to_dict()
                    lessons_by_date = course_data.get('lessons_by_date', {})
                    prev_day_date = (joined_date + timedelta(days=day-2)).strftime("%Y-%m-%d")
                    previous_day_lesson = lessons_by_date.get(prev_day_date)
                    print(f"✅ Loaded previous day ({prev_day_date}) for context")
            except Exception as e:
                print(f"⚠️ Could not load previous day: {e}")
                previous_day_lesson = None

        # ========== STEP 3: Load Prompt Template ==========
        prompt_file = f"prompt_plan_{day:02}.txt"
        prompt_template = load_prompt(prompt_file)
        if not prompt_template:
            return jsonify({"error": f"{prompt_file} not found"}), 404

        # Insert safely escaped user inputs
        prompt = prompt_template.replace("<<goal_name>>", safe_goal_name)
        prompt = prompt.replace("<<user_answers>>", safe_user_answers)
        if previous_day_lesson:
            placeholder = f"<<day_{day-1}_json>>"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, json.dumps(previous_day_lesson))

        # ========== STEP 4: Generate AI Plan ==========
        try:
            response = client.chat.completions.create(
                model="groq/compound",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=4096
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "API request failed", "exception": str(e)}), 500

        # Robust JSON extraction
        import re
        def extract_json(text: str):
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
            return None

        parsed_day_plan = extract_json(result)
        if not parsed_day_plan:
            return jsonify({"error": f"Failed to parse Day {day} as valid JSON", "raw_response": result}), 500
        print(f"✅ Day {day} plan generated from AI")

        # ========== STEP 5: Transform to App Structure ==========
        expected_keys = {
            "title": ["title", "day_title", "name"],
            "summary": ["summary", "overview", "description"],
            "lesson": ["lesson", "content", "instructions"],
            "motivation": ["motivation", "inspiration", "encouragement"],
            "why": ["why", "purpose", "importance"],
            "book_quote": ["book_quote", "citation"],
            "secret_hacks_and_shortcuts": ["secret_hacks_and_shortcuts", "tips", "hacks"],
            "self_coaching_questions": ["self_coaching_questions", "questions", "prompts"],
            "tiny_daily_rituals_that_transform": ["tiny_daily_rituals_that_transform", "rituals", "micro_habits"],
            "visual_infographic_html": ["visual_infographic_html", "infographic", "html"],
            "task": ["task", "tasks", "actions"]
        }

        lesson_data = {}
        for key, alternatives in expected_keys.items():
            value = None
            for alt in alternatives:
                if alt in parsed_day_plan:
                    value = parsed_day_plan[alt]
                    break
            # sensible defaults
            if value is None:
                if key == "task":
                    value = []
                elif key == "self_coaching_questions":
                    value = []
                elif key == "book_quote" or key == "motivation" or key == "summary" or key == "title":
                    value = ""
                else:
                    value = ""
            lesson_data[key] = value

        # Normalize tasks
        raw_tasks = lesson_data.get("task", [])
        if isinstance(raw_tasks, list):
            lesson_data["task"] = [
                {
                    "task_number": i+1,
                    "description": task if isinstance(task, str) else task.get("description", "")
                }
                for i, task in enumerate(raw_tasks[:3])
            ]
            # Ensure exactly 3 tasks
            while len(lesson_data["task"]) < 3:
                lesson_data["task"].append({"task_number": len(lesson_data["task"])+1, "description": ""})
        else:
            lesson_data["task"] = []

        # Add date and completion info
        lesson_data["date"] = day_date
        lesson_data["completed"] = False
        lesson_data["reflection"] = ""

        # ========== STEP 6: Save to Firebase ==========
        try:
            course_ref = get_course_ref(user_id, course_id)
            course_doc = course_ref.get()
            if course_doc.exists:
                course_data = course_doc.to_dict()
                lessons_by_date = course_data.get('lessons_by_date', {})
                lessons_by_date[day_date] = lesson_data
                course_ref.update({'lessons_by_date': lessons_by_date})
            else:
                course_ref.set({
                    'joined_date': joined_date.strftime("%Y-%m-%d"),
                    'goal_name': goal_name,
                    'lessons_by_date': {day_date: lesson_data},
                    'created_at': datetime.now().isoformat()
                })
            print(f"✅ Saved Day {day} to Firebase")
        except Exception as e:
            return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500

        # ========== STEP 7: Return Response ==========
        return jsonify({
            "success": True,
            "day": day,
            "date": day_date,
            "course_id": course_id,
            "lesson": lesson_data,
            "message": f"Day {day} lesson created successfully"
        })
    
    return final_plan_day_func

# ============ CREATE ALL ENDPOINTS ============
for i in range(1, 6):
    create_day_endpoint(i)


# ============ OPTIONAL: Batch Create All Days ==========
@app.route('/create-full-course', methods=['POST'])
def create_full_course():
    """Create all 5 days at once"""
    data = request.get_json()
    goal_name = data.get("goal_name", "").strip()
    user_answers = data.get("user_answers", [])
    user_id = data.get("user_id", "").strip()
    join_date_str = data.get("join_date")
    
    if not goal_name or not isinstance(user_answers, list) or not user_id:
        return jsonify({"error": "Missing required fields"}), 400
    
    results = []
    errors = []
    
    for day in range(1, 6):
        try:
            # Call each day endpoint internally
            endpoint_func = app.view_functions[f"final_plan_day_{day}"]
            # Note: This is simplified - in production, make actual HTTP calls
            results.append(f"Day {day} created")
        except Exception as e:
            errors.append(f"Day {day} failed: {str(e)}")
    
    return jsonify({
        "success": len(errors) == 0,
        "results": results,
        "errors": errors
    })

# ============ UTILITY: Get Course Progress ==========
@app.route('/get-course/<user_id>/<course_id>', methods=['GET'])
def get_course(user_id, course_id):
    """Get course data for debugging"""
    try:
        course_ref = get_course_ref(user_id, course_id)
        course_doc = course_ref.get()
        
        if not course_doc.exists:
            return jsonify({"error": "Course not found"}), 404
        
        return jsonify({
            "success": True,
            "data": course_doc.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)



@app.route('/start-ai-helper', methods=['POST'])
def start_ai_helper():
    data = request.get_json()
    ai_plan = data.get("ai_plan")
    user_id = data.get("user_id")

    if not isinstance(ai_plan, dict):
        return jsonify({"error": "Missing or invalid ai_plan"}), 400

    prompt_template = load_prompt("prompt_ai_helper_start.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_ai_helper_start.txt not found"}), 500

    prompt = prompt_template.replace("<<ai_plan>>", json.dumps(ai_plan, indent=2))

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        save_to_firebase(user_id, "ai_helper_starts", {
            "ai_plan": ai_plan,
            "ai_intro": result
        })
        return jsonify({"ai_intro": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
@app.route('/ai-helper-reply', methods=['POST'])
def ai_helper_reply():
    data = request.get_json()
    ai_plan = data.get("ai_plan")
    chat_history = data.get("chat_history", [])
    user_id = data.get("user_id")

    if not isinstance(ai_plan, dict) or not isinstance(chat_history, list):
        return jsonify({"error": "Missing or invalid ai_plan or chat_history"}), 400

    history_text = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history if isinstance(m, dict)]
    )

    prompt_template = load_prompt("prompt_ai_helper_reply.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_ai_helper_reply.txt not found"}), 500

    prompt = (
        prompt_template
        .replace("<<ai_plan>>", json.dumps(ai_plan, indent=2))
        .replace("<<chat_history>>", history_text)
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1500
        )
        result = response.choices[0].message.content.strip()

        save_to_firebase(user_id, "ai_helper_replies", {
            "ai_plan": ai_plan,
            "chat_history": chat_history,
            "ai_reply": result
        })

        return jsonify({"ai_reply": result})

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/daily-dashboard', methods=['POST'])
def daily_dashboard():
    data = request.get_json()
    day_number = data.get("day", 1)
    raw_html = data.get("goalplanner_saved_html", "")
    user_id = data.get("user_id")

    if not raw_html:
        return jsonify({"error": "Missing goalplanner_saved_html"}), 400

    soup = BeautifulSoup(raw_html, "html.parser")
    day_header = f"Skyler Day{day_number}"
    section = None

    for div in soup.find_all("div"):
        if day_header in div.text:
            section = div
            break

    if not section:
        return jsonify({"error": f"No content found for {day_header}"}), 404

    task_text = ""
    for p in section.find_all("p"):
        if p.find("strong") and "Task" in p.find("strong").text:
            task_text = p.text.replace("Task:", "").strip()
            break

    tasks = [t.strip() for t in task_text.split(",") if t.strip()]

    prompt_template = load_prompt("prompt_dashboard.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_dashboard.txt not found"}), 500

    prompt = (
        prompt_template
        .replace("<<day>>", str(day_number))
        .replace("<<tasks>>", json.dumps(tasks, indent=2))
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
        parsed = json.loads(result)

        save_to_firebase(user_id, "dashboards", {
            "day": day_number,
            "tasks": tasks,
            "dashboard": parsed
        })

        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON from model", "raw_response": result}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/get-user-logs', methods=['GET'])
def get_all_logs():
    logs = read_logs()
    return jsonify({"logs": logs})

@app.route('/generate-reward-questions', methods=['POST'])
def generate_reward_questions():
    data = request.get_json()
    user_id = data.get("user_id", "")

    prompt_template = load_prompt("prompt_reward_questions.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_reward_questions.txt not found"}), 500

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.5,
            max_tokens=400
        )
        questions = response.choices[0].message.content.strip()

        save_to_firebase(user_id, "reward_questions", {
            "questions": questions
        })

        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

@app.route('/analyze-reward', methods=['POST'])
def analyze_reward():
    data = request.get_json()
    user_id = data.get("user_id")
    answers = data.get("answers", [])

    if not user_id or not isinstance(answers, list) or len(answers) == 0:
        return jsonify({"error": "Missing user_id or answers"}), 400

    formatted_answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])

    prompt_template = load_prompt("prompt_reward_analysis.txt")
    if not prompt_template:
        return jsonify({"error": "prompt_reward_analysis.txt not found"}), 500

    prompt = prompt_template.replace("<<user_answers>>", formatted_answers)

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=200
        )
        reward = response.choices[0].message.content.strip()

        rewards = read_rewards()
        rewards[user_id] = {
            "reward": reward,
            "task_completed": False
        }
        write_rewards(rewards)

        save_to_firebase(user_id, "rewards", {
            "answers": answers,
            "reward": reward
        })

        return jsonify({"reward": reward})
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

@app.route('/claim-reward', methods=['GET'])
def claim_reward():
    user_id = request.args.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    rewards = read_rewards()
    if user_id not in rewards:
        return jsonify({"error": "No reward set for user"}), 404

    reward_data = rewards[user_id]

    return jsonify({"reward": reward_data.get("reward")})

@app.route('/complete-task', methods=['POST'])
def complete_task():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    rewards = read_rewards()
    if user_id not in rewards:
        return jsonify({"error": "User not found"}), 404

    rewards[user_id]["task_completed"] = True
    write_rewards(rewards)

    save_to_firebase(user_id, "task_completions", {
        "task_completed": True
    })

    return jsonify({"message": "Task marked complete. Reward unlocked!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



























































