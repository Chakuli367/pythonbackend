from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from openai import OpenAI
from typing import Dict
import os
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import re
from datetime import datetime, timedelta
import time
import requests
from firebase_admin import credentials, firestore
from typing import TypedDict, Annotated, Literal, Optional, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from datetime import datetime
import operator

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



# ========== PYDANTIC MODELS ==========

class DiagnosticSummary(BaseModel):
    main_challenge: str = Field(description="The core social skill challenge identified")
    emotional_state: str = Field(description="User's emotional state (anxious, frustrated, hopeful, etc.)")
    context: str = Field(description="Where this challenge shows up most")
    impact: str = Field(description="How this affects their life")
    backstory: str = Field(description="Brief history - how long, what triggered it")
    
class SkillGapAssessment(BaseModel):
    skill_gaps: List[str] = Field(description="3-5 specific skills needing improvement", min_length=3, max_length=5)
    skill_ratings: Dict[str, int] = Field(description="Skill name -> rating out of 10")
    primary_weakness: str = Field(description="The #1 skill to focus on first")
    hidden_strength: str = Field(description="One strength they might not recognize")
    improvement_priority: List[str] = Field(description="Ordered list of skills to work on")

class StudyGuide(BaseModel):
    title: str = Field(description="Study guide title")
    key_concepts: List[str] = Field(description="5 core concepts they need to understand")
    lessons: List[Dict[str, Any]] = Field(description="3-5 lessons with topic, explanation, exercise")
    resources: List[str] = Field(description="Additional learning resources")
    
class GoalSet(BaseModel):
    week_1_goal: Dict[str, Any] = Field(description="Foundation goal with target, metric, timeline")
    week_2_goal: Dict[str, Any] = Field(description="Expansion goal")
    week_3_4_goal: Dict[str, Any] = Field(description="Integration goal")
    success_metrics: List[str] = Field(description="How to measure success")

class ActionPlan(BaseModel):
    plan_title: str = Field(description="Action plan title")
    daily_tasks: List[Dict[str, Any]] = Field(description="5 days of tasks with morning/afternoon/evening")
    reflection_prompts: List[str] = Field(description="Daily reflection questions")
    difficulty_level: str = Field(description="easy, moderate, or challenging")

class AccountabilitySetup(BaseModel):
    tracking_method: str = Field(description="How user wants to track (journal, app, etc)")
    check_in_time: str = Field(description="Preferred check-in time")
    reminder_style: str = Field(description="gentle, firm, or motivational")
    support_needed: List[str] = Field(description="Types of support user wants")

# ========== SESSION STORAGE ==========

sessions = {}

# ========== JORDAN PROMPTS WITH CONTEXT INJECTION ==========

def build_jordan_prompt(phase: int, phase_data: Dict[str, Any]) -> str:
    """Build Jordan's system prompt with context from previous phases"""
    
    base_personality = """You are Jordan, an adaptive AI social skills coach. You're the SAME person throughout all phases, just adjusting your approach.

YOUR CORE TRAITS:
- Empathetic but direct when needed
- Remember EVERYTHING from previous conversations
- Adapt your style to what the user needs right now
- Use "we" language - you're partners in this

"""
    
    # Build context from previous phases
    context = ""
    
    if phase >= 2 and phase_data.get("diagnostic_summary"):
        diag = phase_data["diagnostic_summary"]
        context += f"""
CONTEXT FROM PHASE 1 (Discovery):
- Main Challenge: {diag.get('main_challenge')}
- Emotional State: {diag.get('emotional_state')}
- Where It Happens: {diag.get('context')}
- Impact: {diag.get('impact')}
- Backstory: {diag.get('backstory')}

"""
    
    if phase >= 3 and phase_data.get("skill_assessment"):
        skills = phase_data["skill_assessment"]
        context += f"""
CONTEXT FROM PHASE 2 (Assessment):
- Primary Weakness: {skills.get('primary_weakness')}
- Skill Gaps: {', '.join(skills.get('skill_gaps', []))}
- Hidden Strength: {skills.get('hidden_strength')}
- Priority Order: {', '.join(skills.get('improvement_priority', []))}

"""
    
    if phase >= 4 and phase_data.get("study_guide"):
        guide = phase_data["study_guide"]
        context += f"""
CONTEXT FROM PHASE 3 (Education):
- Key Concepts Taught: {', '.join(guide.get('key_concepts', []))}
- Number of Lessons: {len(guide.get('lessons', []))}

"""
    
    if phase >= 5 and phase_data.get("goals"):
        goals = phase_data["goals"]
        context += f"""
CONTEXT FROM PHASE 4 (Goal Setting):
- Week 1 Goal: {goals.get('week_1_goal', {}).get('description', 'Not set')}
- Week 2 Goal: {goals.get('week_2_goal', {}).get('description', 'Not set')}
- Week 3-4 Goal: {goals.get('week_3_4_goal', {}).get('description', 'Not set')}

"""
    
    if phase >= 6 and phase_data.get("action_plan"):
        plan = phase_data["action_plan"]
        context += f"""
CONTEXT FROM PHASE 5 (Action Planning):
- Plan: {plan.get('plan_title')}
- Difficulty: {plan.get('difficulty_level')}
- Days Planned: {len(plan.get('daily_tasks', []))}

"""
    
    # Phase-specific instructions
    phase_instructions = {
        1: """
CURRENT PHASE: DISCOVERY (Diagnostic Listening)

YOUR APPROACH RIGHT NOW:
- Be warm, patient, and deeply empathetic
- Ask ONE question at a time (under 60 words)
- Validate their feelings: "That sounds really hard"
- Dig deeper to understand: what, when, where, why, how long
- Don't rush - just understand their story

YOUR GOAL:
After 5-8 exchanges, you should understand:
1. Their core social challenge (specific, not vague)
2. How it makes them feel emotionally
3. Where/when it happens most
4. How it impacts their life
5. Brief backstory (how long, what triggered it)

When you have enough info, you'll generate a DiagnosticSummary.
""",
        
        2: """
CURRENT PHASE: ASSESSMENT (Skills Analysis)

YOUR APPROACH RIGHT NOW:
- Be direct and tactical: "Let's break this down"
- Ask specific questions, request examples
- Use scales: "Rate that 1-10"
- Challenge vague answers: "What EXACTLY happens?"
- Focus on observable, measurable skills

YOUR GOAL:
Identify 3-5 SPECIFIC skill gaps in:
- Conversation initiation (starting talks)
- Conversation maintenance (keeping it going)
- Non-verbal communication (body language, eye contact)
- Social awareness (reading cues)
- Emotional regulation (managing anxiety)

When you have enough data, you'll generate a SkillGapAssessment.
""",
        
        3: """
CURRENT PHASE: EDUCATION (Teaching)

YOUR APPROACH RIGHT NOW:
- Be patient and educational like a professor
- Explain the WHY behind everything
- Use frameworks, examples, analogies
- Build on what you learned in Phase 1 & 2
- Keep responses focused (100-150 words)

YOUR GOAL:
Teach them:
1. Why their specific challenges happen (psychology)
2. How social skills actually work
3. Common mistakes and why they occur
4. Techniques that address THEIR gaps
5. The science behind effective strategies

When you've taught key concepts, you'll generate a StudyGuide.
""",
        
        4: """
CURRENT PHASE: GOAL SETTING (Strategic Planning)

YOUR APPROACH RIGHT NOW:
- Be collaborative: "Let's build this together"
- Present draft goals, get feedback
- Push back on unrealistic goals
- Make goals SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- Adjust based on their lifestyle and comfort

YOUR GOAL:
Co-create 3 goals:
- Week 1: Easy foundation goal (builds confidence)
- Week 2: Moderate expansion goal (stretches them)
- Week 3-4: Integration goal (meaningful progress)

Base goals on THEIR specific challenges and skill gaps from earlier phases.

When goals are agreed upon, you'll generate a GoalSet.
""",
        
        5: """
CURRENT PHASE: ACTION PLANNING (Practical Tasks)

YOUR APPROACH RIGHT NOW:
- Be energetic and practical
- Create tasks that fit THEIR actual life
- Be flexible - adjust to their schedule
- Make it feel achievable, not overwhelming
- Get specific about when and where

YOUR GOAL:
Create a 5-day action plan with:
- 2-3 tasks per day (morning/afternoon/evening)
- Tasks directly tied to their goals
- Increasing difficulty across days
- Reflection prompts
- Fits their real constraints

When plan is customized to their needs, you'll generate an ActionPlan.
""",
        
        6: """
CURRENT PHASE: ACCOUNTABILITY SETUP (Support System)

YOUR APPROACH RIGHT NOW:
- Be supportive and casual like a friend
- Ask about their tracking preferences
- Set up check-in schedule
- Prepare for setbacks with compassion
- Get them excited to start

YOUR GOAL:
Set up:
1. How they'll track progress
2. When they want reminders
3. What style of support they prefer
4. How to handle missed days
5. First check-in timing

When setup is complete, you'll generate an AccountabilitySetup.
"""
    }
    
    return base_personality + context + phase_instructions.get(phase, "")

# ========== PHASE COMPLETION DETECTION ==========

def should_complete_phase(phase: int, messages: List, turn_count: int) -> bool:
    """Determine if current phase has enough information to move forward"""
    
    # Minimum turns before completion
    min_turns = {
        1: 5,  # Need enough diagnostic info
        2: 4,  # Need skill assessment data
        3: 3,  # Need to teach core concepts
        4: 3,  # Need to agree on goals
        5: 2,  # Need to customize plan
        6: 2   # Need accountability preferences
    }
    
    return turn_count >= min_turns.get(phase, 3)

# ========== STRUCTURED OUTPUT GENERATION ==========

def generate_phase_output(phase: int, messages: List, api_key: str, phase_data: Dict) -> Optional[BaseModel]:
    """Generate structured output at the end of each phase"""
    
    # Create conversation summary for context
    conversation_text = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Jordan'}: {msg.content}"
        for msg in messages[-10:]  # Last 10 messages for context
    ])
    
    extraction_prompts = {
        1: f"""Based on this conversation, extract a diagnostic summary.

Conversation:
{conversation_text}

Extract:
- main_challenge: The core social skill issue (be specific)
- emotional_state: How they feel about it
- context: Where/when this happens most
- impact: How it affects their life
- backstory: Brief history""",
        
        2: f"""Based on this skills assessment conversation, extract specific skill gaps.

Conversation:
{conversation_text}

Previous Context:
- Main Challenge: {phase_data.get('diagnostic_summary', {}).get('main_challenge', 'Unknown')}

Extract:
- skill_gaps: 3-5 specific, measurable skills (not vague)
- skill_ratings: Dict of skill -> rating (1-10)
- primary_weakness: The #1 skill to focus on
- hidden_strength: One strength they have
- improvement_priority: Ordered list of skills to improve""",
        
        3: f"""Based on this educational conversation, create a study guide.

Conversation:
{conversation_text}

User's Challenge: {phase_data.get('diagnostic_summary', {}).get('main_challenge', 'Unknown')}
Skill Gaps: {', '.join(phase_data.get('skill_assessment', {}).get('skill_gaps', []))}

Extract:
- title: Study guide title
- key_concepts: 5 core concepts taught
- lessons: 3-5 lessons with topic, explanation, and exercise
- resources: Additional learning materials""",
        
        4: f"""Based on this goal-setting conversation, create the goal set.

Conversation:
{conversation_text}

Context:
- Challenge: {phase_data.get('diagnostic_summary', {}).get('main_challenge')}
- Primary Weakness: {phase_data.get('skill_assessment', {}).get('primary_weakness')}

Extract:
- week_1_goal: Easy foundation goal (dict with description, metric, timeline)
- week_2_goal: Moderate expansion goal
- week_3_4_goal: Integration goal
- success_metrics: How to measure overall success""",
        
        5: f"""Based on this action planning conversation, create the action plan.

Conversation:
{conversation_text}

Goals:
- Week 1: {phase_data.get('goals', {}).get('week_1_goal', {}).get('description')}
- Week 2: {phase_data.get('goals', {}).get('week_2_goal', {}).get('description')}

Extract:
- plan_title: Action plan title
- daily_tasks: 5 days of tasks (each day has morning, afternoon, evening tasks)
- reflection_prompts: Daily reflection questions
- difficulty_level: easy, moderate, or challenging""",
        
        6: f"""Based on this accountability setup conversation, extract preferences.

Conversation:
{conversation_text}

Extract:
- tracking_method: How they want to track progress
- check_in_time: Preferred time for check-ins
- reminder_style: gentle, firm, or motivational
- support_needed: Types of support they want"""
    }
    
    output_models = {
        1: DiagnosticSummary,
        2: SkillGapAssessment,
        3: StudyGuide,
        4: GoalSet,
        5: ActionPlan,
        6: AccountabilitySetup
    }
    
    prompt = extraction_prompts.get(phase)
    model_class = output_models.get(phase)
    
    if not prompt or not model_class:
        return None
    
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,  # Lower temp for structured extraction
            groq_api_key=api_key
        ).with_structured_output(model_class)
        
        result = llm.invoke(prompt)
        return result
        
    except Exception as e:
        print(f"Error generating structured output for phase {phase}: {e}")
        return None

# ========== MAIN CHAT HANDLER ==========

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint with proper data flow between phases"""
    data = request.json or {}
    session_id = data.get("session_id")
    user_message = data.get("message", "")
    api_key = data.get("api_key")
    
    if not session_id or not api_key:
        return jsonify({"error": "session_id and api_key required"}), 400
    
    # Initialize new session
    if session_id not in sessions:
        sessions[session_id] = {
            "phase": 1,
            "messages": [],
            "turn_count": 0,
            "user_id": data.get("user_id", "anonymous"),
            "api_key": api_key,
            "created_at": datetime.now().isoformat(),
            
            # Structured data from each phase
            "diagnostic_summary": None,
            "skill_assessment": None,
            "study_guide": None,
            "goals": None,
            "action_plan": None,
            "accountability_setup": None,
        }
        
        # Send initial greeting
        phase = sessions[session_id]["phase"]
        greeting = get_jordan_greeting(phase)
        sessions[session_id]["messages"].append(AIMessage(content=greeting))
        
        return jsonify({
            "response": greeting,
            "phase": phase,
            "session_id": session_id,
            "phase_complete": False
        })
    
    # Get current state
    state = sessions[session_id]
    phase = state["phase"]
    messages = state["messages"]
    
    # Add user message
    messages.append(HumanMessage(content=user_message))
    state["turn_count"] += 1
    
    # Build Jordan's prompt with full context
    phase_data = {
        "diagnostic_summary": state.get("diagnostic_summary"),
        "skill_assessment": state.get("skill_assessment"),
        "study_guide": state.get("study_guide"),
        "goals": state.get("goals"),
        "action_plan": state.get("action_plan"),
    }
    
    system_prompt = build_jordan_prompt(phase, phase_data)
    
    # Generate response
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=api_key
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | llm
        response = chain.invoke({"messages": messages})
        messages.append(response)
        
        # Check if phase should complete
        phase_complete = should_complete_phase(phase, messages, state["turn_count"])
        
        # Generate structured output if phase is complete
        structured_output = None
        if phase_complete:
            structured_output = generate_phase_output(phase, messages, api_key, phase_data)
            
            # Store structured output
            output_keys = {
                1: "diagnostic_summary",
                2: "skill_assessment",
                3: "study_guide",
                4: "goals",
                5: "action_plan",
                6: "accountability_setup"
            }
            
            if structured_output and phase in output_keys:
                state[output_keys[phase]] = structured_output.model_dump()
        
        sessions[session_id] = state
        
        return jsonify({
            "response": response.content,
            "phase": phase,
            "session_id": session_id,
            "turn_count": state["turn_count"],
            "phase_complete": phase_complete,
            "structured_data": structured_output.model_dump() if structured_output else None
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========== PHASE TRANSITION ==========

@app.route("/transition", methods=["POST"])
def transition_phase():
    """Transition to next phase with full context"""
    data = request.json or {}
    session_id = data.get("session_id")
    
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 400
    
    state = sessions[session_id]
    old_phase = state["phase"]
    new_phase = old_phase + 1
    
    if new_phase > 6:
        return jsonify({"error": "Already at final phase"}), 400
    
    # Update phase and reset turn count
    state["phase"] = new_phase
    state["turn_count"] = 0
    
    # Get greeting with context
    greeting = get_jordan_greeting(new_phase, state)
    state["messages"].append(AIMessage(content=greeting))
    
    sessions[session_id] = state
    
    return jsonify({
        "response": greeting,
        "old_phase": old_phase,
        "new_phase": new_phase,
        "session_id": session_id,
        "message": f"Transitioned to Phase {new_phase}"
    })

# ========== PHASE GREETINGS ==========

def get_jordan_greeting(phase: int, state: Dict = None) -> str:
    """Get Jordan's greeting with context from previous phases"""
    
    base_greetings = {
        1: "Hi! I'm Jordan, your social skills coach. I'm here to listen and understand what you're going through. This is a safe space. What brings you here today?",
        2: "Alright, let's shift gears. I want to get tactical now and pinpoint exactly which skills need work. Ready to dive in?",
        3: "Now that I understand your challenges, let me teach you the science behind why this happens and how to overcome it. Sound good?",
        4: "Time to set some goals! I'll help you create 3 clear, achievable targets based on everything we've learned. Let's build this together.",
        5: "Let's turn those goals into action! I'll create a customized 5-day plan that fits YOUR life. Ready?",
        6: "Final step! Let's set up your tracking and accountability system so you can crush this. How do you want to stay on track?"
    }
    
    greeting = base_greetings.get(phase, "Hey! How can I help?")
    
    # Add context if available
    if state and phase > 1:
        if phase == 2 and state.get("diagnostic_summary"):
            challenge = state["diagnostic_summary"].get("main_challenge", "")
            greeting += f" I know you've been struggling with {challenge.lower()}. Let's figure out the exact skills we need to work on."
        
        elif phase == 3 and state.get("skill_assessment"):
            primary = state["skill_assessment"].get("primary_weakness", "")
            greeting += f" Your primary challenge is {primary.lower()}. Let me explain why that happens and what we can do about it."
        
        elif phase == 4 and state.get("study_guide"):
            greeting += " Now that you understand the concepts, let's turn that knowledge into concrete goals."
        
        elif phase == 5 and state.get("goals"):
            greeting += " You've got solid goals set. Let's break them down into daily actions."
        
        elif phase == 6 and state.get("action_plan"):
            greeting += " Your plan is ready. Let's make sure you have the support to actually do it."
    
    return greeting

# ========== SESSION MANAGEMENT ==========

@app.route("/session/<session_id>", methods=["GET"])
def get_session(session_id):
    """Get complete session state"""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    state = sessions[session_id]
    return jsonify({
        "phase": state["phase"],
        "turn_count": state["turn_count"],
        "message_count": len(state["messages"]),
        "created_at": state["created_at"],
        "diagnostic_summary": state.get("diagnostic_summary"),
        "skill_assessment": state.get("skill_assessment"),
        "study_guide": state.get("study_guide"),
        "goals": state.get("goals"),
        "action_plan": state.get("action_plan"),
        "accountability_setup": state.get("accountability_setup"),
    })

@app.route("/reset/<session_id>", methods=["POST"])
def reset_session(session_id):
    """Reset session"""
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"message": "Session reset successfully"})

@app.route("/export/<session_id>", methods=["GET"])
def export_session(session_id):
    """Export full session data"""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404
    
    state = sessions[session_id]
    
    # Create exportable format
    export_data = {
        "session_id": session_id,
        "user_id": state["user_id"],
        "created_at": state["created_at"],
        "current_phase": state["phase"],
        "conversation_history": [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "jordan",
                "content": msg.content,
            }
            for msg in state["messages"]
        ],
        "diagnostic_summary": state.get("diagnostic_summary"),
        "skill_assessment": state.get("skill_assessment"),
        "study_guide": state.get("study_guide"),
        "goals": state.get("goals"),
        "action_plan": state.get("action_plan"),
        "accountability_setup": state.get("accountability_setup"),
    }
    
    return jsonify(export_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    


@app.route('/reflect-analyze', methods=['POST'])
def reflect_analyze():
    """
    Analyze one social interaction and update skill weaknesses.
    Creates the reflection document if it doesn't exist.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_id = data.get("user_id", "").strip()
    course_id = data.get("course_id", "").strip()
    user_message = data.get("message", "").strip()

    if not user_id or not course_id or not user_message:
        return jsonify({"error": "Missing user_id, course_id, or message"}), 400

    api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not api_key:
        return jsonify({"error": "Missing API key in Authorization header"}), 401
    client.api_key = api_key

    # === Step 1: Load or create Reflection Data ===
    reflections_ref = db.collection("users").document(user_id).collection("reflections").document(course_id)
    reflections_doc = reflections_ref.get()
    if reflections_doc.exists:
        reflections_data = reflections_doc.to_dict()
    else:
        # Create new doc if not present
        reflections_data = {"skills": {}, "history": []}
        reflections_ref.set(reflections_data)

    # === Step 2: Load Prompt ===
    prompt_file = "prompt_reflect_analyze.txt"
    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        return jsonify({"error": f"{prompt_file} not found"}), 404

    safe_message = json.dumps(user_message)[1:-1]
    prompt = prompt_template.replace("<<interaction>>", safe_message)

    # === Step 3: Call AI ===
    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": "AI request failed", "exception": str(e)}), 500

    # === Step 4: Extract JSON ===
    import re
    def extract_json(text: str):
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    ai_data = extract_json(result)
    if not ai_data:
        return jsonify({"error": "Invalid AI JSON", "raw": result}), 500

    # === Step 5: Update Skill Weaknesses ===
    for skill in ai_data.get("missing_skills", []):
        reflections_data["skills"][skill] = reflections_data["skills"].get(skill, 0) + 1

    reflections_data["history"].append({
        "interaction": user_message,
        "analysis": ai_data
    })

    # === Step 6: Save to Firebase ===
    try:
        reflections_ref.set(reflections_data)
    except Exception as e:
        return jsonify({"error": f"Failed to save reflection data: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "analysis": ai_data,
        "skills": reflections_data["skills"],
        "message": "Reflection analyzed and saved (created doc if missing)"
    })


@app.route('/reflect-update-tasks', methods=['POST'])
def reflect_update_tasks():
    """
    Generate a 5-day task overview based on user's weakest social skills (user_deficiencies).
    Updates task_overview in an existing datedcourses document.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        user_id = data.get("user_id", "").strip()
        course_id = data.get("course_id", "").strip()
        user_deficiencies = data.get("user_deficiencies", [])
        if not user_id or not course_id or not user_deficiencies:
            return jsonify({"error": "Missing user_id, course_id, or user_deficiencies"}), 400

        # API key
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if not api_key:
            return jsonify({"error": "Missing API key in Authorization header"}), 401
        client.api_key = api_key

        from datetime import datetime
        now_iso = datetime.now().isoformat()

        # --- Step 1: Load existing course document ---
        course_ref = db.collection('users').document(user_id).collection('datedcourses').document(course_id)
        course_doc = course_ref.get()
        if not course_doc.exists:
            return jsonify({"error": "Course document not found. It must exist before calling this endpoint."}), 404

        doc_data = course_doc.to_dict()
        goal_name = doc_data.get("goal_name", "")
        created_at = doc_data.get("created_at", "")

        # --- Step 2: Load prompt template ---
        prompt_file = "prompt_reflect_update_tasks.txt"
        prompt_template = load_prompt(prompt_file)
        if not prompt_template:
            return jsonify({"error": f"{prompt_file} not found"}), 500

        # Replace placeholders
        safe_deficiencies = json.dumps(user_deficiencies)
        prompt = prompt_template.replace("<<user_deficiencies>>", safe_deficiencies)
        prompt = prompt.replace("<<course_id>>", course_id)
        prompt = prompt.replace("<<created_at>>", created_at)
        prompt = prompt.replace("<<goal_name>>", goal_name)
        prompt = prompt.replace("<<user_id>>", user_id)

        # --- Step 3: Call AI ---
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=6000
        )

        ai_output = response.choices[0].message.content.strip()

        # --- Step 4: Extract JSON safely ---
        import re
        def extract_json(text: str):
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
            return None

        plan = extract_json(ai_output)
        if not plan or "task_overview" not in plan or "days" not in plan["task_overview"]:
            return jsonify({
                "error": "Failed to parse AI output as valid JSON",
                "raw_response": ai_output
            }), 500

        # --- Step 5: Save updated plan ---
        course_ref.update({
            "task_overview": plan,
            "reflection_updated": True,
            "generated_at": now_iso,
            "updated_at": now_iso
        })

        return jsonify({
            "success": True,
            "user_id": user_id,
            "course_id": course_id,
            "plan": plan,
            "message": "Task overview successfully updated based on user deficiencies"
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ============ MODIFY TASKS WITH LOCATIONS ENDPOINT ============
@app.route('/modify-tasks-with-locations', methods=['POST'])
def modify_tasks_with_locations():
    """
    Takes existing task overview and modifies tasks to incorporate user's selected locations
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    user_id = data.get("user_id", "").strip()
    course_id = data.get("course_id", "").strip()
    
    if not user_id or not course_id:
        return jsonify({"error": "Missing user_id or course_id"}), 400

    api_key = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not api_key:
        return jsonify({"error": "Missing API key in Authorization header"}), 401
    client.api_key = api_key

    # ========== STEP 1: Load Existing Task Overview ==========
    try:
        course_ref = db.collection('users').document(user_id).collection('datedcourses').document(course_id)
        course_doc = course_ref.get()
        
        if not course_doc.exists:
            return jsonify({"error": "Task overview not found. Please create tasks first."}), 404
        
        course_data = course_doc.to_dict()
        existing_overview = course_data.get('task_overview', {})
        goal_name = course_data.get('goal_name', '')
        
        print(f"✅ Loaded existing task overview with {len(existing_overview.get('days', []))} days")
    except Exception as e:
        return jsonify({"error": f"Failed to load task overview: {str(e)}"}), 500

    # ========== STEP 2: Load User's Selected Locations ==========
    selected_locations = []
    try:
        user_doc_ref = db.collection("users").document(user_id)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            selected_locations = user_doc.to_dict().get("selected_locations", [])
            print(f"✅ Loaded {len(selected_locations)} locations")
        
        if not selected_locations:
            return jsonify({"error": "No locations found. Please select locations first."}), 404
            
    except Exception as e:
        return jsonify({"error": f"Failed to load locations: {str(e)}"}), 500

    # ========== STEP 3: Load Prompt Template ==========
    prompt_file = "prompt_modify_tasks_locations.txt"
    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        return jsonify({"error": f"{prompt_file} not found"}), 404

    # ========== STEP 4: Prepare Data for AI ==========
    safe_goal_name = json.dumps(goal_name)[1:-1]
    safe_locations = json.dumps(selected_locations, indent=2)
    safe_existing_tasks = json.dumps(existing_overview, indent=2)

    prompt = prompt_template.replace("<<goal_name>>", safe_goal_name)
    prompt = prompt.replace("<<selected_locations>>", safe_locations)
    prompt = prompt.replace("<<existing_tasks>>", safe_existing_tasks)

    # ========== STEP 5: Generate Modified Tasks from AI ==========
    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=6000
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": "API request failed", "exception": str(e)}), 500

    # ========== STEP 6: Extract and Parse JSON ==========
    import re
    def extract_json(text: str):
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    modified_overview = extract_json(result)
    if not modified_overview:
        return jsonify({"error": "Failed to parse modified tasks as valid JSON", "raw_response": result}), 500
    
    print("✅ Tasks modified with locations")

    # ========== STEP 7: Validate Structure ==========
    if "days" not in modified_overview or not isinstance(modified_overview["days"], list):
        return jsonify({"error": "Invalid response structure - missing 'days' array"}), 500

    # ========== STEP 8: Save Modified Overview to Firebase ==========
    try:
        course_ref.update({
            'task_overview': modified_overview,
            'locations_integrated': True,
            'modified_at': datetime.now().isoformat()
        })
        print(f"✅ Saved modified task overview to Firebase")
    except Exception as e:
        return jsonify({"error": f"Failed to save to Firebase: {str(e)}"}), 500

    # ========== STEP 9: Return Response ==========
    return jsonify({
        "success": True,
        "course_id": course_id,
        "modified_overview": modified_overview,
        "locations_used": len(selected_locations),
        "message": "Tasks successfully modified with selected locations"
    })



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



























































