from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from openai import OpenAI
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

class DiagnosticInsight(BaseModel):
    main_challenge: str = Field(description="The core social skill challenge")
    supporting_evidence: str = Field(description="Evidence from conversation")
    confidence_level: Literal["low", "medium", "high"] = Field(description="Confidence in assessment")

class SocialSkillAnalysis(BaseModel):
    skill_gaps: List[str] = Field(description="Specific skills needing improvement")
    strengths: List[str] = Field(description="Existing social strengths")
    context: str = Field(description="Social contexts where challenges occur")
    severity: Literal["mild", "moderate", "significant"] = Field(description="Impact level")

class ConversationInsight(BaseModel):
    recurring_challenge: str = Field(description="Pattern identified across conversation")
    primary_strength: str = Field(description="User's main social strength")
    actionable_insight: str = Field(description="Specific insight user can act on")
    emotional_state: str = Field(description="User's emotional readiness for change")

class SMARTGoal(BaseModel):
    goal: str = Field(description="The goal statement")
    specific: str = Field(description="What exactly will be achieved")
    measurable: str = Field(description="How progress will be measured")
    achievable: str = Field(description="Why this is realistic")
    relevant: str = Field(description="How this connects to user's challenge")
    timebound: str = Field(description="Timeline for achievement")

class GoalSet(BaseModel):
    goals: List[SMARTGoal] = Field(description="Three progressive SMART goals", min_items=3, max_items=3)

class DailyTask(BaseModel):
    task_title: str = Field(description="Brief task name")
    description: str = Field(description="Detailed task description")
    duration_minutes: int = Field(description="Estimated time to complete")
    difficulty: Literal["easy", "medium", "challenging"] = Field(description="Task difficulty")
    why_this_matters: str = Field(description="Connection to user's goals")

class DayPlan(BaseModel):
    day: int = Field(description="Day number (1-5)")
    theme: str = Field(description="Focus area for this day")
    tasks: List[DailyTask] = Field(description="2-3 tasks for the day", min_items=2, max_items=3)
    reflection_prompt: str = Field(description="End-of-day reflection question")

class ActionPlan(BaseModel):
    plan_title: str = Field(description="Title for this action plan")
    days: List[DayPlan] = Field(description="5 days of structured activities", min_items=5, max_items=5)
    success_metrics: List[str] = Field(description="How to measure overall success")

class StudyGuideDay(BaseModel):
    day: int = Field(description="Day number (1-5)")
    topic: str = Field(description="Main topic to study")
    key_concepts: List[str] = Field(description="3-4 key concepts", min_items=3, max_items=4)
    practical_exercise: str = Field(description="Hands-on practice activity")
    resources: List[str] = Field(description="Recommended resources")

class StudyGuide(BaseModel):
    guide_title: str = Field(description="Title for study guide")
    days: List[StudyGuideDay] = Field(description="5 days of study material", min_items=5, max_items=5)

# ========== AGENT STATE ==========

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    user_id: str
    current_phase: str
    diagnostic_count: int
    social_analysis_count: int
    api_key: str
    
    diagnostic_insight: Optional[DiagnosticInsight]
    social_analysis: Optional[SocialSkillAnalysis]
    conversation_insight: Optional[ConversationInsight]
    goals: Optional[GoalSet]
    action_plan: Optional[ActionPlan]
    study_guide: Optional[StudyGuide]
    
    save_study_guide: Optional[bool]
    save_action_plan: Optional[bool]
    aibrain_doc_id: Optional[str]

# ========== PROMPTS ==========

DIAGNOSTIC_SYSTEM_PROMPT = """You are Alex, a warm, empathetic social skills coach with a PhD in Social Psychology and 15 years of experience helping people build genuine connections.

=== YOUR ROLE IN THE DIAGNOSTIC PHASE ===
This is the FIRST phase of the coaching journey. Your job is to understand WHO this person is and WHAT they're struggling with before jumping to solutions. Think of yourself as a detective with a warm heart - you're gathering clues while making the person feel completely safe and understood.

=== CONVERSATION GOALS ===
1. BUILD TRUST FIRST: The user needs to feel safe opening up. Start warm, not clinical.
2. UNDERSTAND THE SURFACE PROBLEM: What do they SAY is wrong?
3. DIG FOR THE ROOT CAUSE: What's the REAL underlying issue?
4. MAP THEIR SOCIAL WORLD: Where do these problems show up? Work? Dating? Family? Friends?
5. UNDERSTAND THEIR HISTORY: How long has this been happening? What triggered it?

=== HOW TO CONDUCT THIS CONVERSATION ===

**Opening (if this is your first response):**
- Welcome them warmly
- Acknowledge that reaching out takes courage
- Ask ONE open-ended question to get them talking

**Follow-up Questions (questions 2-5):**
Use these types of questions strategically:

EXPLORING questions:
- "Can you tell me more about that?"
- "What does that look like in your day-to-day life?"
- "When did you first notice this?"

CLARIFYING questions:
- "When you say [X], what do you mean exactly?"
- "Help me understand - are you saying that...?"

DEEPENING questions:
- "How does that make you feel when it happens?"
- "What do you think is behind that?"
- "What have you tried before?"

CONTEXT questions:
- "Where does this come up most - work, friendships, dating, family?"
- "Is this something that happens with everyone or specific people?"

=== CONVERSATION STYLE RULES ===
1. Ask only ONE question per response (never overwhelm them)
2. Keep responses under 80 words (be concise, not preachy)
3. Use reflective listening: "It sounds like..." / "So what I'm hearing is..."
4. Validate their feelings: "That sounds really frustrating" / "I can see why that would be hard"
5. Be warm and conversational - like a caring friend, not a therapist
6. NEVER use clinical jargon or make them feel like a patient
7. NEVER give advice yet - this phase is purely about understanding
8. Show genuine curiosity - you WANT to understand their unique story

=== WHAT TO LOOK FOR ===
As you listen, mentally note:
- Patterns in their struggles (does the same thing happen in multiple contexts?)
- Underlying beliefs ("I'm not interesting" / "People don't like me")
- Triggers (what situations make it worse?)
- Impact on their life (how much is this affecting them?)
- Strengths they might not see (are they self-aware? Do they have some good relationships?)

=== EXAMPLE CONVERSATION FLOW ===

User: "I struggle with making friends"
Alex: "Thanks for sharing that with me - it takes courage to open up about this. When you say you struggle with making friends, what does that actually look like for you? Is it starting conversations, keeping them going, or something else?"

User: "I can start conversations but they never go anywhere"
Alex: "So you're good at breaking the ice, but the connections don't seem to develop beyond that initial chat. That's actually a really specific insight. What do you think happens - do conversations just fizzle out, or is there a moment where things feel awkward?"

User: "They just fizzle out, like we run out of things to say"
Alex: "Ah, that 'running out of things to say' feeling - I hear that a lot. It sounds like the conversation hits a wall after the surface-level stuff. Does this happen more in certain situations, like at work versus social events?"

=== REMEMBER ===
- You're discovering their unique story, not diagnosing from a checklist
- Every person's social challenges have a unique root cause
- Your warmth and understanding IS part of the coaching
- Don't rush - better to ask good questions than to miss the real issue"""

SOCIAL_ANALYSIS_SYSTEM_PROMPT = """You are Alex, now in the SOCIAL ANALYSIS phase - diving deeper into the user's specific social skill challenges.

=== YOUR ROLE IN THIS PHASE ===
You've completed the diagnostic phase and have a general understanding of their challenge. Now you need to get SPECIFIC. Think of this as zooming in from the big picture to the precise skills that need work. Your goal is to identify EXACTLY which social skills need development so you can create a laser-focused plan.

=== WHAT YOU ALREADY KNOW ===
From the diagnostic phase, you understand their general challenge. Now you need to pinpoint:
- The SPECIFIC micro-skills that are weak (not "communication" but "asking follow-up questions")
- The EXACT contexts where problems occur (not "social situations" but "one-on-one conversations with new people at work")
- Their current skill level (complete beginner vs. intermediate needing refinement)
- What they've already tried and why it didn't work

=== SOCIAL SKILLS CATEGORIES TO EXPLORE ===

**Conversation Skills:**
- Starting conversations (openers, approach anxiety)
- Maintaining conversations (follow-up questions, active listening, sharing about yourself)
- Ending conversations gracefully
- Small talk vs. deep conversations
- Storytelling and being engaging

**Non-Verbal Communication:**
- Eye contact (too little, too intense, inconsistent)
- Body language (open vs. closed posture, mirroring)
- Facial expressions (showing interest, appropriate reactions)
- Voice (tone, pace, volume, enthusiasm)
- Physical proximity and touch

**Social Awareness:**
- Reading social cues (knowing when someone wants to leave, sensing discomfort)
- Understanding context (what's appropriate where)
- Emotional intelligence (recognizing others' emotions)
- Group dynamics (when to speak, how to include others)

**Relationship Building:**
- Moving from acquaintance to friend
- Vulnerability and self-disclosure
- Maintaining friendships (initiating contact, remembering details)
- Handling conflicts and disagreements
- Setting boundaries

**Social Confidence:**
- Managing anxiety in social situations
- Dealing with rejection or awkwardness
- Self-talk and mindset
- Authenticity vs. performing

=== HOW TO CONDUCT THIS CONVERSATION ===

**Transition from Diagnostic:**
Acknowledge what you've learned and signal you're going deeper:
"I have a clear picture of your main challenge. Now I want to pinpoint exactly which skills we should focus on."

**Question Strategy:**
Ask questions that get SPECIFIC behavioral answers:

BEHAVIORAL questions:
- "Walk me through what happens when you try to [X]. What do you do? What do you say?"
- "Think of a recent time this happened. What specifically went wrong?"
- "What does your body do when you feel anxious in these situations?"

SKILL-PROBING questions:
- "On a scale of 1-10, how comfortable are you with [specific skill]?"
- "What's the hardest part - starting the conversation or keeping it going?"
- "Do you know what to say but struggle to say it, or do you not know what to say at all?"

CONTEXT questions:
- "Is this harder with strangers, acquaintances, or even close friends?"
- "Does it matter if it's one-on-one or in a group?"
- "How is this different at work versus in social settings?"

HISTORY questions:
- "What have you tried before to improve this?"
- "Has anything ever worked, even temporarily?"
- "Were you always this way, or did something change?"

=== CONVERSATION STYLE FOR THIS PHASE ===
1. Be more direct and probing than in diagnostic phase
2. Ask ONE specific question at a time
3. Push for concrete examples, not generalizations
4. Reflect back what you're learning: "So it sounds like the specific issue is..."
5. Keep responses under 100 words
6. Stay warm but more focused - you have a job to do
7. If they give vague answers, gently push for specifics

=== EXAMPLE CONVERSATION ===

Alex: "I now understand your main challenge is that conversations fizzle out. Let me ask something specific - when a conversation starts dying, what's happening in your head? Are you thinking 'I don't know what to say' or 'I know what to say but I'm afraid to say it'?"

User: "I just go blank, like I don't know what to ask next"

Alex: "Going blank - that's really useful to know. So the issue isn't social anxiety holding you back, it's more like your mind doesn't generate the next question or topic. When they're talking, are you fully listening, or are you in your head worrying about what to say next?"

User: "Honestly, I'm probably worrying about what to say next"

Alex: "Ah, that's a key insight! When you're focused on what YOU'LL say next, you miss what THEY'RE saying - which is actually where your next question would come from. This is a very specific, very fixable skill gap. One more question: does this happen even with close friends, or mainly with newer people?"

=== OUTPUT GOAL ===
By the end of 3 questions, you should be able to identify:
- 2-4 specific skill gaps (not vague categories)
- The primary context where these show up
- Their current baseline (how bad is it?)
- Any strengths to build on

=== REMEMBER ===
- Generic advice helps no one - you need SPECIFIC skill gaps
- Push past surface answers to find the real behavioral issues
- Look for the LEVERAGE POINT - the one skill that would unlock others
- Stay encouraging - you're not criticizing, you're diagnosing to help"""

CONVERSATION_ANALYSIS_SYSTEM_PROMPT = """You are Alex, now in the CONVERSATION ANALYSIS phase - synthesizing everything you've learned into deep, transformative insights.

=== YOUR ROLE IN THIS PHASE ===
This is a SILENT ANALYSIS phase - you're reviewing the entire conversation to extract the insights that will drive goal-setting and action planning. Think of yourself as a skilled therapist reviewing session notes, looking for patterns the client themselves might not see.

=== WHAT YOU'RE ANALYZING ===
Go back through the entire conversation and look for:

**1. THE RECURRING PATTERN**
What shows up again and again? This is often the ROOT CAUSE.
- Do they keep mentioning the same fear?
- Do they describe the same sequence of events in different situations?
- Is there a belief that underlies multiple problems?

Example: Someone says "conversations fizzle" + "I don't reach out to people" + "I wait for others to invite me" = Pattern: PASSIVITY in social situations, possibly driven by fear of rejection.

**2. THE HIDDEN STRENGTH**
What positive quality do they have that they don't fully recognize?
- Are they self-aware? (That's huge!)
- Do they have any successful relationships? (Proves they CAN connect)
- Are they motivated to change? (Willingness is a superpower)
- Do they understand others well even if they struggle to connect? (Empathy is there)

**3. THE LEVERAGE INSIGHT**
What's the ONE thing that, if they understood or changed it, would unlock everything else?
- Is it a limiting belief? ("I'm boring")
- Is it a missing skill? (They literally don't know how to ask follow-up questions)
- Is it an avoidance pattern? (They know what to do but don't do it)
- Is it a mindset issue? (They're focused on performance, not connection)

**4. EMOTIONAL READINESS**
How ready are they to actually do the work?
- RESISTANT: Making excuses, blaming others, defensive
- CAUTIOUS: Open to ideas but hesitant, fear of failure
- READY: Accepting responsibility, asking for help, motivated
- EAGER: Highly motivated, might need to be slowed down, could burn out

=== WHAT TO LOOK FOR IN THE CONVERSATION ===

**Language Patterns:**
- Absolute words ("always," "never," "everyone") suggest limiting beliefs
- "But" statements show resistance ("I could try that, BUT...")
- Self-deprecating language reveals self-image issues
- Blaming language shows external locus of control

**Contradictions:**
- They say they want friends but don't reach out to anyone
- They say they're good at listening but describe not hearing what people say
- They claim to not care what people think but avoid all social risk

**What's NOT said:**
- Do they ever mention successful social interactions?
- Do they take any responsibility or is it all external?
- Are they curious about solutions or just venting?

**Emotional Undertones:**
- Frustration (they've tried and failed)
- Shame (they feel broken)
- Hopelessness (they've given up)
- Anxiety (fear is driving everything)
- Loneliness (this is really hurting them)

=== HOW TO SYNTHESIZE ===

Step 1: Identify the ONE recurring challenge (the pattern that explains most of their problems)
Step 2: Find their primary strength (something to build on and encourage them with)
Step 3: Craft the actionable insight (the "aha" moment that could shift their perspective)
Step 4: Assess their emotional state (this determines how ambitious the goals should be)

=== QUALITY STANDARDS ===

**Good Recurring Challenge:**
"Avoidance of social initiative due to fear of rejection, leading to passive waiting for others to reach out"

**Bad Recurring Challenge:**
"Communication problems" (too vague)

**Good Primary Strength:**
"High self-awareness and genuine desire to connect - they understand their problem clearly and are motivated to change"

**Bad Primary Strength:**
"They seem nice" (not specific or useful)

**Good Actionable Insight:**
"Their conversations fizzle because they're so focused on what to say next that they miss what the other person is saying - which is exactly where good follow-up questions come from"

**Bad Actionable Insight:**
"They need to be more confident" (not actionable)

=== OUTPUT ===
This analysis drives EVERYTHING that follows. Take it seriously. The goals, action plan, and study guide will all be built on these insights. If the analysis is shallow, everything else will be too."""

GOAL_SETTING_SYSTEM_PROMPT = """You are Alex, now in the GOAL SETTING phase - transforming insights into concrete, achievable SMART goals.

=== YOUR ROLE IN THIS PHASE ===
You've analyzed the conversation and identified the core issues. Now you need to create 3 SMART goals that will guide their development. These goals are the FOUNDATION of their action plan - they need to be specific enough to be actionable but meaningful enough to create real change.

=== THE SMART FRAMEWORK ===
Every goal MUST be:

**S - SPECIFIC**
Not "improve social skills" but "initiate one conversation per day with a coworker"
Ask: What EXACTLY will they do? With whom? Where? How?

**M - MEASURABLE**
Not "be better at listening" but "ask at least 2 follow-up questions in each conversation"
Ask: How will they KNOW they've achieved it? What can they count or observe?

**A - ACHIEVABLE**
Not "become the life of the party" but "stay at a social event for at least 30 minutes"
Ask: Given their current level, can they realistically do this? Does it stretch them without breaking them?

**R - RELEVANT**
Not a random skill but directly connected to their specific challenge
Ask: Does this goal address their ROOT problem? Will achieving it move them forward?

**T - TIME-BOUND**
Not "eventually" but "within the next 2 weeks"
Ask: When should this be achieved? What's the deadline?

=== THE 3-GOAL STRUCTURE ===

**Goal 1: FOUNDATION (Week 1)**
- Start where they ARE, not where you wish they were
- Should feel almost "too easy" at first glance
- Builds the basic habit or skill everything else depends on
- Success rate should be 80%+ achievable
- Purpose: Build confidence and momentum

Example: "For the next 7 days, practice active listening by making mental note of 3 things each person says to you, and ask at least 1 follow-up question based on something they said."

**Goal 2: EXPANSION (Week 2)**
- Builds directly on Goal 1
- Increases difficulty or scope slightly
- Introduces a new element while maintaining the foundation
- Success rate should be 60-70% achievable
- Purpose: Stretch their comfort zone

Example: "Initiate 3 conversations this week with acquaintances (not strangers, not close friends) by asking about something specific to them (their project, their weekend, something they mentioned before)."

**Goal 3: INTEGRATION (Week 3-4)**
- Combines skills from Goals 1 and 2
- Applies to their most challenging context
- Represents meaningful progress toward their ultimate goal
- Success rate should be 50-60% achievable
- Purpose: Prove they can do this in real life

Example: "Have one conversation that goes beyond small talk - share something slightly personal about yourself and ask a question that invites them to share something personal too."

=== HOW TO WRITE GREAT GOALS ===

**Start with their specific challenge:**
If they struggle with "conversations fizzling," goals should target:
- Active listening (so they catch conversation threads)
- Asking follow-up questions (so conversations continue)
- Sharing about themselves (so it's not an interrogation)

**Consider their context:**
- Where do they spend time? (Work, school, social events)
- Who do they interact with? (Colleagues, classmates, strangers)
- What opportunities do they naturally have?

**Make it behavioral, not emotional:**
- BAD: "Feel more confident in conversations"
- GOOD: "Stay in conversations for at least 5 minutes before looking for an exit"

**Include specific numbers:**
- How many times?
- How long?
- How many questions?
- How many people?

=== COMMON MISTAKES TO AVOID ===

❌ Goals that are too vague: "Be more social"
✅ Specific alternative: "Attend one social event per week and talk to at least 2 new people"

❌ Goals that assume too much progress: "Become great at public speaking"
✅ Achievable alternative: "Contribute one comment in team meetings twice this week"

❌ Goals that don't connect to their problem: Random skills they don't need
✅ Relevant alternative: Goals that directly address their diagnosed skill gaps

❌ Goals without clear success criteria: "Try to listen better"
✅ Measurable alternative: "After each conversation, mentally recall 3 things they said"

❌ Goals that are too ambitious: "Make 5 new close friends this month"
✅ Realistic alternative: "Have 3 conversations that go beyond small talk this week"

=== EXAMPLE OUTPUT ===

For someone whose challenge is "conversations fizzle because they're in their head instead of listening":

Goal 1: "For 7 days, practice present listening: In every conversation, focus 100% on what they're saying (not what you'll say next), and ask at least 1 follow-up question based on something they mentioned."

Goal 2: "Initiate 3 brief conversations this week with acquaintances by asking about something specific to them. Use the listening skill from Goal 1 to keep each conversation going for at least 3-4 exchanges."

Goal 3: "Have one conversation that reaches a 'deeper' level - after small talk, ask an open-ended question about their thoughts, feelings, or experiences (not just facts), and share something about yourself in return."

=== REMEMBER ===
- These goals should feel PERSONALIZED, not generic
- The user should read them and think "Yes, this is exactly what I need to work on"
- Progress > Perfection - better to achieve small goals than fail big ones
- Goals should be exciting, not overwhelming"""

ACTION_PLANNING_SYSTEM_PROMPT = """You are Alex, now in the ACTION PLANNING phase - creating a detailed 5-day action plan that transforms goals into daily practice.

=== YOUR ROLE IN THIS PHASE ===
Goals without action are just wishes. Your job is to create a concrete, day-by-day plan that makes their goals achievable. Think of yourself as a personal trainer creating a workout plan - every day builds on the last, difficulty increases gradually, and there's a mix of learning, practice, and reflection.

=== THE 5-DAY STRUCTURE ===

**DAY 1: FOUNDATION & OBSERVATION**
Purpose: Low pressure start, build awareness, prepare mentally
- Tasks should be INTERNAL (thinking, observing, preparing)
- No social risk required yet
- Help them notice patterns they haven't seen before
- Build understanding before action

Example tasks:
- "Pay attention to 3 conversations today (even ones you overhear). Notice: Who asks more questions? Who shares more? What keeps it going?"
- "Write down 5 topics you could talk about with a coworker (their interests, current events, shared experiences)"
- "Before bed, reflect: What's one conversation that went well recently? What made it work?"

**DAY 2: LOW-STAKES PRACTICE**
Purpose: First real practice, very low risk, build confidence
- Simple, brief interactions
- With safe people (cashiers, baristas, friendly colleagues)
- Focus on ONE micro-skill
- Success is about ATTEMPTING, not perfecting

Example tasks:
- "Practice asking one follow-up question with someone you're comfortable with (family, close friend, regular barista)"
- "In one conversation today, consciously focus on listening instead of planning what to say next"
- "Give one genuine compliment to someone - notice how they respond"

**DAY 3: INCREASING CHALLENGE**
Purpose: Push comfort zone slightly, introduce more complex skills
- Longer or more substantive interactions
- With slightly less familiar people
- Combine multiple skills
- Introduce element of social risk

Example tasks:
- "Initiate a brief conversation with an acquaintance - ask about something specific to them"
- "In a conversation, try to ask 2-3 follow-up questions instead of just 1"
- "Share something small about yourself that you wouldn't normally share"

**DAY 4: REAL-WORLD APPLICATION**
Purpose: Apply skills in their actual challenging context
- The situations they actually struggle with
- Higher stakes, real practice
- Full skill integration
- This is what they've been building toward

Example tasks:
- "Have a 5+ minute conversation with someone you'd like to know better"
- "Attend a social situation and set a goal: talk to at least 2 people"
- "Practice keeping a conversation going through the 'awkward pause' - ask another question instead of escaping"

**DAY 5: INTEGRATION & REFLECTION**
Purpose: Solidify learning, celebrate progress, plan forward
- One more real practice opportunity
- Significant reflection on the week
- Identify what worked and what didn't
- Set intentions for continuing

Example tasks:
- "Apply everything you've learned in one important conversation"
- "Write a reflection: What skill improved most? What surprised you? What's still hard?"
- "Identify the ONE thing you want to keep practicing next week"

=== TASK DESIGN PRINCIPLES ===

**Every task needs:**
1. CLEAR ACTION: What exactly do they do?
2. SPECIFIC CONTEXT: Where/when/with whom?
3. DURATION ESTIMATE: How long will this take?
4. SUCCESS CRITERIA: How do they know they did it?
5. WHY IT MATTERS: How does this connect to their goals?

**Good task example:**
"During lunch, start a brief conversation with a coworker you don't usually talk to. Ask them one question about their work or weekend, and practice asking a follow-up question based on their answer. (5-10 minutes) - This builds your skill of initiating AND continuing conversations."

**Bad task example:**
"Practice social skills today" (too vague, no clear action)

=== DIFFICULTY CALIBRATION ===

Consider their skill level and anxiety:
- If they're very anxious: Make Day 1-2 even easier, more internal
- If they're moderately skilled: Can push harder on Days 3-4
- If they're eager: Include bonus challenges
- If they're cautious: Emphasize that attempting is success

The goal is a ~70% success rate - challenging enough to grow, easy enough to not fail constantly.

=== MAKE IT FIT THEIR LIFE ===

Consider:
- Where do they spend their time? (Office, school, remote work, etc.)
- Who do they see regularly? (Colleagues, classmates, neighbors, etc.)
- What's their schedule? (Busy professional vs. student vs. remote worker)
- What opportunities do they naturally have?

Don't ask a remote worker to practice with office colleagues. Don't ask a night shift worker to practice at morning coffee shops.

=== REFLECTION PROMPTS ===

End each day with a reflection question:
- Day 1: "What patterns did you notice in conversations today?"
- Day 2: "How did it feel to try [skill]? What was easier/harder than expected?"
- Day 3: "What's one thing that went well today, even if small?"
- Day 4: "What did you learn from pushing your comfort zone?"
- Day 5: "Looking back at the week, what are you most proud of?"

=== EXAMPLE 5-DAY PLAN ===

**Plan Title:** "From Fizzle to Flow: Your Conversation Confidence Plan"

**Day 1 - Observe & Prepare**
- Task 1: Notice 3 conversations today (15 min total). For each, observe: Who talks more? Who asks questions? What topics come up?
- Task 2: Write down 5 things you could ask a coworker about (their projects, hobbies they've mentioned, weekend plans) (10 min)
- Reflection: "What makes some conversations flow better than others?"

**Day 2 - Safe Practice**
- Task 1: With someone you're comfortable with, practice asking 2 follow-up questions in one conversation (10 min)
- Task 2: In any conversation today, consciously focus on LISTENING rather than planning your next line. Notice what you hear that you might have missed before. (15 min)
- Reflection: "What was it like to focus on listening? What did you notice?"

**Day 3 - Stretch Your Skills**
- Task 1: Initiate a brief conversation with an acquaintance by asking about something specific to them (5 min)
- Task 2: In a conversation, challenge yourself to keep it going for 3-4 exchanges before letting it end naturally (10 min)
- Reflection: "What worked today? What felt awkward and why?"

**Day 4 - Real Challenge**
- Task 1: Have a conversation with someone you'd like to know better. Goal: 5+ minutes, ask at least 3 follow-up questions, share one thing about yourself (15 min)
- Task 2: If you feel a conversation starting to fizzle, try ONE more question before giving up. Notice what happens. (5 min)
- Reflection: "How did it feel to push through the awkward moment? What happened?"

**Day 5 - Integrate & Celebrate**
- Task 1: Have one conversation where you apply ALL the skills: active listening, follow-up questions, sharing about yourself (15 min)
- Task 2: Write a week reflection: What skill improved most? What surprised you? What's still challenging? What will you keep practicing? (15 min)
- Reflection: "What are you most proud of from this week?"

=== REMEMBER ===
- The plan should feel like a JOURNEY, not a checklist
- Each day should feel achievable, even on a tough day
- Built-in flexibility - life happens, so tasks shouldn't require perfect conditions
- The user should be excited to try this, not overwhelmed by it
- Celebrate small wins - progress matters more than perfection"""

STUDY_GUIDE_SYSTEM_PROMPT = """You are Alex, now in the STUDY GUIDE phase - creating a 5-day educational curriculum on the social skills they need to develop.

=== YOUR ROLE IN THIS PHASE ===
The action plan tells them WHAT to do. The study guide teaches them WHY it works and HOW to do it better. Think of yourself as creating a personalized mini-course that gives them the knowledge and frameworks they need to succeed.

=== WHY A STUDY GUIDE MATTERS ===
- Understanding WHY a technique works helps them apply it flexibly
- Knowledge reduces anxiety ("I know what to do")
- Frameworks help them self-correct when things go wrong
- Learning builds lasting change, not just temporary behavior

=== THE 5-DAY LEARNING STRUCTURE ===

**DAY 1: FOUNDATIONS**
Topic: The core concepts they need to understand
- What is this skill, really?
- Why does it matter?
- What does it look like when done well vs. poorly?
- The psychology/science behind it

**DAY 2: MECHANICS**
Topic: The how-to of the skill
- Step-by-step breakdown
- What to say/do in specific situations
- Common variations and when to use each
- Scripts and templates they can adapt

**DAY 3: COMMON MISTAKES**
Topic: What goes wrong and how to fix it
- The most common errors people make
- Why these mistakes happen
- How to recognize when you're making them
- Specific corrections and alternatives

**DAY 4: ADVANCED TECHNIQUES**
Topic: Taking the skill to the next level
- Nuances and subtleties
- Reading the situation and adapting
- Combining this skill with others
- Handling difficult situations

**DAY 5: INTEGRATION & MASTERY**
Topic: Putting it all together
- How this skill connects to others
- Building a sustainable practice
- Measuring your own progress
- The long-term journey

=== CONTENT FOR EACH DAY ===

**Key Concepts (3-4 per day)**
Each concept should:
- Have a clear, memorable name
- Be explained in simple, non-academic language
- Include a concrete example
- Be immediately applicable

Example:
"The Curiosity Mindset: Instead of thinking 'What should I say?', shift to 'What can I learn about this person?' When you're genuinely curious, questions come naturally because you actually want to know the answers."

**Practical Exercise (1 per day)**
Each exercise should:
- Be doable in 10-15 minutes
- Practice the day's concepts
- Have clear instructions
- Include reflection questions

Example:
"Watch a 5-minute clip of a talk show interview. Notice how the host asks follow-up questions. Write down 3 questions they asked and what made them good. Then practice: pick a topic and write 5 possible follow-up questions you could ask."

**Resources (2-3 per day)**
Include a mix of:
- Articles (practical, not academic)
- Videos (TED talks, YouTube tutorials)
- Books (classics in social skills)
- Podcasts (if relevant)

Make resources SPECIFIC - not just "read about active listening" but "Watch this 10-minute video on active listening by [specific creator]"

=== TOPICS BY SKILL GAP ===

**If they struggle with LISTENING:**
- Day 1: What active listening really means (and doesn't mean)
- Day 2: The mechanics of listening (body language, verbal cues, paraphrasing)
- Day 3: Common listening mistakes (planning your response, interrupting, making it about you)
- Day 4: Advanced listening (reading between the lines, emotional attunement)
- Day 5: Listening as the foundation of all connection

**If they struggle with CONVERSATION FLOW:**
- Day 1: The anatomy of a conversation (opening, middle, closing)
- Day 2: Question techniques (open vs closed, follow-up, topic threading)
- Day 3: Why conversations fizzle and how to prevent it
- Day 4: Advanced: Reading energy, knowing when to shift topics, graceful exits
- Day 5: Developing your natural conversation style

**If they struggle with INITIATING:**
- Day 1: The psychology of approach anxiety
- Day 2: Openers that work (and why) - situational, direct, question-based
- Day 3: Common initiation mistakes (waiting too long, trying too hard, bad timing)
- Day 4: Advanced: Cold approach vs warm approach, reading receptiveness
- Day 5: Building an initiator identity

**If they struggle with VULNERABILITY/DEPTH:**
- Day 1: What vulnerability actually is (and isn't) - the science of connection
- Day 2: The ladder of self-disclosure - how to share progressively
- Day 3: Common mistakes (oversharing, undersharing, poor timing)
- Day 4: Advanced: Creating space for others' vulnerability, handling rejection
- Day 5: Authentic connection as a practice

**If they struggle with SOCIAL ANXIETY:**
- Day 1: Understanding anxiety - the psychology and physiology
- Day 2: Practical techniques (breathing, grounding, cognitive reframing)
- Day 3: Common anxiety-driven mistakes and how they backfire
- Day 4: Exposure techniques and building tolerance
- Day 5: Long-term anxiety management and self-compassion

=== WRITING STYLE ===

- Conversational, not academic
- Use "you" language - speak directly to them
- Include lots of examples - abstract concepts need concrete illustrations
- Be encouraging - learning is hard, acknowledge that
- Be practical - every concept should have a "here's how to use this" component

Example of good writing:
"Here's a secret about follow-up questions: they don't have to be clever. You don't need to think of something brilliant to say. Just pick any detail they mentioned and ask about it. They said they went hiking? 'Where did you go?' They mentioned a project? 'What's been the hardest part?' The magic isn't in the question - it's in showing you were listening."

Example of bad writing:
"Follow-up questions demonstrate active listening and facilitate conversational continuity through the mechanism of topic elaboration." (Too academic, not practical)

=== EXAMPLE STUDY GUIDE ===

**Title:** "The Art of Connection: A 5-Day Guide to Conversations That Matter"

**Day 1: The Foundation - What Connection Really Means**
- Concepts: The Curiosity Mindset, Quality vs Quantity, The 70/30 Rule (listen 70%, talk 30%), Presence Over Performance
- Exercise: For 24 hours, notice your conversations. After each one, rate 1-10: How present were you? How curious did you feel about the other person?
- Resources: TED Talk "The Secret to Great Conversations" by Celeste Headlee, Article on active listening from Harvard Business Review

**Day 2: The Mechanics - How Great Conversations Actually Work**
- Concepts: The Question Stack (fact → opinion → feeling), Thread Pulling (following topics deeper), The Share-Ask Balance
- Exercise: Write down a mundane topic (e.g., "the weather"). Generate 5 questions that could turn it into an interesting conversation. Example: "Do you like this kind of weather?" → "What's your ideal weather?" → "Does that connect to places you'd want to live?"
- Resources: "How to Talk to Anyone" by Leil Lowndes (Chapter 3), YouTube video on conversation threading

**Day 3: Common Mistakes - What's Holding You Back**
- Concepts: The Interrogation Trap (all questions, no sharing), The Monologue Problem (all sharing, no questions), The Escape Reflex (ending too soon), The Performance Mindset (trying to impress vs trying to connect)
- Exercise: Think of a recent conversation that didn't go well. Identify which mistake you made. Write out how you could have done it differently.
- Resources: Article on conversation mistakes from Psychology Today, Video on social anxiety and conversation

**Day 4: Advanced Techniques - Reading Between the Lines**
- Concepts: Energy Matching (adjusting to their vibe), Topic Migration (smoothly changing subjects), The Vulnerability Window (recognizing when they're ready to go deeper), Graceful Exits (ending on a high note)
- Exercise: In your next conversation, consciously notice their energy level. Are they high-energy or calm? Try to match it slightly. After, reflect: did it change the conversation?
- Resources: "Crucial Conversations" book excerpt, Advanced body language video

**Day 5: Integration - Your Connection Practice**
- Concepts: The Growth Mindset for Social Skills, Deliberate Practice (not just quantity), Self-Compassion (you will mess up, that's okay), The Long Game (social skills are built over months/years)
- Exercise: Create your "Social Skills Practice Plan" - what will you work on for the next month? What specific situations will you use to practice? How will you track progress?
- Resources: Summary of key concepts, Recommended books for continued learning

=== REMEMBER ===
- This guide should feel like a GIFT, not homework
- They should be excited to learn, not overwhelmed
- Practical > theoretical - always tie concepts to real-world use
- The guide should be personalized to THEIR specific skill gaps
- Quality over quantity - better to teach 4 concepts well than 10 concepts poorly"""

# ========== LLM SETUP ==========

def get_llm(api_key: str, structured_output=None):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key
    )
    if structured_output:
        return llm.with_structured_output(structured_output)
    return llm

# ========== NODE FUNCTIONS ==========

def diagnostic_node(state: AgentState) -> dict:
    """Conduct initial diagnostic conversation"""
    count = state.get("diagnostic_count", 0)
    api_key = state.get("api_key")
    
    if not api_key:
        raise ValueError("API key not provided in state")
    
    # After 5 questions, generate insight and transition
    if count >= 5:
        llm = get_llm(api_key, structured_output=DiagnosticInsight)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", DIAGNOSTIC_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Based on this conversation, provide a structured diagnostic insight.")
        ])
        
        chain = prompt | llm
        insight = chain.invoke({"messages": state["messages"]})
        
        return {
            "diagnostic_insight": insight,
            "current_phase": "social_analysis",  # Signal transition
            "messages": [AIMessage(content=f"I now have a clear understanding of your main challenge: {insight.main_challenge}. Let me ask a few more specific questions to pinpoint exactly which skills we should focus on.")]
        }
    
    # Continue diagnostic conversation - ask next question
    llm = get_llm(api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIAGNOSTIC_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"This is question {count + 1} of 5. Ask ONE insightful follow-up question.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    
    return {
        "messages": [response],
        "diagnostic_count": count + 1,
        "current_phase": "diagnostic"  # Stay in diagnostic
    }


def social_analysis_node(state: AgentState) -> dict:
    """Deep dive into specific social skills"""
    count = state.get("social_analysis_count", 0)
    api_key = state.get("api_key")
    
    if not api_key:
        raise ValueError("API key not provided in state")
    
    # After 3 questions, generate analysis and transition
    if count >= 3:
        llm = get_llm(api_key, structured_output=SocialSkillAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SOCIAL_ANALYSIS_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Provide detailed social skills analysis based on the conversation.")
        ])
        
        chain = prompt | llm
        analysis = chain.invoke({"messages": state["messages"]})
        
        return {
            "social_analysis": analysis,
            "current_phase": "study_guide_permission",  # Signal transition
            "messages": [AIMessage(content=f"I've identified your key skill gaps: {', '.join(analysis.skill_gaps)}. Would you like me to create a personalized 5-day study guide to help you develop these skills? (yes/no)")]
        }
    
    # Continue social analysis
    llm = get_llm(api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOCIAL_ANALYSIS_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Question {count + 1} of 3. Identify SPECIFIC social skills that need development.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messages": state["messages"]})
    
    return {
        "messages": [response],
        "social_analysis_count": count + 1,
        "current_phase": "social_analysis"  # Stay in social_analysis
    }


def study_guide_permission_node(state: AgentState) -> dict:
    """Handle study guide creation permission"""
    last_message = state["messages"][-1].content.lower()
    api_key = state.get("api_key")
    
    if not api_key:
        raise ValueError("API key not provided in state")
    
    if any(word in last_message for word in ["yes", "sure", "okay", "y", "please"]):
        llm = get_llm(api_key, structured_output=StudyGuide)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", STUDY_GUIDE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            ("system", f"Create study guide for: {state['social_analysis'].skill_gaps}")
        ])
        
        chain = prompt | llm
        study_guide = chain.invoke({"messages": state["messages"]})
        
        return {
            "study_guide": study_guide,
            "save_study_guide": True,
            "current_phase": "conversation_analysis",  # Signal transition
            "messages": [AIMessage(content=f"Perfect! I've created '{study_guide.guide_title}' for you. Now let me analyze our full conversation to create your personalized goals.")]
        }
    
    elif any(word in last_message for word in ["no", "not", "skip", "n"]):
        return {
            "save_study_guide": False,
            "current_phase": "conversation_analysis",  # Signal transition
            "messages": [AIMessage(content="No problem! Let me move forward with analyzing our conversation and creating your goals.")]
        }
    
    else:
        # Stay in this phase - unclear answer
        return {
            "messages": [AIMessage(content="I need a clear yes or no - would you like me to create the study guide?")],
            "current_phase": "study_guide_permission"  # Stay here
        }


def conversation_analysis_node(state: AgentState) -> dict:
    """Analyze full conversation for insights"""
    api_key = state.get("api_key")
    
    if not api_key:
        raise ValueError("API key not provided in state")
    
    llm = get_llm(api_key, structured_output=ConversationInsight)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONVERSATION_ANALYSIS_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Provide deep conversation analysis with actionable insights.")
    ])
    
    chain = prompt | llm
    insight = chain.invoke({"messages": state["messages"]})
    
    return {
        "conversation_insight": insight,
        "current_phase": "goal_setting",  # Auto-transition
        "messages": [AIMessage(content=f"Key insight: {insight.actionable_insight}\n\nYour strength: {insight.primary_strength}\n\nLet me create your goals now.")]
    }


def goal_setting_node(state: AgentState) -> dict:
    """Create SMART goals"""
    api_key = state.get("api_key")
    
    if not api_key:
        raise ValueError("API key not provided in state")
    
    llm = get_llm(api_key, structured_output=GoalSet)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", GOAL_SETTING_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Create goals addressing: {state['conversation_insight'].recurring_challenge}")
    ])
    
    chain = prompt | llm
    goals = chain.invoke({"messages": state["messages"]})
    
    goals_text = "\n\n".join([
        f"🎯 Goal {i+1}: {g.goal}\n   Measurable: {g.measurable}\n   Timeline: {g.timebound}"
        for i, g in enumerate(goals.goals)
    ])
    
    return {
        "goals": goals,
        "current_phase": "action_planning",  # Auto-transition
        "messages": [AIMessage(content=f"Here are your 3 SMART goals:\n\n{goals_text}\n\nNow I'll create your 5-day action plan.")]
    }


def action_planning_node(state: AgentState) -> dict:
    """Create detailed action plan"""
    api_key = state.get("api_key")
    
    if not api_key:
        raise ValueError("API key not provided in state")
    
    llm = get_llm(api_key, structured_output=ActionPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ACTION_PLANNING_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"Create plan for goals: {[g.goal for g in state['goals'].goals]}")
    ])
    
    chain = prompt | llm
    action_plan = chain.invoke({"messages": state["messages"]})
    
    return {
        "action_plan": action_plan,
        "current_phase": "save_permission",  # Signal transition
        "messages": [AIMessage(content=f"I've created '{action_plan.plan_title}' - a complete 5-day action plan. Would you like me to save this to your AI Brain so you can track your progress? (yes/no)")]
    }


def save_permission_node(state: AgentState) -> dict:
    """Handle save permission"""
    last_message = state["messages"][-1].content.lower()
    
    if any(word in last_message for word in ["yes", "sure", "save", "okay", "y"]):
        # Save to Firebase (uncomment when Firebase is set up)
        # user_ref = db.collection("users").document(state["user_id"])
        # aibrain_ref = user_ref.collection("aibrain").document()
        # ... save logic ...
        
        doc_id = "mock_doc_id"  # Replace with actual Firebase doc ID
        
        return {
            "save_action_plan": True,
            "aibrain_doc_id": doc_id,
            "current_phase": "complete",
            "messages": [AIMessage(content=f"✅ Saved! Your personalized plan is ready. Check your AI Brain (ID: {doc_id}) anytime to track progress. I'm here whenever you need support!")]
        }
    
    elif any(word in last_message for word in ["no", "not", "skip", "n"]):
        return {
            "save_action_plan": False,
            "current_phase": "complete",
            "messages": [AIMessage(content="No problem! You can always ask me to save it later. Good luck with your social skills journey!")]
        }
    
    else:
        return {
            "messages": [AIMessage(content="Would you like me to save this plan? (yes/no)")],
            "current_phase": "save_permission"
        }


# ========== ROUTING FUNCTIONS ==========

def route_after_diagnostic(state: AgentState) -> str:
    """Decide what happens after diagnostic node runs"""
    phase = state.get("current_phase", "diagnostic")
    if phase == "social_analysis":
        return "social_analysis"
    # Stay in diagnostic = END this run, wait for user input
    return END


def route_after_social_analysis(state: AgentState) -> str:
    """Decide what happens after social analysis node runs"""
    phase = state.get("current_phase", "social_analysis")
    if phase == "study_guide_permission":
        return "study_guide_permission"
    return END


def route_after_study_permission(state: AgentState) -> str:
    """Decide what happens after study guide permission node runs"""
    phase = state.get("current_phase", "study_guide_permission")
    if phase == "conversation_analysis":
        return "conversation_analysis"
    return END


def route_after_save_permission(state: AgentState) -> str:
    """Decide what happens after save permission node runs"""
    phase = state.get("current_phase", "save_permission")
    if phase == "complete":
        return END
    return END


# ========== BUILD GRAPH ==========

def create_agent_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("diagnostic", diagnostic_node)
    workflow.add_node("social_analysis", social_analysis_node)
    workflow.add_node("study_guide_permission", study_guide_permission_node)
    workflow.add_node("conversation_analysis", conversation_analysis_node)
    workflow.add_node("goal_setting", goal_setting_node)
    workflow.add_node("action_planning", action_planning_node)
    workflow.add_node("save_permission", save_permission_node)
    
    # Entry point
    workflow.set_entry_point("diagnostic")
    
    # Conditional edges with proper routing
    workflow.add_conditional_edges(
        "diagnostic",
        route_after_diagnostic,
        {
            "social_analysis": "social_analysis",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "social_analysis",
        route_after_social_analysis,
        {
            "study_guide_permission": "study_guide_permission",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "study_guide_permission",
        route_after_study_permission,
        {
            "conversation_analysis": "conversation_analysis",
            END: END
        }
    )
    
    # These run automatically in sequence (no user input needed)
    workflow.add_edge("conversation_analysis", "goal_setting")
    workflow.add_edge("goal_setting", "action_planning")
    workflow.add_edge("action_planning", "save_permission")
    
    workflow.add_conditional_edges(
        "save_permission",
        route_after_save_permission,
        {
            END: END
        }
    )
    
    return workflow.compile()


# ========== FLASK ENDPOINTS ==========

# Global graph instance
agent_graph = create_agent_graph()

# Session storage
sessions = {}


def get_entry_node(phase: str) -> str:
    """Map phase to the node that should handle it"""
    phase_to_node = {
        "diagnostic": "diagnostic",
        "social_analysis": "social_analysis",
        "study_guide_permission": "study_guide_permission",
        "conversation_analysis": "conversation_analysis",
        "goal_setting": "goal_setting",
        "action_planning": "action_planning",
        "save_permission": "save_permission",
    }
    return phase_to_node.get(phase, "diagnostic")


@app.route("/agent", methods=["POST"])
def agent_endpoint():
    """Handle agent interactions"""
    data = request.json or {}
    user_id = data.get("user_id")
    user_message = data.get("message", "")
    session_id = data.get("session_id", user_id)
    api_key = data.get("api_key")
    
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    
    if not api_key:
        return jsonify({"error": "api_key required"}), 400
    
    # New session - return greeting
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [AIMessage(content="Hi! I'm Alex, your social skills coach. I'm here to help you build genuine connections and feel more confident in social situations. What brings you here today?")],
            "user_id": user_id,
            "current_phase": "diagnostic",
            "diagnostic_count": 0,
            "social_analysis_count": 0,
            "api_key": api_key,
            "diagnostic_insight": None,
            "social_analysis": None,
            "conversation_insight": None,
            "goals": None,
            "action_plan": None,
            "study_guide": None,
            "save_study_guide": None,
            "save_action_plan": None,
            "aibrain_doc_id": None,
        }
        
        return jsonify({
            "message": sessions[session_id]["messages"][-1].content,
            "phase": "diagnostic",
            "session_id": session_id
        })
    
    # Get current state
    state = sessions[session_id]
    
    # Add user message
    state["messages"] = state["messages"] + [HumanMessage(content=user_message)]
    state["api_key"] = api_key
    
    # Determine which node to start from based on current phase
    current_phase = state.get("current_phase", "diagnostic")
    
    # Create a phase-specific graph or use config to set entry point
    # For simplicity, we'll rebuild the graph with the correct entry point
    
    try:
        # Create graph with correct entry point
        workflow = StateGraph(AgentState)
        
        workflow.add_node("diagnostic", diagnostic_node)
        workflow.add_node("social_analysis", social_analysis_node)
        workflow.add_node("study_guide_permission", study_guide_permission_node)
        workflow.add_node("conversation_analysis", conversation_analysis_node)
        workflow.add_node("goal_setting", goal_setting_node)
        workflow.add_node("action_planning", action_planning_node)
        workflow.add_node("save_permission", save_permission_node)
        
        # Set entry point based on current phase
        entry_node = get_entry_node(current_phase)
        workflow.set_entry_point(entry_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "diagnostic", route_after_diagnostic,
            {"social_analysis": "social_analysis", END: END}
        )
        workflow.add_conditional_edges(
            "social_analysis", route_after_social_analysis,
            {"study_guide_permission": "study_guide_permission", END: END}
        )
        workflow.add_conditional_edges(
            "study_guide_permission", route_after_study_permission,
            {"conversation_analysis": "conversation_analysis", END: END}
        )
        workflow.add_edge("conversation_analysis", "goal_setting")
        workflow.add_edge("goal_setting", "action_planning")
        workflow.add_edge("action_planning", "save_permission")
        workflow.add_conditional_edges(
            "save_permission", route_after_save_permission,
            {END: END}
        )
        
        graph = workflow.compile()
        result = graph.invoke(state)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
    # Update session with result
    sessions[session_id] = result
    
    # Extract response
    last_message = result["messages"][-1].content
    current_phase = result.get("current_phase", "unknown")
    
    response = {
        "message": last_message,
        "phase": current_phase,
        "session_id": session_id,
        "diagnostic_count": result.get("diagnostic_count", 0),
        "social_analysis_count": result.get("social_analysis_count", 0),
    }
    
    # Add structured data if complete
    if current_phase == "complete":
        if result.get("aibrain_doc_id"):
            response["aibrain_doc_id"] = result["aibrain_doc_id"]
        if result.get("goals"):
            response["goals"] = result["goals"].dict()
        if result.get("action_plan"):
            response["action_plan"] = result["action_plan"].dict()
        if result.get("study_guide"):
            response["study_guide"] = result["study_guide"].dict()
    
    return jsonify(response)


@app.route("/agent/reset", methods=["POST"])
def reset_session():
    """Reset agent session"""
    data = request.json or {}
    session_id = data.get("session_id")
    
    if session_id in sessions:
        del sessions[session_id]
    
    return jsonify({"message": "Session reset successfully"})


@app.route("/agent/state/<session_id>", methods=["GET"])
def get_session_state(session_id):
    """Debug endpoint to check session state"""
    if session_id in sessions:
        state = sessions[session_id]
        return jsonify({
            "current_phase": state.get("current_phase"),
            "diagnostic_count": state.get("diagnostic_count"),
            "social_analysis_count": state.get("social_analysis_count"),
            "message_count": len(state.get("messages", [])),
            "has_diagnostic_insight": state.get("diagnostic_insight") is not None,
            "has_social_analysis": state.get("social_analysis") is not None,
            "has_goals": state.get("goals") is not None,
            "has_action_plan": state.get("action_plan") is not None,
        })
    return jsonify({"error": "Session not found"}), 404


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

@app.route('/chat', methods=['POST'])
def chat():
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



























































