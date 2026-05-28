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
import threading
import uuid
import threading
import traceback
from datetime import datetime, timedelta
from flask import Response, stream_with_context
import httpx
from firebase_admin import firestore
import traceback
import io
import itertools
import requests as http_requests

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
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        res.headers['Access-Control-Max-Age'] = '3600'
        res.status_code = 200
        return res
        

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


# ── ElevenLabs Key Rotation Setup ───────────────────────────
ELEVENLABS_API_KEYS = [v for k, v in os.environ.items() if k.startswith("ELEVENLABS_API_KEY") and v]
if not ELEVENLABS_API_KEYS:
    print("⚠️ Warning: No ElevenLabs API keys configured")

_key_cycle = itertools.cycle(ELEVENLABS_API_KEYS) if ELEVENLABS_API_KEYS else None
_key_lock = threading.Lock()
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — calm, warm

def get_next_elevenlabs_key():
    with _key_lock:
        return next(_key_cycle)

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



# ============================================================
# THERAPY SESSION ENDPOINTS
# Two endpoints:
#   1. /therapy-session  — stateful CBT mini-session (4 phases)
#   2. /session-to-plan  — converts completed session into task
# ============================================================

import uuid
from datetime import datetime

# ── ENDPOINT 0: /transcribe ─────────────────────────────────
@app.route('/transcribe', methods=['POST', 'OPTIONS'])
def transcribe():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio_bytes = request.files['audio'].read()
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.webm", io.BytesIO(audio_bytes), "audio/webm"),
        )
        return jsonify({"success": True, "transcript": result.text.strip()})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ── ENDPOINT 0b: /speak ─────────────────────────────────────
# ── ENDPOINT 0b: /speak (ElevenLabs with key rotation) ──────
@app.route('/speak', methods=['POST', 'OPTIONS'])
def speak():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.headers['Access-Control-Max-Age'] = '3600'
        res.status_code = 200
        return res
        
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        if not ELEVENLABS_API_KEYS:
            return jsonify({"error": "No ElevenLabs API keys configured"}), 500

        # Clean markdown/formatting so TTS doesn't sound robotic
        def clean_text_for_tts(t: str) -> str:
            t = re.sub(r'\*\*?(.*?)\*\*?', r'\1', t)
            t = re.sub(r'^\s*[-•*]\s+', '', t, flags=re.MULTILINE)
            t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)
            t = re.sub(r'^#+\s+', '', t, flags=re.MULTILINE)
            t = re.sub(r'\[.*?\]|\(.*?\)', '', t)
            t = re.sub(r'\n{2,}', '. ', t)
            t = re.sub(r'\n', ' ', t)
            t = re.sub(r'\s{2,}', ' ', t).strip()
            return t

        cleaned_text = clean_text_for_tts(text)
        last_error = None

        for _ in range(len(ELEVENLABS_API_KEYS)):
            api_key = get_next_elevenlabs_key()
            try:
                el_response = http_requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                    headers={
                        "xi-api-key": api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "text": cleaned_text,
                        "model_id": "eleven_turbo_v2",
                        "voice_settings": {
                            "stability": 0.4,
                            "similarity_boost": 0.75,
                            "style": 0.3,
                            "use_speaker_boost": True
                        }
                    },
                    timeout=15
                )

                if el_response.status_code == 200:
                    return Response(
                        el_response.content,
                        mimetype="audio/mpeg",
                        headers={
                            "Content-Type": "audio/mpeg",
                            "Access-Control-Allow-Origin": "*",
                        }
                    )
                elif el_response.status_code == 429:
                    print(f"[ElevenLabs] Quota hit on key ...{api_key[-4:]}, rotating...")
                    last_error = f"429 quota hit on key ...{api_key[-4:]}"
                    continue
                else:
                    last_error = f"ElevenLabs error {el_response.status_code}: {el_response.text}"
                    print(f"[ElevenLabs] {last_error}")
                    break

            except http_requests.exceptions.Timeout:
                print(f"[ElevenLabs] Timeout on key ...{api_key[-4:]}, rotating...")
                last_error = "Request timed out"
                continue

        return jsonify({"error": last_error or "All ElevenLabs keys exhausted"}), 503

    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
        


# ── ENDPOINT 1: /therapy-session ────────────────────────────
@app.route('/therapy-session', methods=['POST', 'OPTIONS'])
def therapy_session():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        user_id        = data.get("user_id")
        user_message   = data.get("message", "").strip()
        session_id     = data.get("session_id")       # None on first turn
        start_new      = data.get("start_new", False) # True to force new session

        if not user_id or not user_message:
            return jsonify({"error": "user_id and message required"}), 400

        # API key from Authorization header
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return jsonify({"error": "Server API key not configured"}), 500

        client.api_key = api_key

        # ── Load or create session ───────────────────────────
        if not session_id or start_new:
            session_id = f"therapy_{user_id}_{int(datetime.now().timestamp())}"
            session_data = {
                "session_id":       session_id,
                "user_id":          user_id,
                "messages":         [],
                "phase":            1,
                "extracted":        {
                    "situation":        "",
                    "anxious_thought":  "",
                    "emotion":          "",
                    "reframe":          "",
                    "proposed_task":    {
                        "name":         "",
                        "type":         "",
                        "why":          "",
                        "anxiety_pre":  5,
                        "action_steps": []
                    }
                },
                "session_complete": False,
                "created_at":       datetime.utcnow().isoformat()
            }
            # Save new session
            db.collection("users").document(user_id)\
              .collection("therapy_sessions").document(session_id)\
              .set(session_data)
        else:
            # Load existing session
            doc = db.collection("users").document(user_id)\
                    .collection("therapy_sessions").document(session_id).get()
            if not doc.exists:
                return jsonify({"error": "Session not found. Pass start_new: true to begin."}), 404
            session_data = doc.to_dict()

        # ── Guard: already complete ──────────────────────────
        if session_data.get("session_complete"):
            return jsonify({
                "session_id":       session_id,
                "reply":            "This session is complete. Call /session-to-plan to convert it into your task.",
                "phase":            5,
                "session_complete": True,
                "extracted":        session_data.get("extracted", {})
            })

        # ── Build message history ────────────────────────────
        messages = session_data.get("messages", [])

        # On turn 1: inject system prompt
        if len(messages) == 0:
            system_prompt = load_prompt("prompt_therapy_session.txt")
            if not system_prompt:
                return jsonify({"error": "prompt_therapy_session.txt not found"}), 500

            # Inject any prior context we have about this user
            try:
                user_doc = db.collection("users").document(user_id).get()
                user_profile = user_doc.to_dict() if user_doc.exists else {}
                prior_context = f"""
USER PROFILE (use this to personalise — do not mention it explicitly):
- Past activities: {user_profile.get('completed_activities_count', 0)} completed
- Typical anxiety level: {user_profile.get('baseline_anxiety', 'unknown')}
- Goal type: {user_profile.get('goal', 'general')}
"""
            except Exception:
                prior_context = ""

            messages = [{"role": "system", "content": system_prompt + prior_context}]

        # Append user turn
        messages.append({"role": "user", "content": user_message})

        # ── Build phase context injected as system reminder ──
        current_phase = session_data.get("phase", 1)
        extracted_so_far = session_data.get("extracted", {})
        phase_reminder = {
            "role": "system",
            "content": f"""
CURRENT PHASE: {current_phase}
EXTRACTED SO FAR: {json.dumps(extracted_so_far, indent=2)}
INSTRUCTION: Respond with valid JSON only. Follow the phase rules in your system prompt.
Session must not exceed 4 phases. Phase 4 ends the session.
"""
        }

        messages_for_model = [messages[0], phase_reminder] + messages[1:]

        # ── Call Groq ────────────────────────────────────────
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_model,
            temperature=0.65,
            max_tokens=700
        )
        raw_reply = response.choices[0].message.content.strip()

        # ── Parse JSON from LLM response ─────────────────────
        parsed = parse_json_response(raw_reply)
        if not parsed:
            # Fallback: return raw text, don't break session
            messages.append({"role": "assistant", "content": raw_reply})
            db.collection("users").document(user_id)\
              .collection("therapy_sessions").document(session_id)\
              .update({"messages": messages})
            return jsonify({
                "session_id": session_id,
                "reply":      raw_reply,
                "phase":      current_phase,
                "session_complete": False
            })

        ai_reply       = parsed.get("message", raw_reply)
        next_phase     = parsed.get("phase", current_phase)
        session_complete = parsed.get("session_complete", False)
        new_extracted  = parsed.get("extracted", {})

        # ── Merge extracted data (never overwrite with empty) ─
        merged_extracted = session_data.get("extracted", {})
        for key, val in new_extracted.items():
            if isinstance(val, dict):
                if key not in merged_extracted:
                    merged_extracted[key] = {}
                for subkey, subval in val.items():
                    if subval and subval != "" and subval != 0:
                        merged_extracted[key][subkey] = subval
            else:
                if val and val != "" and val != 0:
                    merged_extracted[key] = val

        # ── Append assistant turn to history ─────────────────
        messages.append({"role": "assistant", "content": ai_reply})

        # ── Save updated session to Firestore ─────────────────
        update_payload = {
            "messages":         messages,
            "phase":            next_phase,
            "extracted":        merged_extracted,
            "session_complete": session_complete,
            "updated_at":       datetime.utcnow().isoformat()
        }
        if session_complete:
            update_payload["completed_at"] = datetime.utcnow().isoformat()

        db.collection("users").document(user_id)\
          .collection("therapy_sessions").document(session_id)\
          .update(update_payload)

        return jsonify({
            "session_id":       session_id,
            "reply":            ai_reply,
            "phase":            next_phase,
            "session_complete": session_complete,
            "extracted":        merged_extracted,
            "turn_count":       len([m for m in messages if m["role"] == "user"])
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500




# ── ENDPOINT 2: /session-to-plan ────────────────────────────
@app.route('/session-to-plan', methods=['POST', 'OPTIONS'])
def session_to_plan():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data       = request.get_json()
        user_id    = data.get("user_id")
        session_id = data.get("session_id")

        if not user_id or not session_id:
            return jsonify({"error": "user_id and session_id required"}), 400

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return jsonify({"error": "Server API key not configured"}), 500

        client.api_key = api_key

        # ── Load the completed session ───────────────────────
        doc = db.collection("users").document(user_id)\
                .collection("therapy_sessions").document(session_id).get()
        if not doc.exists:
            return jsonify({"error": "Session not found"}), 404

        session_data = doc.to_dict()

        if not session_data.get("session_complete"):
            return jsonify({
                "error": "Session is not complete yet. Finish the therapy session first.",
                "current_phase": session_data.get("phase", 1)
            }), 400

        extracted = session_data.get("extracted", {})
        if not extracted.get("proposed_task", {}).get("name"):
            return jsonify({"error": "No task was extracted from this session"}), 400

        # ── Load prompt ──────────────────────────────────────
        prompt_template = load_prompt("prompt_session_to_plan.txt")
        if not prompt_template:
            return jsonify({"error": "prompt_session_to_plan.txt not found"}), 500

        # Build the prompt with session data injected
        session_summary = f"""
SITUATION: {extracted.get('situation', '')}
ANXIOUS THOUGHT: {extracted.get('anxious_thought', '')}
EMOTION: {extracted.get('emotion', '')}
CBT REFRAME: {extracted.get('reframe', '')}
PROPOSED TASK: {json.dumps(extracted.get('proposed_task', {}), indent=2)}
"""
        full_prompt = prompt_template.replace("<<session_summary>>", session_summary)

        # ── Call Groq ────────────────────────────────────────
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,   # Low temp — we want precise structured output
            max_tokens=600
        )
        raw = response.choices[0].message.content.strip()

        # ── Parse the plan JSON ──────────────────────────────
        plan = parse_json_response(raw)
        if not plan:
            return jsonify({"error": "Failed to parse plan from LLM", "raw": raw}), 500

        # ── Stamp with metadata ──────────────────────────────
        plan["session_id"]    = session_id
        plan["created_at"]    = int(datetime.now().timestamp() * 1000)
        plan["completed"]     = False
        plan["source"]        = "therapy_session"

        # Ensure scheduledDate is a timestamp ms integer
        if "scheduledDate" not in plan or not plan["scheduledDate"]:
            # Default: 24h from now
            plan["scheduledDate"] = int((datetime.now().timestamp() + 86400) * 1000)

        # ── Save plan to activities (same collection as app) ──
        activity_ref = db.collection("users").document(user_id)\
                         .collection("activities").document()
        activity_ref.set(plan)
        plan["id"] = activity_ref.id

        # ── Also update the session doc with plan reference ───
        db.collection("users").document(user_id)\
          .collection("therapy_sessions").document(session_id)\
          .update({
              "plan_id":         activity_ref.id,
              "plan_created_at": datetime.utcnow().isoformat()
          })

        # ── Return both session summary and the plan ──────────
        return jsonify({
            "success":      True,
            "plan":         plan,
            "activity_id":  activity_ref.id,
            "session_summary": {
                "situation":       extracted.get("situation", ""),
                "anxious_thought": extracted.get("anxious_thought", ""),
                "reframe":         extracted.get("reframe", ""),
                "task_name":       extracted.get("proposed_task", {}).get("name", "")
            },
            "message": "Plan saved to your activities."
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ── ENDPOINT 3: /therapy-session/history ─────────────────────
# Fetch all past sessions for a user (for the history view)
@app.route('/therapy-session/history', methods=['POST', 'OPTIONS'])
def therapy_session_history():
    if request.method == 'OPTIONS':
        return '', 204

    data    = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    try:
        sessions_ref = db.collection("users").document(user_id)\
                         .collection("therapy_sessions")\
                         .order_by("created_at", direction=firestore.Query.DESCENDING)\
                         .limit(20)
        docs = sessions_ref.stream()

        sessions = []
        for doc in docs:
            d = doc.to_dict()
            sessions.append({
                "session_id":       doc.id,
                "created_at":       d.get("created_at"),
                "session_complete": d.get("session_complete", False),
                "phase":            d.get("phase", 1),
                "plan_id":          d.get("plan_id"),
                "situation":        d.get("extracted", {}).get("situation", ""),
                "task_name":        d.get("extracted", {}).get("proposed_task", {}).get("name", ""),
                "turn_count":       len([m for m in d.get("messages", []) if m.get("role") == "user"])
            })

        return jsonify({"success": True, "sessions": sessions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
# ================== PHASE CONFIG ==================
# Phase 0 (frontend-only): Condition selector — user taps cards + optional custom input
# Phase 1: Companion personality calibration (chat)
# Phase 2: Personal context & memory seeding (chat)
# Phase 3: Rhythm & availability (chat)
# Phase 4: Confirmation → companion profile generated
# Phase 5: Done — companion profile active

PHASE_REQUIREMENTS = {
    1: ["companion_name", "companion_persona", "support_style", "topics_to_avoid"],
    2: ["emotional_state", "important_people", "ongoing_situations", "goals"],
    3: ["check_in_frequency", "best_times", "current_stressors", "notification_preference"],
    4: ["confirmation"],
}

# ================== CONDITION CATALOG ==================
# Sent to frontend on /init-session so the card selector is always in sync with backend.
# Each entry: id, label, emoji, description (shown on card), category

CONDITION_CATALOG = [
    # Mental health
    {"id": "anxiety",          "label": "Anxiety",               "emoji": "😰", "description": "Worry, panic, overthinking", "category": "mental_health"},
    {"id": "depression",       "label": "Depression",            "emoji": "🌧",  "description": "Low mood, emptiness, no motivation", "category": "mental_health"},
    {"id": "stress",           "label": "Stress & Burnout",      "emoji": "🔥", "description": "Overwhelm, exhaustion, pressure", "category": "mental_health"},
    {"id": "loneliness",       "label": "Loneliness",            "emoji": "🫧", "description": "Feeling disconnected or isolated", "category": "mental_health"},
    {"id": "anger",            "label": "Anger & Frustration",   "emoji": "⚡", "description": "Irritability, rage, resentment", "category": "mental_health"},
    {"id": "grief",            "label": "Grief & Loss",          "emoji": "🕊",  "description": "Loss of a person, relationship, or chapter", "category": "mental_health"},
    {"id": "trauma",           "label": "Trauma & PTSD",         "emoji": "🧩", "description": "Flashbacks, hypervigilance, past wounds", "category": "mental_health"},
    {"id": "ocd",              "label": "OCD",                   "emoji": "🔁", "description": "Intrusive thoughts, compulsions, rituals", "category": "mental_health"},
    # Relationships
    {"id": "relationship",     "label": "Relationship Issues",   "emoji": "💔", "description": "Conflict, breakups, communication", "category": "relationships"},
    {"id": "social_anxiety",   "label": "Social Anxiety",        "emoji": "👥", "description": "Fear of judgment, avoidance, shyness", "category": "relationships"},
    {"id": "family",           "label": "Family Stress",         "emoji": "🏠", "description": "Parent, sibling, or household tension", "category": "relationships"},
    # Life & identity
    {"id": "self_esteem",      "label": "Self-Esteem",           "emoji": "🪞", "description": "Self-doubt, shame, not feeling enough", "category": "identity"},
    {"id": "purpose",          "label": "Purpose & Direction",   "emoji": "🧭", "description": "Lost, stuck, unsure what you want", "category": "identity"},
    {"id": "identity",         "label": "Identity & Belonging",  "emoji": "🌈", "description": "Who am I? Where do I fit?", "category": "identity"},
    # Physical & lifestyle
    {"id": "sleep",            "label": "Sleep Problems",        "emoji": "🌙", "description": "Insomnia, nightmares, exhaustion", "category": "lifestyle"},
    {"id": "adhd",             "label": "ADHD & Focus",          "emoji": "🎯", "description": "Distraction, impulsivity, overwhelm", "category": "lifestyle"},
    {"id": "eating",           "label": "Eating & Body Image",   "emoji": "🍃", "description": "Difficult relationship with food or body", "category": "lifestyle"},
]

# ================== AGENT PROMPTS ==================

PHASE_1_PROMPT = """
# AGENT IDENTITY
You are the user's AI companion — warm, curious, and non-clinical. You're not a therapist.
You're the kind of presence that actually listens and remembers things.

The user has just selected their condition(s) from a card selector. You already know what they're
dealing with — don't ask them to repeat it. Now your job is to help them shape YOU as a companion.

# YOUR MISSION (PHASE 1: COMPANION CALIBRATION)
You know their condition(s). Your job now:
1. Acknowledge what they're going through — ONE sentence, warm and specific to their condition
2. Ask them what kind of support feels right (listen / challenge / distract / just be there)
3. Ask if they want to give you a name and pick a vibe (calm, warm, direct)
4. Extract those preferences

# RESPONSE FORMAT — valid JSON only, no markdown outside it:
{
  "message": "Your warm 3-4 sentence opener + questions",
  "extracted_data": {
    "companion_name": "name they chose or null",
    "companion_persona": "calm | warm | direct | null",
    "support_style": "listener | challenger | distractor | presence | null",
    "topics_to_avoid": ["topic1"] or []
  },
  "ready_for_next_phase": false
}

Set ready_for_next_phase to true only once you have companion_name AND companion_persona AND support_style.
Keep asking naturally if still missing any of these — one question at a time, conversational.

# TONE RULES
- Never clinical ("symptoms", "disorder", "treatment")
- Short messages — 3-5 sentences max
- Warm, like a friend who gets it
- Never say "I understand how you feel" — too generic
"""

PHASE_2_PROMPT = """
# AGENT IDENTITY
You are their companion. You know their condition(s) and how they want to be supported.
Now you're building your memory — the things that make every future conversation feel personal.

# YOUR MISSION (PHASE 2: MEMORY SEEDING)
1. Ask about the people in their life who matter (or stress them out)
2. Ask what's actually going on for them right now — one real situation
3. Ask what they're hoping for or working toward
4. Keep it conversational — one question at a time

# RESPONSE FORMAT — valid JSON only:
{
  "message": "Your conversational message (3-4 sentences max)",
  "extracted_data": {
    "emotional_state": "how they seem right now in 2-3 words",
    "important_people": ["person + context, e.g. 'Maya - best friend they're fighting with'"],
    "ongoing_situations": ["situation in 1 sentence"],
    "goals": ["what they're working toward"]
  },
  "ready_for_next_phase": false
}

Set ready_for_next_phase to true once you have at least one entry each in
important_people, ongoing_situations, and goals.
emotional_state you infer from how they write — don't ask directly.
"""

PHASE_3_PROMPT = """
# AGENT IDENTITY
You are their companion, almost fully configured. Last step — figuring out when to show up.

# YOUR MISSION (PHASE 3: RHYTHM & AVAILABILITY)
1. Ask when they usually feel worst (time of day / situations)
2. Ask how often they'd want to check in — daily, few times a week, whenever they need
3. Ask if they want gentle nudges or to come to you on their own
4. Keep it short, no pressure

# RESPONSE FORMAT — valid JSON only:
{
  "message": "Your conversational message (3-4 sentences max)",
  "extracted_data": {
    "check_in_frequency": "daily | few_times_week | on_demand",
    "best_times": ["morning", "late night"] etc.,
    "current_stressors": ["brief stressor description"],
    "notification_preference": "gentle_nudges | only_when_i_come | both"
  },
  "ready_for_next_phase": false
}

Set ready_for_next_phase to true once you have check_in_frequency AND notification_preference.
"""

PHASE_4_CONFIRMATION_PROMPT = """
# AGENT IDENTITY
You are their companion. Everything is set. Now you present yourself — fully formed — and ask
if they're ready to begin.

# YOUR MISSION (PHASE 4: COMPANION BORN)
Review ALL collected data and send a final message that:
1. Introduces yourself by the name they chose (or a default if none given)
2. Shows you already know them — reference their condition(s), one person they mentioned,
   one situation they shared, and their goal
3. Makes it feel like you were always here — not "setup complete"
4. Ends with "Ready when you are." — nothing more

# RESPONSE FORMAT — valid JSON only:
{
  "message": "Your personal intro message (5-6 sentences, specific, warm)",
  "confirmation_summary": {
    "companion_name": "final name",
    "companion_persona": "calm | warm | direct",
    "conditions": ["condition ids"],
    "support_style": "their preference",
    "check_in_frequency": "their preference",
    "memory_snapshot": "1-2 sentence summary of what you now remember about them"
  },
  "ready_to_activate": false
}

When user confirms (yes / ready / let's go / sounds right):
{
  "message": "One line. Warm. Like you've been waiting.",
  "ready_to_activate": true
}
"""

PHASE_PROMPTS = {
    1: PHASE_1_PROMPT,
    2: PHASE_2_PROMPT,
    3: PHASE_3_PROMPT,
    4: PHASE_4_CONFIRMATION_PROMPT,
}

# ================== HELPERS ==================

def extract_json_from_response(text):
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def store_extracted(session_state, extracted, phase):
    phase_key = f"phase_{phase}"
    if phase_key not in session_state["phase_data"]:
        session_state["phase_data"][phase_key] = {}
    for k, v in extracted.items():
        if v and v != "null" and v is not None:
            session_state["phase_data"][phase_key][k] = v


def initialize_companion_profile(session_state):
    """
    Replaces generate_5_day_plan().
    Assembles the companion_profile object from all phase data.
    Stored in Firestore under users/{user_id}/companion/profile.
    """
    p0 = session_state["phase_data"].get("phase_0", {})   # condition selector
    p1 = session_state["phase_data"].get("phase_1", {})   # calibration
    p2 = session_state["phase_data"].get("phase_2", {})   # memory seeding
    p3 = session_state["phase_data"].get("phase_3", {})   # rhythm

    conditions         = p0.get("conditions", [])
    primary_condition  = p0.get("primary_condition", conditions[0] if conditions else "general")
    custom_issue       = p0.get("custom_issue", None)

    companion_name     = p1.get("companion_name") or "Luna"
    companion_persona  = p1.get("companion_persona") or "warm"
    support_style      = p1.get("support_style") or "listener"
    topics_to_avoid    = p1.get("topics_to_avoid") or []

    emotional_state    = p2.get("emotional_state") or "uncertain"
    important_people   = p2.get("important_people") or []
    ongoing_situations = p2.get("ongoing_situations") or []
    goals              = p2.get("goals") or []

    check_in_frequency       = p3.get("check_in_frequency") or "on_demand"
    best_times               = p3.get("best_times") or []
    current_stressors        = p3.get("current_stressors") or []
    notification_preference  = p3.get("notification_preference") or "only_when_i_come"

    return {
        "companion_name":    companion_name,
        "companion_persona": companion_persona,
        "support_style":     support_style,
        "topics_to_avoid":   topics_to_avoid,
        "conditions": {
            "all":     conditions,
            "primary": primary_condition,
            "custom":  custom_issue,
        },
        "memory": {
            "emotional_state":    emotional_state,
            "important_people":   important_people,
            "ongoing_situations": ongoing_situations,
            "goals":              goals,
            "current_stressors":  current_stressors,
        },
        "schedule": {
            "check_in_frequency":      check_in_frequency,
            "best_times":              best_times,
            "notification_preference": notification_preference,
        },
        "sessions":    0,
        "created_at":  datetime.utcnow().isoformat(),
        "status":      "active",
    }


def _build_llm_payload(phase, form_data, session_state):
    """Build the Groq request payload for a given phase + form data."""
    prompt_template = PHASE_PROMPTS.get(phase, PHASE_1_PROMPT)
    system_content = (
        f"{prompt_template}\n\n"
        f"USER'S CONDITIONS (from card selector):\n"
        f"{json.dumps(session_state['phase_data'].get('phase_0', {}), indent=2)}\n\n"
        f"PHASE {phase} INPUT:\n"
        f"{json.dumps(form_data, indent=2)}\n\n"
        f"PREVIOUSLY COLLECTED DATA:\n"
        f"{json.dumps(session_state.get('phase_data', {}), indent=2)}\n\n"
        "Analyze the input, extract the required information, respond in the specified JSON format."
    )
    return {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.7,
        "max_tokens": 1000,
        "messages": [{"role": "system", "content": system_content}],
    }


def _save_session_background(session_ref, update_data, companion_profile, user_id):
    """Fire-and-forget Firestore write."""
    try:
        if companion_profile:
            update_data["companion_activated"] = True
            update_data["companion_profile"] = companion_profile
            # Store companion profile under users/{user_id}/companion/profile
            companion_ref = (
                db.collection("users")
                .document(user_id)
                .collection("companion")
                .document("profile")
            )
            companion_ref.set(companion_profile, merge=True)
        session_ref.set(update_data, merge=True)
    except Exception as e:
        print(f"[BACKGROUND SAVE ERROR] {e}")


# ================== JOB STATUS HELPERS ==================

def _set_job_status(job_id, status, payload=None):
    doc = {"status": status, "updated_at": firestore.SERVER_TIMESTAMP}
    if payload:
        doc.update(payload)
    db.collection("jobs").document(job_id).set(doc, merge=True)


def _get_job(job_id):
    doc = db.collection("jobs").document(job_id).get()
    return doc.to_dict() if doc.exists else None


# ================== BACKGROUND WORKER ==================

def _process_job(job_id, user_id, phase, form_data, api_key):
    """
    Runs in a background thread.
    Calls Groq (streaming), accumulates the full response, then:
    - stores extracted data
    - advances phase
    - writes result to Firestore jobs/{job_id}
    The SSE endpoint reads jobs/{job_id} to push tokens to the client.
    """
    try:
        # ── 1. Load session ──────────────────────────────────────────────────
        session_ref = db.collection("sessions").document(user_id)
        session_doc = session_ref.get()
        session_state = session_doc.to_dict() if session_doc.exists else {
            "phase": phase,
            "user_id": user_id,
            "phase_data": {f"phase_{i}": {} for i in range(0, 5)},
            "messages": [],
            "forms_completed": [],
        }

        payload = _build_llm_payload(phase, form_data, session_state)

        # ── 2. Stream from Groq ──────────────────────────────────────────────
        full_text = ""
        token_buffer = []
        flush_every = 5

        with httpx.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={**payload, "stream": True},
            timeout=60.0,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        full_text += token
                        token_buffer.append(token)
                        if len(token_buffer) >= flush_every:
                            db.collection("jobs").document(job_id).set(
                                {"status": "streaming", "partial": full_text},
                                merge=True,
                            )
                            token_buffer = []
                except (json.JSONDecodeError, KeyError):
                    continue

        # ── 3. Parse final JSON ──────────────────────────────────────────────
        parsed = extract_json_from_response(full_text)
        if not parsed:
            _set_job_status(job_id, "error", {"error": "Failed to parse LLM response", "raw": full_text})
            return

        extracted_data = parsed.get("extracted_data", {})
        if extracted_data:
            store_extracted(session_state, extracted_data, phase)

        forms_completed = session_state.get("forms_completed", [])
        if phase not in forms_completed:
            forms_completed.append(phase)

        ready_for_next = parsed.get("ready_for_next_phase", False)
        companion_profile = None

        if ready_for_next:
            # phases 1→2→3→4, after phase 3 confirmed → 5 (done)
            next_phase = phase + 1
        else:
            next_phase = phase

        # Phase 4 confirmation fires companion profile generation
        if phase == 4 and parsed.get("ready_to_activate", False):
            companion_profile = initialize_companion_profile(session_state)
            next_phase = 5

        # ── 4. Write final result to jobs doc ────────────────────────────────
        _set_job_status(job_id, "done", {
            "message": parsed.get("message", ""),
            "partial": full_text,
            "next_phase": next_phase,
            "extracted": extracted_data,
            "companion_profile": companion_profile,
            "confirmation_summary": parsed.get("confirmation_summary"),
            "phase_data": session_state["phase_data"],
        })

        # ── 5. Persist session (background) ──────────────────────────────────
        update_data = {
            "phase": next_phase,
            "user_id": user_id,
            "phase_data": session_state["phase_data"],
            "forms_completed": forms_completed,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        threading.Thread(
            target=_save_session_background,
            args=(session_ref, update_data, companion_profile, user_id),
            daemon=True,
        ).start()

        # ── 6. Prefetch next phase ────────────────────────────────────────────
        if next_phase <= 4:
            threading.Thread(
                target=_prefetch_next_phase,
                args=(user_id, next_phase),
                daemon=True,
            ).start()

    except httpx.TimeoutException:
        _set_job_status(job_id, "error", {"error": "LLM request timed out"})
    except Exception as e:
        _set_job_status(job_id, "error", {"error": str(e), "traceback": traceback.format_exc()})


def _prefetch_next_phase(user_id, next_phase):
    try:
        session_doc = db.collection("sessions").document(user_id).get()
        if session_doc.exists:
            db.collection("prefetch").document(user_id).set({
                "session": session_doc.to_dict(),
                "warmed_at": firestore.SERVER_TIMESTAMP,
                "next_phase": next_phase,
            })
    except Exception as e:
        print(f"[PREFETCH ERROR] {e}")


# ================== ENDPOINTS ==================

@app.route("/init-session", methods=["POST"])
def init_session():
    data = request.json
    user_id = data.get("user_id", "anonymous")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    try:
        session_ref = db.collection("sessions").document(user_id)
        session_doc = session_ref.get()

        if session_doc.exists:
            existing = session_doc.to_dict()
            return jsonify({
                "success": True,
                "message": "Welcome back.",
                "phase": existing.get("phase", 1),
                "user_id": user_id,
                "reconnected": True,
                "condition_catalog": CONDITION_CATALOG,
            })

        session_data = {
            "phase": 1,
            "user_id": user_id,
            "phase_data": {f"phase_{i}": {} for i in range(0, 5)},
            "messages": [],
            "forms_completed": [],
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        session_ref.set(session_data)

        return jsonify({
            "success": True,
            "phase": 1,
            "user_id": user_id,
            "reconnected": False,
            # Frontend uses this to populate the condition selector cards
            "condition_catalog": CONDITION_CATALOG,
        })

    except Exception as e:
        return jsonify({
            "error": "Session initialization failed",
            "details": str(e),
            "traceback": traceback.format_exc(),
        }), 500


# ── NEW: receive condition selector result (phase 0, no LLM needed) ──────────
@app.route("/select-conditions", methods=["POST"])
def select_conditions():
    """
    Called when the user taps Done on the condition selector screen.
    No LLM involved — just stores selections and advances to phase 1.

    Body:
    {
      "user_id": "...",
      "conditions": ["anxiety", "loneliness"],   // selected condition ids
      "primary_condition": "anxiety",             // the one they marked as primary
      "custom_issue": "I struggle with..."        // optional free text
    }
    """
    data = request.json
    user_id          = data.get("user_id")
    conditions       = data.get("conditions", [])
    primary_condition = data.get("primary_condition")
    custom_issue     = data.get("custom_issue", "")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not conditions:
        return jsonify({"error": "at least one condition required"}), 400
    if not primary_condition:
        primary_condition = conditions[0]

    try:
        session_ref = db.collection("sessions").document(user_id)
        session_doc = session_ref.get()

        if session_doc.exists:
            session_state = session_doc.to_dict()
        else:
            session_state = {
                "phase": 1,
                "user_id": user_id,
                "phase_data": {f"phase_{i}": {} for i in range(0, 5)},
                "messages": [],
                "forms_completed": [],
            }

        # Store condition data in phase_0
        session_state["phase_data"]["phase_0"] = {
            "conditions": conditions,
            "primary_condition": primary_condition,
            "custom_issue": custom_issue if custom_issue else None,
        }

        session_ref.set({
            **session_state,
            "phase": 1,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        return jsonify({
            "success": True,
            "phase": 1,
            "conditions": conditions,
            "primary_condition": primary_condition,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to save conditions: {e}"}), 500


# ── Enqueue job, return immediately ──────────────────────────────────────────
@app.route("/submit-phase-data", methods=["POST"])
def submit_phase_data():
    """
    Returns a job_id in < 50ms.
    The actual LLM call happens in a background thread.
    Client polls /job-status/<job_id> or connects to /stream/<job_id>.
    """
    data = request.json
    user_id   = data.get("user_id")
    phase     = data.get("phase")
    form_data = data.get("form_data")
    api_key   = data.get("api_key")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not api_key:
        return jsonify({"error": "api_key required"}), 400
    if not form_data:
        return jsonify({"error": "form_data required"}), 400
    if not phase:
        return jsonify({"error": "phase required"}), 400

    job_id = str(uuid.uuid4())

    # Write pending job to Firestore (fast, small doc)
    db.collection("jobs").document(job_id).set({
        "status": "pending",
        "user_id": user_id,
        "phase": phase,
        "created_at": firestore.SERVER_TIMESTAMP,
    })

    # Kick off background worker
    threading.Thread(
        target=_process_job,
        args=(job_id, user_id, phase, form_data, api_key),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "status": "pending"}), 202


# ── SSE stream — client connects here after getting job_id ───────────────────
@app.route("/stream/<job_id>")
def stream_job(job_id):
    """
    Server-Sent Events endpoint.
    Polls the Firestore jobs doc and pushes tokens to the client as they arrive.
    Closes the stream when status == "done" or "error".
    """
    import time

    def generate():
        last_len = 0
        poll_ms  = 0.25
        max_wait_s = 90
        elapsed  = 0

        yield "retry: 1000\n\n"

        while elapsed < max_wait_s:
            try:
                job = _get_job(job_id)
            except Exception:
                yield "event: error\ndata: firestore read failed\n\n"
                return

            if not job:
                yield "event: error\ndata: job not found\n\n"
                return

            status  = job.get("status", "pending")
            partial = job.get("partial", "")

            if len(partial) > last_len:
                new_tokens = partial[last_len:]
                yield f"data: {json.dumps({'token': new_tokens})}\n\n"
                last_len = len(partial)

            if status == "done":
                final = {
                    "done":                 True,
                    "message":              job.get("message", ""),
                    "next_phase":           job.get("next_phase"),
                    "extracted":            job.get("extracted", {}),
                    "companion_profile":    job.get("companion_profile"),
                    "confirmation_summary": job.get("confirmation_summary"),
                    "phase_data":           job.get("phase_data", {}),
                }
                yield f"event: done\ndata: {json.dumps(final)}\n\n"
                return

            if status == "error":
                yield f"event: error\ndata: {json.dumps({'error': job.get('error', 'Unknown error')})}\n\n"
                return

            time.sleep(poll_ms)
            elapsed += poll_ms

        yield f"event: error\ndata: {json.dumps({'error': 'stream timeout'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Lightweight poll endpoint (alternative to SSE for mobile clients) ─────────
@app.route("/job-status/<job_id>")
def job_status(job_id):
    """
    Simple polling fallback. Client hits this every ~500ms if SSE isn't supported.
    """
    try:
        job = _get_job(job_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not job:
        return jsonify({"error": "job not found"}), 404

    return jsonify({
        "status":               job.get("status"),
        "partial":              job.get("partial", ""),
        "message":              job.get("message", ""),
        "next_phase":           job.get("next_phase"),
        "extracted":            job.get("extracted", {}),
        "companion_profile":    job.get("companion_profile"),
        "confirmation_summary": job.get("confirmation_summary"),
        "phase_data":           job.get("phase_data", {}),
        "error":                job.get("error"),
    })


# ── /chat: conversational phases 1-4 + confirmation ──────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json
    user_id      = data.get("user_id")
    user_message = data.get("message", "")
    api_key      = data.get("api_key")

    if not user_id or not api_key:
        return jsonify({"error": "user_id and api_key required"}), 400

    try:
        session_ref  = db.collection("sessions").document(user_id)
        session_doc  = session_ref.get()
        if not session_doc.exists:
            session_state = {
                "phase": 1,
                "user_id": user_id,
                "phase_data": {f"phase_{i}": {} for i in range(0, 5)},
                "messages": [],
                "forms_completed": [],
            }
            session_ref.set(session_state)
        else:
            session_state = session_doc.to_dict()
    except Exception as e:
        return jsonify({"error": f"Failed to load session: {e}"}), 500

    phase = session_state.get("phase", 1)

    # ── Phase 5: companion active, hand off to main sessions flow ────────────
    if phase == 5:
        return jsonify({
            "response": "Your companion is ready. Head to the sessions page to start talking.",
            "phase": 5,
            "complete": True,
        })

    # ── Phase 4: confirmation conversation ───────────────────────────────────
    if phase == 4:
        confirm_words = ["yes", "looks good", "let's go", "ready", "confirm", "yep", "yeah", "sounds right"]
        modify_words  = ["no", "change", "modify", "different", "wait"]

        if any(w in user_message.lower() for w in confirm_words):
            try:
                companion_profile = initialize_companion_profile(session_state)
                companion_ref = (
                    db.collection("users")
                    .document(user_id)
                    .collection("companion")
                    .document("profile")
                )
                companion_ref.set(companion_profile, merge=True)
                session_ref.update({
                    "phase": 5,
                    "companion_activated": True,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                })
                return jsonify({
                    "response": "I'm here. Let's go.",
                    "phase": 5,
                    "companion_activated": True,
                    "companion_profile": companion_profile,
                    "complete": True,
                })
            except Exception as e:
                return jsonify({"error": "Failed to activate companion", "details": str(e)}), 500

        if any(w in user_message.lower() for w in modify_words):
            return jsonify({
                "response": "No problem — what do you want to change? Your name for me, how I support you, or something else?",
                "phase": 4,
                "awaiting_modification": True,
            })

        # Generate confirmation summary via LLM
        try:
            context = (
                f"{PHASE_4_CONFIRMATION_PROMPT}\n\n"
                f"COLLECTED DATA:\n{json.dumps(session_state.get('phase_data', {}), indent=2)}\n\n"
                f"USER MESSAGE: {user_message}\n\n"
                "Generate a confirmation summary based on all collected data."
            )
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "temperature": 0.7,
                    "max_tokens": 600,
                    "messages": [{"role": "system", "content": context}],
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed  = extract_json_from_response(content)

            if not parsed:
                return jsonify({"response": "Here's what I know about you so far. Sound right?", "phase": 4})

            return jsonify({
                "response":             parsed.get("message", ""),
                "confirmation_summary": parsed.get("confirmation_summary", {}),
                "phase": 4,
                "ready_to_activate": parsed.get("ready_to_activate", False),
            })
        except Exception as e:
            return jsonify({"error": "AI processing failed", "details": str(e)}), 500

    # ── Phases 1–3: regular companion onboarding chat ────────────────────────
    try:
        messages = session_state.get("messages", [])
        messages.append({"role": "user", "content": user_message})

        prompt_text = PHASE_PROMPTS.get(phase, PHASE_1_PROMPT)
        context = (
            f"{prompt_text}\n\n"
            f"CURRENT PHASE: {phase}\n\n"
            f"USER'S CONDITIONS:\n{json.dumps(session_state['phase_data'].get('phase_0', {}), indent=2)}\n\n"
            f"COLLECTED DATA SO FAR:\n{json.dumps(session_state.get('phase_data', {}), indent=2)}\n\n"
            f"RECENT CONVERSATION:\n{json.dumps(messages[-5:], indent=2)}\n\n"
            f"USER'S LATEST MESSAGE: {user_message}\n\n"
            "Respond according to your phase instructions."
        )

        llm_resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.7,
                "max_tokens": 800,
                "messages": [{"role": "system", "content": context}],
            },
            timeout=30.0,
        )
        llm_resp.raise_for_status()
        llm_content = llm_resp.json()["choices"][0]["message"]["content"]
        parsed = extract_json_from_response(llm_content)

        if not parsed:
            messages.append({"role": "assistant", "content": llm_content})
            session_ref.update({"messages": messages, "updated_at": firestore.SERVER_TIMESTAMP})
            return jsonify({"response": llm_content, "phase": phase, "phase_data": session_state.get("phase_data", {})})

        ai_reply = parsed.get("message", llm_content)
        messages.append({"role": "assistant", "content": ai_reply})

        extracted_data = parsed.get("extracted_data", {})
        if extracted_data:
            store_extracted(session_state, extracted_data, phase)

        if parsed.get("ready_for_next_phase"):
            new_phase = phase + 1
            session_ref.update({
                "phase":      new_phase,
                "messages":   messages,
                "phase_data": session_state["phase_data"],
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            return jsonify({
                "response":     ai_reply,
                "phase":        new_phase,
                "phase_data":   session_state.get("phase_data", {}),
                "phase_complete": True,
            })

        session_ref.update({
            "messages":   messages,
            "phase_data": session_state["phase_data"],
            "updated_at": firestore.SERVER_TIMESTAMP,
        })
        return jsonify({
            "response":       ai_reply,
            "phase":          phase,
            "phase_data":     session_state.get("phase_data", {}),
            "needs_more_info": parsed.get("needs_more_info", True),
        })

    except Exception as e:
        return jsonify({
            "error":             "AI processing failed",
            "details":           str(e),
            "traceback":         traceback.format_exc(),
            "fallback_response": "Sorry, I hit a snag. Can you say that again?",
        }), 500


@app.route("/get-session-status", methods=["POST"])
def get_session_status():
    data    = request.json
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    try:
        session_doc = db.collection("sessions").document(user_id).get()
        if not session_doc.exists:
            return jsonify({"error": "Session not found"}), 404
        s = session_doc.to_dict()
        return jsonify({
            "user_id":         user_id,
            "phase":           s.get("phase", 1),
            "phase_data":      s.get("phase_data", {}),
            "forms_completed": s.get("forms_completed", []),
            "message_count":   len(s.get("messages", [])),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get session: {e}"}), 500
        

# ============================================================
# TASK MANAGEMENT ENDPOINTS
# Add these to your existing app.py
# These operate on: users/{user_id}/datedcourses/life_skills
# ============================================================

from flask import Blueprint
from datetime import datetime
import uuid

# ----------------------------------------------------------------
# HELPER: Get the life_skills course ref for a user
# ----------------------------------------------------------------
def get_life_skills_ref(user_id):
    return db.collection("users").document(user_id).collection("datedcourses").document("life_skills")


# ================================================================
# ENDPOINT 1: GET ALL TASKS
# GET-style via POST — returns full task_overview for the user
# ================================================================
@app.route("/tasks/get", methods=["POST"])
def get_tasks():
    """
    Fetch all tasks (and days) from the user's life_skills plan.

    Body: { "user_id": "..." }

    Returns:
    {
        "tasks": [...],   # flat task list
        "days":  [...],   # day-by-day breakdown
        "user_context": {...}
    }
    """
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        doc = get_life_skills_ref(user_id).get()
        if not doc.exists:
            return jsonify({"error": "No plan found. Complete the onboarding first."}), 404

        task_overview = doc.to_dict().get("task_overview", {})

        return jsonify({
            "success": True,
            "tasks": task_overview.get("tasks", []),
            "days": task_overview.get("days", []),
            "user_context": task_overview.get("user_context", {})
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ENDPOINT 2: EDIT AN EXISTING TASK
# Allows editing title, description, scheduled_time, location,
# comfortLevel, estimatedTime, or marking done/undone.
# ================================================================
@app.route("/tasks/edit", methods=["POST"])
def edit_task():
    """
    Edit any field of an existing task.

    Body:
    {
        "user_id":      "...",
        "task_id":      "day1_task",          # id field on the task
        "updates": {
            "title":          "New title",       # optional
            "description":    "New desc",        # optional
            "scheduled_time": "09:30",           # optional  HH:MM
            "scheduled_date": "2025-04-20",      # optional  YYYY-MM-DD
            "location":       "Starbucks",       # optional
            "estimatedTime":  "20 minutes",      # optional
            "comfortLevel":   "easy",            # optional: easy | moderate | challenging
            "done":           true               # optional
        }
    }
    """
    data = request.get_json()
    user_id  = data.get("user_id")
    task_id  = data.get("task_id")
    updates  = data.get("updates", {})

    if not user_id or not task_id:
        return jsonify({"error": "user_id and task_id are required"}), 400

    if not updates:
        return jsonify({"error": "No updates provided"}), 400

    # Whitelist editable fields to prevent accidental overwrites
    EDITABLE_FIELDS = {
        "title", "description", "scheduled_time", "scheduled_date",
        "location", "estimatedTime", "comfortLevel", "done"
    }
    invalid_fields = set(updates.keys()) - EDITABLE_FIELDS
    if invalid_fields:
        return jsonify({
            "error": f"Fields not editable: {invalid_fields}. Allowed: {EDITABLE_FIELDS}"
        }), 400

    # Validate scheduled_time format if provided
    if "scheduled_time" in updates:
        try:
            datetime.strptime(updates["scheduled_time"], "%H:%M")
        except ValueError:
            return jsonify({"error": "scheduled_time must be in HH:MM format (e.g. 09:30)"}), 400

    # Validate scheduled_date format if provided
    if "scheduled_date" in updates:
        try:
            datetime.strptime(updates["scheduled_date"], "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "scheduled_date must be YYYY-MM-DD format"}), 400

    # Validate comfortLevel if provided
    if "comfortLevel" in updates and updates["comfortLevel"] not in {"easy", "moderate", "challenging"}:
        return jsonify({"error": "comfortLevel must be: easy | moderate | challenging"}), 400

    try:
        ref = get_life_skills_ref(user_id)
        doc = ref.get()

        if not doc.exists:
            return jsonify({"error": "Plan not found"}), 404

        doc_data     = doc.to_dict()
        task_overview = doc_data.get("task_overview", {})
        tasks        = task_overview.get("tasks", [])
        days         = task_overview.get("days", [])

        # ── Update in flat tasks list ──────────────────────────────
        task_found = False
        for task in tasks:
            if task.get("id") == task_id:
                task.update(updates)
                task["last_edited"] = datetime.utcnow().isoformat()
                task_found = True
                break

        if not task_found:
            return jsonify({"error": f"Task '{task_id}' not found"}), 404

        # ── Mirror done-status into days breakdown ─────────────────
        if "done" in updates:
            for day in days:
                for day_task in day.get("tasks", []):
                    if day_task.get("task_number") and task_id.startswith(f"day{day.get('day')}_"):
                        day_task["done"] = updates["done"]

        # ── Recalculate completion rate ────────────────────────────
        total     = len(tasks)
        completed = sum(1 for t in tasks if t.get("done"))
        completion_rate = round((completed / total * 100), 1) if total > 0 else 0.0

        task_overview["tasks"] = tasks
        task_overview["days"]  = days

        ref.update({
            "task_overview":   task_overview,
            "completion_rate": completion_rate,
            "last_updated":    datetime.utcnow().isoformat()
        })

        return jsonify({
            "success":         True,
            "updated_task":    next(t for t in tasks if t["id"] == task_id),
            "completion_rate": completion_rate,
            "message":         f"Task '{task_id}' updated successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ENDPOINT 3: ADD A NEW TASK
# Adds a custom task to a specific day (1-5).
# ================================================================
@app.route("/tasks/add", methods=["POST"])
def add_task():
    """
    Add a brand-new custom task to the plan.

    Body:
    {
        "user_id":        "...",
        "day":            2,                        # which day (1-5)
        "title":          "Practice at the mall",
        "description":    "Go to the food court and ask someone a question",
        "scheduled_time": "14:00",                  # HH:MM — optional
        "scheduled_date": "2025-04-21",             # YYYY-MM-DD — optional
        "location":       "Mall food court",        # optional
        "estimatedTime":  "30 minutes",             # optional
        "comfortLevel":   "moderate",               # easy | moderate | challenging
        "xp":             30                        # optional, defaults to 30
    }
    """
    data = request.get_json()
    user_id  = data.get("user_id")
    day      = data.get("day")
    title    = data.get("title", "").strip()
    desc     = data.get("description", "").strip()

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not day or not isinstance(day, int) or day < 1 or day > 5:
        return jsonify({"error": "day must be an integer between 1 and 5"}), 400
    if not title:
        return jsonify({"error": "title is required"}), 400

    # Validate optional time/date fields
    scheduled_time = data.get("scheduled_time", "")
    scheduled_date = data.get("scheduled_date", "")

    if scheduled_time:
        try:
            datetime.strptime(scheduled_time, "%H:%M")
        except ValueError:
            return jsonify({"error": "scheduled_time must be HH:MM"}), 400

    if scheduled_date:
        try:
            datetime.strptime(scheduled_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "scheduled_date must be YYYY-MM-DD"}), 400

    comfort = data.get("comfortLevel", "moderate")
    if comfort not in {"easy", "moderate", "challenging"}:
        return jsonify({"error": "comfortLevel must be: easy | moderate | challenging"}), 400

    try:
        ref = get_life_skills_ref(user_id)
        doc = ref.get()

        if not doc.exists:
            return jsonify({"error": "Plan not found. Complete onboarding first."}), 404

        doc_data      = doc.to_dict()
        task_overview = doc_data.get("task_overview", {})
        tasks         = task_overview.get("tasks", [])
        days          = task_overview.get("days", [])

        # ── Build new task object ──────────────────────────────────
        task_id = f"day{day}_custom_{uuid.uuid4().hex[:8]}"

        new_task = {
            "id":             task_id,
            "title":          title,
            "description":    desc,
            "done":           False,
            "xp":             data.get("xp", 30),
            "scheduled_time": scheduled_time,
            "scheduled_date": scheduled_date,
            "location":       data.get("location", ""),
            "estimatedTime":  data.get("estimatedTime", ""),
            "comfortLevel":   comfort,
            "type":           "custom_task",
            "difficulty":     day,
            "skill_focus":    task_overview.get("user_context", {}).get("skill_gaps", ""),
            "contextAnchor":  task_overview.get("user_context", {}).get("problem", ""),
            "is_custom":      True,
            "created_at":     datetime.utcnow().isoformat()
        }

        tasks.append(new_task)

        # ── Inject into the correct day breakdown ──────────────────
        day_found = False
        for d in days:
            if d.get("day") == day:
                d["tasks"].append({
                    "task_number": len(d["tasks"]) + 1,
                    "description": desc,
                    "done":        False,
                    "task_id":     task_id
                })
                day_found = True
                break

        # If somehow the day doesn't exist yet, create it
        if not day_found:
            days.append({
                "day":   day,
                "title": f"Day {day}: Custom",
                "tasks": [{
                    "task_number": 1,
                    "description": desc,
                    "done":        False,
                    "task_id":     task_id
                }],
                "completed": False
            })
            days.sort(key=lambda d: d["day"])

        task_overview["tasks"] = tasks
        task_overview["days"]  = days

        ref.update({
            "task_overview": task_overview,
            "last_updated":  datetime.utcnow().isoformat()
        })

        return jsonify({
            "success":  True,
            "task_id":  task_id,
            "new_task": new_task,
            "message":  f"Task added to Day {day} successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ENDPOINT 4: DELETE A TASK
# ================================================================
@app.route("/tasks/delete", methods=["POST"])
def delete_task():
    """
    Delete a task from the plan.
    Note: only custom tasks (is_custom: true) can be deleted.
    Pass force: true to delete AI-generated tasks too.

    Body:
    {
        "user_id": "...",
        "task_id": "day2_custom_abc12345",
        "force":   false     # set true to delete AI-generated tasks
    }
    """
    data = request.get_json()
    user_id = data.get("user_id")
    task_id = data.get("task_id")
    force   = data.get("force", False)

    if not user_id or not task_id:
        return jsonify({"error": "user_id and task_id are required"}), 400

    try:
        ref = get_life_skills_ref(user_id)
        doc = ref.get()

        if not doc.exists:
            return jsonify({"error": "Plan not found"}), 404

        doc_data      = doc.to_dict()
        task_overview = doc_data.get("task_overview", {})
        tasks         = task_overview.get("tasks", [])
        days          = task_overview.get("days", [])

        # Find the task first
        target = next((t for t in tasks if t.get("id") == task_id), None)
        if not target:
            return jsonify({"error": f"Task '{task_id}' not found"}), 404

        # Guard: prevent accidental deletion of AI-generated tasks
        if not target.get("is_custom") and not force:
            return jsonify({
                "error": "This is an AI-generated task. Pass force: true to delete it.",
                "task":  target
            }), 403

        # Remove from flat list
        tasks = [t for t in tasks if t.get("id") != task_id]

        # Remove from days breakdown
        for day in days:
            day["tasks"] = [
                dt for dt in day.get("tasks", [])
                if dt.get("task_id") != task_id
            ]
            # Re-number remaining tasks
            for idx, dt in enumerate(day["tasks"]):
                dt["task_number"] = idx + 1

        # Recalculate completion rate
        total     = len(tasks)
        completed = sum(1 for t in tasks if t.get("done"))
        completion_rate = round((completed / total * 100), 1) if total > 0 else 0.0

        task_overview["tasks"] = tasks
        task_overview["days"]  = days

        ref.update({
            "task_overview":   task_overview,
            "completion_rate": completion_rate,
            "last_updated":    datetime.utcnow().isoformat()
        })

        return jsonify({
            "success":         True,
            "deleted_task_id": task_id,
            "completion_rate": completion_rate,
            "message":         f"Task '{task_id}' deleted successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ENDPOINT 5: ASSIGN / UPDATE TIME FOR A TASK
# Convenience shortcut — just for updating schedule fields.
# ================================================================
@app.route("/tasks/schedule", methods=["POST"])
def schedule_task():
    """
    Assign or update the time (and optionally date) for a task.

    Body:
    {
        "user_id":        "...",
        "task_id":        "day1_task",
        "scheduled_time": "08:00",        # required  HH:MM
        "scheduled_date": "2025-04-18"    # optional  YYYY-MM-DD
    }
    """
    data           = request.get_json()
    user_id        = data.get("user_id")
    task_id        = data.get("task_id")
    scheduled_time = data.get("scheduled_time", "").strip()
    scheduled_date = data.get("scheduled_date", "").strip()

    if not user_id or not task_id or not scheduled_time:
        return jsonify({"error": "user_id, task_id, and scheduled_time are required"}), 400

    try:
        datetime.strptime(scheduled_time, "%H:%M")
    except ValueError:
        return jsonify({"error": "scheduled_time must be HH:MM (e.g. 09:30)"}), 400

    if scheduled_date:
        try:
            datetime.strptime(scheduled_date, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "scheduled_date must be YYYY-MM-DD"}), 400

    try:
        ref = get_life_skills_ref(user_id)
        doc = ref.get()

        if not doc.exists:
            return jsonify({"error": "Plan not found"}), 404

        doc_data      = doc.to_dict()
        task_overview = doc_data.get("task_overview", {})
        tasks         = task_overview.get("tasks", [])

        task_found = False
        updated_task = None
        for task in tasks:
            if task.get("id") == task_id:
                task["scheduled_time"] = scheduled_time
                if scheduled_date:
                    task["scheduled_date"] = scheduled_date
                task["last_edited"] = datetime.utcnow().isoformat()
                task_found   = True
                updated_task = task
                break

        if not task_found:
            return jsonify({"error": f"Task '{task_id}' not found"}), 404

        task_overview["tasks"] = tasks

        ref.update({
            "task_overview": task_overview,
            "last_updated":  datetime.utcnow().isoformat()
        })

        return jsonify({
            "success":      True,
            "updated_task": updated_task,
            "message":      f"Task '{task_id}' scheduled for {scheduled_time}" +
                            (f" on {scheduled_date}" if scheduled_date else "")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ENDPOINT 6: BULK SCHEDULE — set times for multiple tasks at once
# ================================================================
@app.route("/tasks/bulk-schedule", methods=["POST"])
def bulk_schedule_tasks():
    """
    Assign scheduled times to multiple tasks in one call.

    Body:
    {
        "user_id": "...",
        "schedules": [
            { "task_id": "day1_task", "scheduled_time": "08:00", "scheduled_date": "2025-04-18" },
            { "task_id": "day2_task", "scheduled_time": "09:30" },
            { "task_id": "day3_task", "scheduled_time": "18:00", "scheduled_date": "2025-04-20" }
        ]
    }
    """
    data      = request.get_json()
    user_id   = data.get("user_id")
    schedules = data.get("schedules", [])

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    if not schedules or not isinstance(schedules, list):
        return jsonify({"error": "schedules must be a non-empty list"}), 400

    # Validate all entries before touching Firestore
    for entry in schedules:
        if not entry.get("task_id") or not entry.get("scheduled_time"):
            return jsonify({"error": "Each schedule entry needs task_id and scheduled_time"}), 400
        try:
            datetime.strptime(entry["scheduled_time"], "%H:%M")
        except ValueError:
            return jsonify({
                "error": f"Invalid scheduled_time '{entry['scheduled_time']}' for task '{entry['task_id']}'. Use HH:MM."
            }), 400
        if entry.get("scheduled_date"):
            try:
                datetime.strptime(entry["scheduled_date"], "%Y-%m-%d")
            except ValueError:
                return jsonify({
                    "error": f"Invalid scheduled_date for task '{entry['task_id']}'. Use YYYY-MM-DD."
                }), 400

    try:
        ref = get_life_skills_ref(user_id)
        doc = ref.get()

        if not doc.exists:
            return jsonify({"error": "Plan not found"}), 404

        doc_data      = doc.to_dict()
        task_overview = doc_data.get("task_overview", {})
        tasks         = task_overview.get("tasks", [])

        # Build a quick lookup
        task_map = {t["id"]: t for t in tasks}

        results   = []
        not_found = []

        for entry in schedules:
            tid = entry["task_id"]
            if tid not in task_map:
                not_found.append(tid)
                continue

            task_map[tid]["scheduled_time"] = entry["scheduled_time"]
            if entry.get("scheduled_date"):
                task_map[tid]["scheduled_date"] = entry["scheduled_date"]
            task_map[tid]["last_edited"] = datetime.utcnow().isoformat()
            results.append(tid)

        # Write updated tasks back
        task_overview["tasks"] = list(task_map.values())

        ref.update({
            "task_overview": task_overview,
            "last_updated":  datetime.utcnow().isoformat()
        })

        return jsonify({
            "success":          True,
            "scheduled_tasks":  results,
            "not_found_tasks":  not_found,
            "message":          f"{len(results)} task(s) scheduled. {len(not_found)} not found."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================================================================
# ENDPOINT 7: REORDER TASKS WITHIN A DAY
# ================================================================
@app.route("/tasks/reorder", methods=["POST"])
def reorder_tasks():
    """
    Reorder tasks for a specific day.

    Body:
    {
        "user_id":       "...",
        "day":           2,
        "ordered_ids":   ["day2_task", "day2_custom_abc12345"]   # desired order
    }
    """
    data        = request.get_json()
    user_id     = data.get("user_id")
    day         = data.get("day")
    ordered_ids = data.get("ordered_ids", [])

    if not user_id or not day or not ordered_ids:
        return jsonify({"error": "user_id, day, and ordered_ids are required"}), 400

    try:
        ref = get_life_skills_ref(user_id)
        doc = ref.get()

        if not doc.exists:
            return jsonify({"error": "Plan not found"}), 404

        doc_data      = doc.to_dict()
        task_overview = doc_data.get("task_overview", {})
        tasks         = task_overview.get("tasks", [])
        days          = task_overview.get("days", [])

        # Build map for quick access
        task_map = {t["id"]: t for t in tasks}

        # Reorder within the day object
        day_obj = next((d for d in days if d.get("day") == day), None)
        if not day_obj:
            return jsonify({"error": f"Day {day} not found in plan"}), 404

        # Validate all IDs exist
        missing = [oid for oid in ordered_ids if oid not in task_map]
        if missing:
            return jsonify({"error": f"Task IDs not found: {missing}"}), 404

        # Reorder the day's task list
        existing_day_tasks = {dt.get("task_id"): dt for dt in day_obj.get("tasks", [])}
        reordered_day_tasks = []
        for idx, oid in enumerate(ordered_ids):
            dt = existing_day_tasks.get(oid, {
                "task_id":     oid,
                "description": task_map[oid].get("description", ""),
                "done":        task_map[oid].get("done", False)
            })
            dt["task_number"] = idx + 1
            reordered_day_tasks.append(dt)

        day_obj["tasks"] = reordered_day_tasks

        task_overview["days"] = days

        ref.update({
            "task_overview": task_overview,
            "last_updated":  datetime.utcnow().isoformat()
        })

        return jsonify({
            "success":        True,
            "day":            day,
            "new_task_order": ordered_ids,
            "message":        f"Day {day} tasks reordered successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    user_id = data.get("user_id")
    
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    
    try:
        session_ref = db.collection("sessions").document(user_id)
        session_ref.delete()
        
        return jsonify({
            "success": True, 
            "message": "Session reset successfully"
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to reset session: {str(e)}"
        }), 500




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

@app.route('/mentor-chat', methods=['POST', 'OPTIONS'])
def mentor_chat():
    """
    AI Mentor chatbot - someone who's been through what you're going through
    Provides empathetic, experience-based advice
    """
    if request.method == 'OPTIONS':
        return '', 204  # Handle preflight
    
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        user_message = data.get("message", "").strip()
        conversation_id = data.get("conversation_id", "")
        
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
            conversation_id = f"mentor_{user_id}_{int(time.time())}"
        
        # Load conversation history from Firebase
        doc_ref = db.collection("mentor_conversations").document(conversation_id)
        doc = doc_ref.get()
        
        if doc.exists:
            history = doc.to_dict().get("messages", [])
        else:
            # First time: load the mentor prompt
            mentor_prompt = load_prompt("prompt_mentor.txt")
            if not mentor_prompt:
                return jsonify({"error": "prompt_mentor.txt not found"}), 500
            
            history = [{"role": "system", "content": mentor_prompt}]
        
        # Append user message to history
        history.append({"role": "user", "content": user_message})
        
        # Call the AI model
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.7,
            max_tokens=400
        )
        
        ai_reply = response.choices[0].message.content.strip()
        
        # Append AI response to history
        history.append({"role": "assistant", "content": ai_reply})
        
        # Save updated conversation to Firebase
        doc_ref.set({
            "messages": history,
            "user_id": user_id,
            "last_updated": firestore.SERVER_TIMESTAMP,
            "conversation_type": "mentor"
        }, merge=True)
        
        # Return response
        return jsonify({
            "success": True,
            "reply": ai_reply,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        import traceback
        print("--- /mentor-chat ERROR ---")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }), 500


@app.route('/mentor-chat/new', methods=['POST', 'OPTIONS'])
def start_new_mentor_chat():
    """Start a fresh mentor conversation"""
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    user_id = data.get("user_id")
    
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    
    conversation_id = f"mentor_{user_id}_{int(time.time())}"
    
    return jsonify({
        "success": True,
        "conversation_id": conversation_id,
        "message": "New conversation started. How can I help today?"
    })


@app.route('/mentor-chat/history/<conversation_id>', methods=['GET'])
def get_mentor_history(conversation_id):
    """Get conversation history"""
    try:
        doc_ref = db.collection("mentor_conversations").document(conversation_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return jsonify({"error": "Conversation not found"}), 404
        
        doc_data = doc.to_dict()
        # Filter out system messages for frontend display
        messages = [
            msg for msg in doc_data.get("messages", [])
            if msg.get("role") != "system"
        ]
        
        return jsonify({
            "success": True,
            "messages": messages,
            "conversation_id": conversation_id
        })
    
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
