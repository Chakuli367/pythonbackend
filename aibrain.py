import json
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS   # <--- FIXED

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========== GROQ CONFIG ==========
GROQ_API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {GROQ_API_KEY}"
}

# ========== AGENT STATE ==========
agent_state = {
    "current_phase": "diagnostic",
    "user_data": {},
    "conversation_history": [],
    "tasks_completed": [],
    "memory": {}
}

# ========== AI QUERY HELPER ==========
def ai_query(prompt, system_msg="You are a helpful assistant.", max_tokens=500):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# ========== AUTO-PHASE AGENT LOGIC ==========
def autonomous_agent(user_input=None):
    phase = agent_state["current_phase"]
    response_payload = {}

    if user_input:
        last_question = agent_state.get("last_question", "general question")
        agent_state["user_data"][f"response_{len(agent_state['user_data'])}"] = {
            "question": last_question,
            "answer": user_input,
            "timestamp": datetime.now().isoformat()
        }
        agent_state["conversation_history"].append({"role": "user", "content": user_input})

    if phase == "diagnostic":
        prompt = f"""
You are an empathetic social coach. The user wants to improve social skills.
User data so far: {json.dumps(agent_state['user_data'], indent=2)}
Ask ONE question to learn more about the user's social habits. Include one supportive, motivating sentence.
Keep it conversational and human-like.
"""
        next_question = ai_query(prompt, "You are an empathetic social coach.", max_tokens=150)
        agent_state["last_question"] = next_question
        response_payload = {"type": "question", "content": next_question}

        if len(agent_state["user_data"]) >= 3:
            agent_state["current_phase"] = "conversation_analysis"

    elif phase == "conversation_analysis":
        prompt = f"""
You are a social coach analyzing user responses.
User responses: {json.dumps(agent_state['user_data'], indent=2)}
Give 1 short insight or key takeaway that will help the user improve socially. 2-3 sentences max.
"""
        insight = ai_query(prompt, "You are a social coach.", max_tokens=150)
        response_payload = {"type": "insight", "content": insight}

        agent_state["current_phase"] = "goal_setting"

    elif phase == "goal_setting":
        prompt = f"""
Based on user responses: {json.dumps(agent_state['user_data'], indent=2)}
Create 3 specific, measurable goals for the user over the next 5 days. Keep it clear and actionable.
"""
        goals = ai_query(prompt, "You are a goal-setting coach.", max_tokens=300)
        agent_state["memory"]["goals"] = goals
        response_payload = {"type": "goals", "content": goals}
        agent_state["current_phase"] = "action_planning"

    elif phase == "action_planning":
        prompt = f"""
User profile and goals: {json.dumps(agent_state, indent=2)}
Create a detailed 5-day action plan. Each day should have one task, why it matters, and how to do it.
Keep it practical and achievable.
"""
        plan = ai_query(prompt, "You are an implementation coach.", max_tokens=800)
        agent_state["memory"]["action_plan"] = plan
        response_payload = {"type": "action_plan", "content": plan}
        agent_state["current_phase"] = "complete"

    elif phase == "complete":
        response_payload = {"type": "complete", "content": "All phases complete. Your plan is ready!"}

    return response_payload

# ========== ENDPOINT ==========
@app.route("/agent", methods=["POST"])
def agent_endpoint():
    data = request.json or {}
    user_input = data.get("answer")
    response = autonomous_agent(user_input)
    return jsonify(response)

# ❌ DO NOT ADD app.run() — Render handles it
