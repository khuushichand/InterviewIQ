"""InterviewIQ MVP backend. Works in demo mode without API keys."""
import os, json, ast, subprocess, sys, tempfile, re
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVEN_API_KEY")
VOICE = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MODEL = "gemini-2.5-flash"

client = None
if GEMINI_KEY:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_KEY)

def llm(system: str, prompt: str, fallback: str) -> str:
    if not client:
        return fallback
    try:
        r = client.models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system))
        return r.text.strip()
    except Exception as e:
        return f"{fallback}\n(LLM error: {e})"

app = FastAPI(title="InterviewIQ")

CHALLENGES = [
    {"id": "two_sum", "title": "Two Sum", "fn": "two_sum",
     "prompt": "Return indices of two numbers in a list that add up to target.",
     "starter": "def two_sum(nums, target):\n    # your code\n    pass\n",
     "tests": [[[[2, 7, 11, 15], 9], [0, 1]], [[[3, 2, 4], 6], [1, 2]], [[[3, 3], 6], [0, 1]]]},
    {"id": "palindrome", "title": "Valid Palindrome", "fn": "is_palindrome",
     "prompt": "Check if a string is a palindrome ignoring case and non-alphanumerics.",
     "starter": "def is_palindrome(s):\n    pass\n",
     "tests": [[["A man, a plan, a canal: Panama"], True], [["race a car"], False], [[""], True]]},
]

class ChatReq(BaseModel):
    history: list[dict]  # [{role: 'interviewer'|'candidate', text}]
    role: str = "Software Engineer"
    level: str = "Fresher"
    kind: str = "technical"

@app.post("/api/interview/chat")
def chat(r: ChatReq):
    system = (f"You are a professional {r.kind} interviewer for a {r.level} {r.role} role. "
              "Speak ONLY in English. Ask ONE question at a time, 1-3 sentences. Use the candidate's "
              "last answer for an adaptive follow-up; raise difficulty if they do well, simplify if they struggle.")
    convo = "\n".join(f"{m['role']}: {m['text']}" for m in r.history) or "(start)"
    fb = "Tell me about yourself and a project you are proud of." if not r.history else \
         "Interesting. Can you explain the time complexity of the approach you described?"
    return {"reply": llm(system, convo + "\ninterviewer:", fb)}

class AssistReq(BaseModel):
    text: str
    kind: str = "explain"  # explain | term | hint

@app.post("/api/assist")
def assist(r: AssistReq):
    task = {"explain": "Explain this interview question in simple Kannada (ಕನ್ನಡ). Keep key technical terms in English in brackets.",
            "term": "Explain this technical concept in simple Kannada with one example. Give a short English-Kannada vocabulary list.",
            "hint": "Give ONE small progressive hint in Kannada for this coding problem. Do NOT reveal the full solution."}[r.kind]
    return {"answer": llm(task, r.text, "(Demo mode: set GEMINI_API_KEY for Kannada output.)")}

@app.get("/api/challenges")
def challenges():
    return [{k: v for k, v in c.items() if k != "tests"} for c in CHALLENGES]

class CodeReq(BaseModel):
    challenge_id: str
    code: str

HARNESS = """
import json,sys
{code}
tests=json.loads(sys.stdin.read()); out=[]
for args,exp in tests:
    try: got={fn}(*args); out.append(got==exp or (isinstance(got,list) and sorted(got)==sorted(exp) if isinstance(exp,list) else False))
    except Exception as e: out.append(False)
print(json.dumps(out))
"""

def diagnose(code: str):
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error line {e.lineno}: {e.msg}"], 0
    loops_depth = 0
    def depth(n, d=0):
        m = d
        for c in ast.iter_child_nodes(n):
            m = max(m, depth(c, d + 1 if isinstance(c, (ast.For, ast.While)) else d))
        return m
    loops_depth = depth(tree)
    if loops_depth >= 2: issues.append(f"Nested loops (depth {loops_depth}): likely O(n^{loops_depth}). Can a hash map help?")
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and not ast.get_docstring(n) and len(n.body) > 8:
            issues.append(f"Function '{n.name}' is long; consider splitting.")
        if isinstance(n, ast.ExceptHandler) and n.type is None:
            issues.append("Bare 'except:' hides errors.")
    return issues, loops_depth

@app.post("/api/code/run")
def run_code(r: CodeReq):
    c = next(c for c in CHALLENGES if c["id"] == r.challenge_id)
    issues, d = diagnose(r.code)
    if issues and issues[0].startswith("Syntax"):
        return {"passed": 0, "total": len(c["tests"]), "diagnostics": issues, "error": issues[0]}
    src = HARNESS.format(code=r.code, fn=c["fn"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        p = subprocess.run([sys.executable, f.name], input=json.dumps(c["tests"]),
                           capture_output=True, text=True, timeout=5)
        res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else []
        err = p.stderr[-300:] if p.returncode else ""
    except subprocess.TimeoutExpired:
        res, err = [], "Time limit exceeded (5s)"
    finally:
        os.unlink(f.name)
    return {"passed": sum(res), "total": len(c["tests"]), "diagnostics": issues,
            "est_complexity": f"O(n^{d})" if d else "O(n) or better (heuristic)", "error": err}

class TTSReq(BaseModel):
    text: str

@app.post("/api/tts")
async def tts(r: TTSReq):
    if not ELEVEN_KEY:
        return Response(status_code=204)  # frontend falls back to browser TTS
    async with httpx.AsyncClient(timeout=30) as h:
        resp = await h.post(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}",
            headers={"xi-api-key": ELEVEN_KEY},
            json={"text": r.text, "model_id": "eleven_turbo_v2_5"})
    return Response(resp.content, media_type="audio/mpeg") if resp.status_code == 200 else Response(status_code=204)

FILLERS = ["um", "uh", "like", "you know", "basically", "actually", "so"]

class ReportReq(BaseModel):
    history: list[dict]
    code_passed: int = 0
    code_total: int = 0
    seconds_spoken: float = 0

@app.post("/api/report")
def report(r: ReportReq):
    answers = " ".join(m["text"] for m in r.history if m["role"] == "candidate")
    words = re.findall(r"[a-zA-Z']+", answers.lower())
    n = max(len(words), 1)
    fill = {f: len(re.findall(rf"\b{f}\b", answers.lower())) for f in FILLERS}
    filler_total = sum(fill.values())
    wpm = round(n / (r.seconds_spoken / 60)) if r.seconds_spoken > 5 else None
    code_score = round(100 * r.code_passed / r.code_total) if r.code_total else None
    system = ("Evaluate this interview. Score TECHNICAL knowledge independently from English fluency "
              "(do not penalise grammar in the technical score). Reply with ONLY JSON: "
              '{"technical":0-100,"communication":0-100,"english":0-100,"strengths":[..],"improvements":[..],"better_phrasing":[..]}')
    raw = llm(system, "\n".join(f"{m['role']}: {m['text']}" for m in r.history),
              '{"technical":60,"communication":60,"english":60,"strengths":["Demo mode"],"improvements":["Set GEMINI_API_KEY"],"better_phrasing":[]}')
    try:
        ai = json.loads(re.sub(r"^```json|```$", "", raw.strip(), flags=re.M).strip())
    except Exception:
        ai = {"technical": 0, "communication": 0, "english": 0, "strengths": [], "improvements": [raw[:200]], "better_phrasing": []}
    ai["coding"] = code_score
    ai["fillers"] = {k: v for k, v in fill.items() if v}
    ai["filler_rate_pct"] = round(100 * filler_total / n, 1)
    ai["wpm"] = wpm
    ai["note"] = "Gaze/posture indicators are coaching hints, not measures of confidence."
    return ai

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def index(): return FileResponse("static/index.html")
