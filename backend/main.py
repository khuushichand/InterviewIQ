"""InterviewIQ MVP backend (FastAPI). Works without API keys using offline fallbacks."""
import os, ast, json, re, subprocess, sys, tempfile, shutil
from pathlib import Path
from dotenv import load_dotenv
import time
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

load_dotenv()
from backend.gemini_service import (
    generate_content_with_retry,
    get_client,
    MODEL_NAME
)

app = FastAPI(title="InterviewIQ")
MODEL = MODEL_NAME

def llm(prompt: str, system_instruction: str | None = None, response_mime_type: str | None = None) -> str | None:
    return generate_content_with_retry(
        contents=prompt,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        model=MODEL
    )

FALLBACK_Q = {
    "English": {
        "Technical": ["Explain the difference between a process and a thread.",
                      "What is the time complexity of searching in a hash map, and why?",
                      "How would you detect a cycle in a linked list?",
                      "What is the difference between an array and a linked list?"],
        "HR": ["Tell me about yourself.", "Describe a challenge you faced in a team project and how you handled it.",
               "Why do you want to join our company?", "Where do you see yourself in three years?"],
        "Behavioural": ["Tell me about a time you had a disagreement with a team member and how you resolved it.",
                        "Describe a situation where you had to meet a tight deadline under pressure.",
                        "Can you share an example of a mistake you made and what you learned from it?"],
        "Communication": ["How would you explain a complex technical concept to a non-technical stakeholder?",
                          "Can you walk me through your recent project architecture in under two minutes?",
                          "How do you handle delivering critical feedback to a colleague?"],
        "Technical + HR": ["Tell me about yourself and your primary technical stack.",
                           "Describe how you design scalable APIs and how you manage collaboration in your team.",
                           "What was the most challenging bug you resolved, and how did you communicate the impact?"],
        "Full Placement Interview": ["Introduce yourself and give a brief overview of your technical background.",
                                    "Walk me through how indexing works in a relational database or your primary domain.",
                                    "Tell me about a time you led an initiative or took ownership of an engineering issue."]
    },
    "Kannada": {
        "Technical": ["ಪ್ರೊಸೆಸ್ ಮತ್ತು ಥ್ರೆಡ್ ನಡುವಿನ ವ್ಯತ್ಯಾಸವನ್ನು ವಿವರಿಸಿ.",
                      "ಹ್ಯಾಶ್ ಮ್ಯಾಪ್‌ನಲ್ಲಿ ಹುಡುಕುವ ಸಮಯ ಸಂಕೀರ್ಣತೆ (Time Complexity) ಏನು ಮತ್ತು ಏಕೆ?",
                      "ಲಿಂಕ್ಡ್ ಲಿಸ್ಟ್‌ನಲ್ಲಿ ಸೈಕಲ್ ಅನ್ನು ಹೇಗೆ ಕಂಡುಹಿಡಿಯುತ್ತೀರಿ?"],
        "HR": ["ನಿಮ್ಮ ಬಗ್ಗೆ ಪರಿಚಯಿಸಿಕೊಳ್ಳಿ.", "ತಂಡದ ಪ್ರಾಜೆಕ್ಟ್‌ನಲ್ಲಿ ನೀವು ಎದುರಿಸಿದ ಸವಾಲು ಮತ್ತು ಅದನ್ನು ಹೇಗೆ ಪರಿಹರಿಸಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ."],
        "Behavioural": ["ತಂಡದ ಸದಸ್ಯರೊಂದಿಗೆ ಭಿನ್ನಾಭಿಪ್ರಾಯ ಬಂದಾಗ ಅದನ್ನು ಹೇಗೆ ಸರಿಪಡಿಸಿಕೊಂಡಿದ್ದೀರಿ?"],
        "Communication": ["ತಾಂತ್ರಿಕ ವಿಷಯವನ್ನು ತಾಂತ್ರಿಕೇತರ ವ್ಯಕ್ತಿಗೆ ಹೇಗೆ ಸರಳವಾಗಿ ವಿವರಿಸುತ್ತೀರಿ?"],
        "Technical + HR": ["ನಿಮ್ಮ ಬಗ್ಗೆ ಮತ್ತು ನಿಮ್ಮ ತಾಂತ್ರಿಕ ಹಿನ್ನೆಲೆಯ ಬಗ್ಗೆ ಸಂಕ್ಷಿಪ್ತವಾಗಿ ತಿಳಿಸಿ."],
        "Full Placement Interview": ["ನಿಮ್ಮ ಪರಿಚಯ ನೀಡಿ ಮತ್ತು ಡೇಟಾಬೇಸ್ ಇಂಡೆಕ್ಸಿಂಗ್ ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ ಎಂದು ವಿವರಿಸಿ."]
    },
    "Hindi": {
        "Technical": ["Process और Thread के बीच क्या अंतर है, समझाइए।",
                      "Hash Map में सर्च करने की Time Complexity क्या होती है और क्यों?",
                      "Linked List में cycle का पता कैसे लगाते हैं?"],
        "HR": ["अपने बारे में बताइए।", "किसी टीम प्रोजेक्ट में आई चुनौती और उसके समाधान के बारे में बताएं।"],
        "Behavioural": ["जब टीम में मतभेद हुआ तो आपने उसे कैसे सुलझाया?"],
        "Communication": ["एक जटिल तकनीकी विषय को किसी गैर-तकनीकी व्यक्ति को कैसे समझाएंगे?"],
        "Technical + HR": ["अपने और अपने तकनीकी कौशल के बारे में संक्षेप में बताएं।"],
        "Full Placement Interview": ["अपना परिचय दें और रिलेशनल डेटाबेस में इंडेक्सिंग कैसे काम करती है बताएं।"]
    },
    "Tamil": {
        "Technical": ["Process மற்றும் Thread இடையேயான வித்தியாசத்தை விளக்குங்கள்.",
                      "Hash Map-ல் தேடுவதற்கான Time Complexity என்ன மற்றும் ஏன்?",
                      "Linked List-ல் cycle-ஐ எவ்வாறு கண்டறிவீர்கள்?"],
        "HR": ["உங்களைப் பற்றி சுருக்கமாக கூறுங்கள்.", "ஒரு குழு திட்டத்தில் நீங்கள் எதிர்கொண்ட சவாலை எவ்வாறு கையாண்டீர்கள்?"],
        "Behavioural": ["குழு உறுப்பினருடன் கருத்து வேறுபாடு ஏற்பட்டபோது அதை எவ்வாறு தீர்த்தீர்கள்?"],
        "Communication": ["ஒரு தொழில்நுட்ப கருத்தை தொழில்நுட்பம் அல்லாத ஒருவருக்கு எவ்வாறு விளக்குவீர்கள்?"],
        "Technical + HR": ["உங்களைப் பற்றியும் உங்கள் தொழில்நுட்ப திறன்களைப் பற்றியும் கூறுங்கள்."],
        "Full Placement Interview": ["உங்களை அறிமுகப்படுத்தி, டேட்டாபேஸ் இன்டெக்சிங் எவ்வாறு செயல்படுகிறது என்பதை விளக்குங்கள்."]
    },
    "Telugu": {
        "Technical": ["Process మరియు Thread మధ్య తేడాను వివరించండి.",
                      "Hash Map లో వెతకడానికి Time Complexity ఎంత మరియు ఎందుకు?",
                      "Linked List లో cycle ని ఎలా గుర్తిస్తారు?"],
        "HR": ["మీ గురించి క్లుప్తంగా చెప్పండి.", "టీమ్ ప్రాజెక్ట్‌లో మీరు ఎదుర్కొన్న సవాలు మరియు దానిని ఎలా పరిష్కరించారో చెప్పండి."],
        "Behavioural": ["టీమ్ సభ్యుడితో భేదాభిప్రాయం వచ్చినప్పుడు మీరు దానిని ఎలా పరిష్కరించారు?"],
        "Communication": ["ఒక సంక్లిష్టమైన సాంకేతిక భావనను సాంకేతికేతర వ్యక్తికి ఎలా వివరిస్తారు?"],
        "Technical + HR": ["మీ గురించి మరియు మీ సాంకేతిక నైపుణ్యాల గురించి క్లుప్తంగా చెప్పండి."],
        "Full Placement Interview": ["మీ పరిచయం ఇవ్వండి మరియు డేటాబేస్ ఇండెక్సింగ్ ఎలా పనిచేస్తుందో వివరించండి."]
    },
    "Malayalam": {
        "Technical": ["Process ഉം Thread ഉം തമ്മിലുള്ള വ്യത്യാസം വിശദീകരിക്കുക.",
                      "Hash Map ൽ തിരയുന്നതിനുള്ള Time Complexity എന്താണ്, എന്തുകൊണ്ട്?",
                      "Linked List ൽ ഒരു cycle എങ്ങനെ കണ്ടെത്താം?"],
        "HR": ["നിങ്ങളെക്കുറിച്ച് സ്വയം പരിചയപ്പെടുത്തുക.", "ഒരു പ്രോജക്റ്റിൽ നിങ്ങൾ നേരിട്ട പ്രധാന വെല്ലുവിളി എന്തായിരുന്നു?"],
        "Behavioural": ["ഒരു ടീം അംഗവുമായി അഭിപ്രായവ്യത്യാസം ഉണ്ടായപ്പോൾ നിങ്ങളത് എങ്ങനെ പരിഹരിച്ചു?"],
        "Communication": ["ഒരു സങ്കീർണ്ണമായ സാങ്കേതിക ആശയം സാങ്കേതികേതര വ്യക്തിക്ക് എങ്ങനെ ലളിതമായി വിശദീകരിക്കും?"],
        "Technical + HR": ["നിങ്ങളെക്കുറിച്ചും നിങ്ങളുടെ പ്രധാന സാങ്കേതിക കഴിവുകളെക്കുറിച്ചും വ്യക്തമാക്കുക."],
        "Full Placement Interview": ["നിങ്ങളുടെ സ്വയം പരിചയപ്പെടുത്തലും ഡാറ്റാബേസ് ഇൻഡക്സിംഗ് എങ്ങനെ പ്രവർത്തിക്കുന്നുവെന്നും വ്യക്തമാക്കുക."]
    }
}

class QReq(BaseModel):
    role: str = "Software Engineer"
    level: str = "Fresher (0 years)"
    kind: str = "Technical"
    domain: str = "DSA"
    difficulty: str = "Adaptive"
    current_difficulty: str = "Intermediate"
    duration: int = 15
    interview_language: str = "English"
    history: list[dict] = []   # [{"q":..., "a":...}]

class ValidateReq(BaseModel):
    question: str
    answer: str
    role: str = "Software Engineer"
    level: str = "Fresher (0 years)"
    kind: str = "Technical"
    domain: str = "DSA"
    interview_language: str = "English"

def validate_answer_offline(q: str, ans: str) -> dict:
    cleaned = ans.strip()
    if not cleaned:
        return {
            "valid": False,
            "status": "empty",
            "technical_accuracy": "insufficient",
            "message": "Answer cannot be empty. Please provide an answer or explanation.",
            "retry_allowed": True
        }
    
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if len(words) < 2:
        common_short_ok = {"yes", "no", "true", "false", "o(1)", "o(n)", "o(n^2)", "o(logn)", "o(nlogn)", "tcp", "udp", "stack", "queue", "bfs", "dfs", "tree", "graph", "hashmap", "array"}
        if cleaned.lower() not in common_short_ok:
            return {
                "valid": False,
                "status": "insufficient",
                "technical_accuracy": "insufficient",
                "message": "Your answer is too short or incomplete. Please elaborate a bit more.",
                "retry_allowed": True
            }

    # Check for keyboard spam / repetitive sequences / lack of natural vowels
    letters = re.findall(r"[A-Za-z]", cleaned)
    vowels = len(re.findall(r"[aeiouAEIOU]", cleaned))
    # Repeated characters like 'aaaa', 'zzzz' or keyboard rows like 'asdfgh', 'zxcvb'
    has_repeated = bool(re.search(r"(.)\1{3,}", cleaned.lower()))
    has_low_vowels = len(letters) >= 6 and (vowels / len(letters)) < 0.22
    has_keyboard_spam = any(seq in cleaned.lower() for seq in ["asdf", "qwerty", "zxcv", "ghjkl"])

    if has_repeated or has_low_vowels or has_keyboard_spam:
        return {
            "valid": False,
            "status": "gibberish",
            "technical_accuracy": "incorrect",
            "message": "Your response appears to be random or meaningless characters. Please provide a clear answer.",
            "retry_allowed": True
        }

    return {
        "valid": True,
        "status": "partially_correct",
        "technical_accuracy": "partially_correct",
        "message": "",
        "retry_allowed": False
    }

@app.post("/api/validate-answer")
def validate_answer(r: ValidateReq):
    ans = r.answer.strip()
    lang = r.interview_language or "English"
    if not ans:
        empty_msgs = {
            "Kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಉತ್ತರ ಅಥವಾ ವಿವರಣೆಯನ್ನು ನೀಡಿ.",
            "Hindi": "कृपया अपना उत्तर या व्याख्या दें।",
            "Tamil": "தயவுசெய்து உங்கள் பதிலை அல்லது விளக்கத்தை அளிக்கவும்.",
            "Telugu": "దయచేసి మీ సమాధానం లేదా వివరణను ఇవ్వండి.",
            "Malayalam": "ദയവായി നിങ്ങളുടെ ഉത്തരമോ വിശദീകരണമോ നൽകുക.",
            "English": "Your answer is empty. Please share your thoughts or explanation."
        }
        return {
            "valid": False,
            "status": "empty",
            "technical_accuracy": "insufficient",
            "message": empty_msgs.get(lang, empty_msgs["English"]),
            "retry_allowed": True
        }

    prompt = f"""You are an objective interview evaluator.
Evaluate the candidate's answer for the given interview question.
Role: {r.role} ({r.level})
Round: {r.kind}
Question Language: {lang}
Question: {r.question}
Candidate answer: {ans}

Requirements:
1. Candidate may respond in {lang}, in English, or in a natural MIX of {lang} and English (code-switching). All of these formats are valid and fully accepted.
2. Detect meaningless text, keyboard spam, or gibberish (e.g., "asdfghjk", "xyz abc lol qwerty").
3. Detect irrelevant or dodging responses (e.g. talking about weather when asked a technical question).
4. Detect empty or extremely insufficient answers.
5. Accept short answers if technically sound and concise (e.g., "O(1)", "Stack LIFO", "Processes have isolated memory").
6. Accept partially correct answers (valid=true, status='partially_correct').
7. CRITICAL: Evaluate TECHNICAL CORRECTNESS completely INDEPENDENT of English or language grammar fluency. If the candidate expresses correct understanding in {lang}, English, or mixed {lang}+English, mark it technically correct.
8. If the answer is invalid (gibberish, irrelevant, or extremely insufficient), set valid=false and provide a short, polite retry message (1-2 sentences) in {lang}. If valid, message should be empty string or concise encouragement.

Return ONLY a valid JSON object matching this schema:
{{
  "valid": true | false,
  "status": "correct" | "partially_correct" | "insufficient" | "irrelevant" | "gibberish",
  "technical_accuracy": "correct" | "partially_correct" | "incorrect" | "insufficient",
  "message": "<short helpful retry message in {lang} if invalid, or empty string>",
  "retry_allowed": true | false
}}"""

    res_raw = llm(prompt, response_mime_type="application/json")
    if res_raw:
        try:
            data = json.loads(res_raw)
            return {
                "valid": bool(data.get("valid", True)),
                "status": str(data.get("status", "partially_correct")),
                "technical_accuracy": str(data.get("technical_accuracy", "partially_correct")),
                "message": str(data.get("message", "")),
                "retry_allowed": bool(data.get("retry_allowed", not data.get("valid", True)))
            }
        except Exception as e:
            print("[validate_answer] JSON parse error:", e)

    return validate_answer_offline(r.question, ans)

@app.post("/api/question")
def question(r: QReq):
    n = len(r.history)
    lang = r.interview_language or "English"
    convo = "\n".join(f"Interviewer: {h['q']}\nCandidate: {h['a']}" for h in r.history)
    domain_clause = f"Technical domain focus: '{r.domain}'." if r.kind in ("Technical", "Technical + HR", "Full Placement Interview") and r.domain else ""

    lang_bank = FALLBACK_Q.get(lang, FALLBACK_Q["English"])
    bank = lang_bank.get(r.kind, lang_bank.get(r.kind.capitalize(), lang_bank.get("Technical", FALLBACK_Q["English"]["Technical"])))
    fallback_q = bank[n % len(bank)]

    # Structured prompt for Gemini to evaluate previous answer (if any) and adaptively produce the next question in the target language
    system_instruction = (
        f"You are an expert, professional technical and HR interviewer conducting an adaptive interview in {lang}. "
        f"You maintain interview context, ask questions natively in {lang} (using standard English technical terms where appropriate), "
        "evaluate answers based STRICTLY on technical correctness (never penalizing grammar, English fluency, or regional code-mixing), "
        "and adjust question difficulty dynamically."
    )

    prompt = f"""Conduct the next turn of an adaptive interview.
Target Role: {r.role}
Candidate Experience: {r.level}
Interview Type: {r.kind}
Interview Language: {lang}
{domain_clause}
Base Difficulty Setting: {r.difficulty}
Current Dynamic Difficulty: {r.current_difficulty or 'Intermediate'}
Target Interview Duration: {r.duration} minutes (Pacing question #{n+1})

History of conversation so far:
{convo or '(No previous questions asked yet. This is the first question.)'}

Instructions:
1. Speak/ask in {lang}. You may keep core technical identifiers in English (e.g. 'Process', 'Thread', 'Hash Map', 'API').
2. If there is a previous answer:
   - Candidate may answer in {lang}, English, or mixed {lang}+English.
   - Evaluate the candidate's last answer based STRICTLY on technical accuracy, understanding, and completeness.
   - Do NOT penalize non-native phrasing or regional terminology.
   - If the candidate's last answer was strong and correct: INCREASE difficulty (e.g., dive deeper into internal mechanisms, edge cases, or optimizations).
   - If the candidate struggled or gave an incorrect answer: REDUCE difficulty to foundational concepts.
   - If the candidate's answer was incomplete or vague: ask a RELEVANT FOLLOW-UP question in {lang} probing that specific gap.
3. If this is the start of the interview (history is empty):
   - Ask an appropriate opening question in {lang} tailored to the role, domain, and experience level ({r.level}).
4. Avoid repeating any questions that were already asked in the conversation history.
5. Keep questions appropriate for the candidate's experience ({r.level}).
6. Ask exactly ONE concise question (1-2 sentences). Do not provide answers or preambles.

Return a JSON object conforming to this schema:
{{
  "question": "<the single next interview question or follow-up in {lang}>",
  "next_difficulty": "Beginner" | "Intermediate" | "Advanced",
  "evaluation_summary": "<brief note on why this question was chosen or difficulty adjusted>",
  "follow_up": true | false
}}"""

    res_raw = llm(prompt, system_instruction=system_instruction, response_mime_type="application/json")
    if res_raw:
        try:
            data = json.loads(res_raw)
            return {
                "question": data.get("question") or fallback_q,
                "next_difficulty": data.get("next_difficulty") or r.current_difficulty or "Intermediate",
                "evaluation_summary": data.get("evaluation_summary") or "",
                "follow_up": bool(data.get("follow_up", False)),
                "ai": True
            }
        except Exception as e:
            print("[question] JSON parse error:", e)

    return {
        "question": fallback_q,
        "next_difficulty": r.current_difficulty or "Intermediate",
        "evaluation_summary": f"Offline fallback question ({lang}).",
        "follow_up": False,
        "ai": False
    }

class AssistReq(BaseModel):
    text: str
    mode: str = "explain"   # explain | hint | vocab | polish
    lang: str = "Kannada"
    context: str = ""

@app.post("/api/assist")
def assist(r: AssistReq):
    tasks = {
        "explain": f"Explain this interview question in simple {r.lang}. Keep key technical terms in English with {r.lang} meaning in brackets.",
        "hint": f"Give ONE small progressive hint in {r.lang} (not the full solution) for this problem.",
        "vocab": f"List 5 key technical terms from this text as 'English - {r.lang} meaning'.",
        "polish": f"Rewrite the candidate's answer in professional English (2-3 sentences), then give one tip in {r.lang}.",
    }
    out = llm(f"{tasks.get(r.mode, tasks['explain'])}\nContext: {r.context}\nText: {r.text}")
    return {"reply": out or f"[Offline demo] Add GEMINI_API_KEY to enable {r.lang} assistance."}

CHALLENGES = {
    "two_sum": {
        "title": "Two Sum",
        "fn": "two_sum",
        "prompt": "Return indices of the two numbers in `nums` that add up to `target`.",
        "starter": {
            "python": "def two_sum(nums, target):\n    # Return indices [i, j]\n    pass\n",
            "javascript": "function twoSum(nums, target) {\n    // Return indices [i, j]\n    \n}\n",
            "java": "class Solution {\n    public int[] twoSum(int[] nums, int target) {\n        // Return indices [i, j]\n        return new int[]{};\n    }\n}\n",
            "c": "/**\n * Note: The returned array must be malloced, assume caller calls free().\n */\nint* twoSum(int* nums, int numsSize, int target, int* returnSize) {\n    *returnSize = 2;\n    int* res = (int*)malloc(2 * sizeof(int));\n    return res;\n}\n",
            "cpp": "#include <vector>\nusing namespace std;\n\nclass Solution {\npublic:\n    vector<int> twoSum(vector<int>& nums, int target) {\n        // Return indices\n        return {};\n    }\n};\n"
        },
        "sample_tests": [
            {"input": "[2, 7, 11, 15], target = 9", "expected": "[0, 1]", "args": [[2, 7, 11, 15], 9], "exp": [0, 1]},
            {"input": "[3, 2, 4], target = 6", "expected": "[1, 2]", "args": [[3, 2, 4], 6], "exp": [1, 2]}
        ],
        "hidden_tests": [
            {"input": "[3, 3], target = 6", "expected": "[0, 1]", "args": [[3, 3], 6], "exp": [0, 1]},
            {"input": "[-1, -2, -3, -4, -5], target = -8", "expected": "[2, 4]", "args": [[-1, -2, -3, -4, -5], -8], "exp": [2, 4]},
            {"input": "[0, 4, 3, 0], target = 0", "expected": "[0, 3]", "args": [[0, 4, 3, 0], 0], "exp": [0, 3]}
        ]
    },
    "palindrome": {
        "title": "Valid Palindrome",
        "fn": "is_palindrome",
        "prompt": "Return True if string `s` is a palindrome, ignoring case and non-alphanumerics.",
        "starter": {
            "python": "def is_palindrome(s):\n    # Return True/False\n    pass\n",
            "javascript": "function isPalindrome(s) {\n    // Return boolean\n    \n}\n",
            "java": "class Solution {\n    public boolean isPalindrome(String s) {\n        return false;\n    }\n}\n",
            "c": "#include <stdbool.h>\n\nbool isPalindrome(char* s) {\n    return false;\n}\n",
            "cpp": "#include <string>\nusing namespace std;\n\nclass Solution {\npublic:\n    bool isPalindrome(string s) {\n        return false;\n    }\n};\n"
        },
        "sample_tests": [
            {"input": "\"A man, a plan, a canal: Panama\"", "expected": "true", "args": ["A man, a plan, a canal: Panama"], "exp": True},
            {"input": "\"race a car\"", "expected": "false", "args": ["race a car"], "exp": False}
        ],
        "hidden_tests": [
            {"input": "\"\"", "expected": "true", "args": [""], "exp": True},
            {"input": "\"0P\"", "expected": "false", "args": ["0P"], "exp": False},
            {"input": "\"Was it a car or a cat I saw?\"", "expected": "true", "args": ["Was it a car or a cat I saw?"], "exp": True}
        ]
    },
}

@app.get("/api/challenge/{cid}")
def challenge(cid: str):
    c = CHALLENGES.get(cid, CHALLENGES["two_sum"])
    return {
        "title": c["title"],
        "prompt": c["prompt"],
        "starter": c["starter"],
        "sample_tests": [{"input": t["input"], "expected": t["expected"]} for t in c["sample_tests"]]
    }

class RunReq(BaseModel):
    challenge_id: str = "two_sum"
    code: str
    language: str = "python"
    mode: str = "run"   # "run" (sample tests only) | "submit" (sample + hidden tests)

def check_disallowed_imports(code: str, lang: str) -> str | None:
    """Security check to block unsafe operations/escapes before subprocess execution."""
    lower = code.lower()
    if lang == "python":
        disallowed = ["import os", "import sys", "import subprocess", "import shutil", "__import__", "eval(", "exec(", "open(", "socket", "http.client", "urllib"]
        for d in disallowed:
            if d in lower:
                return f"Security Restriction: '{d}' is not permitted in the coding sandbox."
    elif lang == "javascript":
        disallowed = ["require('child_process')", 'require("child_process")', "require('fs')", 'require("fs")', "require('net')", "require('http')", "process.exit", "eval("]
        for d in disallowed:
            if d in lower:
                return f"Security Restriction: '{d}' is not permitted in the coding sandbox."
    return None

PYTHON_HARNESS = """
import json, sys

__USER_CODE__

raw_tests = json.loads(sys.stdin.read())
results = []
for t in raw_tests:
    args = t["args"]
    expected = t["exp"]
    is_hidden = t.get("hidden", False)
    inp_desc = t.get("input", "")
    exp_desc = t.get("expected", str(expected))
    try:
        got = __FN_NAME__(*args)
        if isinstance(got, (list, tuple)) and isinstance(expected, list):
            ok = sorted(list(got)) == sorted(expected)
        else:
            ok = (got == expected)
        results.append({
            "ok": bool(ok),
            "hidden": is_hidden,
            "input": inp_desc if not is_hidden else "(Hidden test case)",
            "expected": exp_desc if not is_hidden else "(Hidden)",
            "got": repr(got) if not is_hidden else ("Hidden output" if ok else f"Incorrect result {repr(got)}")
        })
    except Exception as e:
        results.append({
            "ok": False,
            "hidden": is_hidden,
            "input": inp_desc if not is_hidden else "(Hidden test case)",
            "expected": exp_desc if not is_hidden else "(Hidden)",
            "got": type(e).__name__ + ": " + str(e)
        })

print("@@" + json.dumps(results))
"""

JS_HARNESS = """
__USER_CODE__

const raw_tests = JSON.parse(process.argv[2]);
const results = [];
for (const t of raw_tests) {
    const args = t.args;
    const expected = t.exp;
    const is_hidden = Boolean(t.hidden);
    const inp_desc = t.input || "";
    const exp_desc = t.expected || JSON.stringify(expected);
    try {
        const got = __FN_NAME__(...args);
        let ok = false;
        if (Array.isArray(got) && Array.isArray(expected)) {
            const sg = [...got].sort();
            const se = [...expected].sort();
            ok = (sg.length === se.length) && sg.every((v, i) => v === se[i]);
        } else {
            ok = (got === expected);
        }
        results.push({
            ok: Boolean(ok),
            hidden: is_hidden,
            input: is_hidden ? "(Hidden test case)" : inp_desc,
            expected: is_hidden ? "(Hidden)" : exp_desc,
            got: is_hidden ? (ok ? "Hidden output" : `Incorrect result ${JSON.stringify(got)}`) : JSON.stringify(got)
        });
    } catch (e) {
        results.push({
            ok: false,
            hidden: is_hidden,
            input: is_hidden ? "(Hidden test case)" : inp_desc,
            expected: is_hidden ? "(Hidden)" : exp_desc,
            got: e.name + ": " + e.message
        });
    }
}
console.log("@@" + JSON.stringify(results));
"""

JS_FN_MAP = {
    "two_sum": "twoSum",
    "is_palindrome": "isPalindrome"
}

@app.post("/api/run")
def run(r: RunReq):
    c = CHALLENGES.get(r.challenge_id, CHALLENGES["two_sum"])
    lang = (r.language or "python").lower()
    mode = (r.mode or "run").lower()

    # Determine test cases: sample tests for "run", sample + hidden tests for "submit"
    selected_tests = list(c.get("sample_tests", []))
    if mode == "submit":
        hidden_cases = [dict(t, hidden=True) for t in c.get("hidden_tests", [])]
        selected_tests.extend(hidden_cases)

    # 1. Pre-execution Security Sandbox check
    sec_err = check_disallowed_imports(r.code, lang)
    if sec_err:
        return {
            "passed": 0,
            "total": len(selected_tests),
            "cases": [],
            "error_type": "SecurityError",
            "error": sec_err,
            "compilation_error": None,
            "runtime_error": sec_err,
            "execution_time_ms": 0,
            "mode": mode
        }

    # 2. Support check for compiled languages
    if lang not in ("python", "javascript"):
        return {
            "passed": 0,
            "total": len(selected_tests),
            "cases": [],
            "error_type": "EnvironmentInfo",
            "error": f"{r.language.capitalize()} runner is in demo/compilation preview mode. Sandboxed automated test execution is actively provided for Python and JavaScript.",
            "compilation_error": None,
            "runtime_error": None,
            "execution_time_ms": 0,
            "mode": mode
        }

    t0 = time.perf_counter()
    d = tempfile.mkdtemp()
    try:
        if lang == "python":
            src = PYTHON_HARNESS.replace("__USER_CODE__", r.code).replace("__FN_NAME__", c["fn"])
            sol_path = Path(d) / "solution.py"
            sol_path.write_text(src, encoding="utf-8")
            # Sandboxed isolated execution: isolated temporary directory, empty environment
            pr = subprocess.run(
                [sys.executable, "-I", str(sol_path)],
                input=json.dumps(selected_tests),
                capture_output=True,
                text=True,
                timeout=5,
                cwd=d,
                env={}
            )
        elif lang == "javascript":
            fn_name = JS_FN_MAP.get(c["fn"], c["fn"])
            src = JS_HARNESS.replace("__USER_CODE__", r.code).replace("__FN_NAME__", fn_name)
            sol_path = Path(d) / "solution.js"
            sol_path.write_text(src, encoding="utf-8")
            pr = subprocess.run(
                ["node", str(sol_path), json.dumps(selected_tests)],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=d,
                env={}
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Check for syntax/compilation or runtime errors
        if pr.returncode != 0:
            raw_err = (pr.stderr or pr.stdout or "Execution error").strip()
            is_compilation = "SyntaxError" in raw_err or "ReferenceError" in raw_err
            return {
                "passed": 0,
                "total": len(selected_tests),
                "cases": [],
                "error_type": "CompilationError" if is_compilation else "RuntimeError",
                "compilation_error": raw_err if is_compilation else None,
                "runtime_error": raw_err if not is_compilation else None,
                "error": raw_err[-600:],
                "execution_time_ms": elapsed_ms,
                "mode": mode
            }

        m = re.search(r"@@(.*)", pr.stdout)
        if not m:
            err_msg = (pr.stderr or "No test result output generated.")[-600:]
            return {
                "passed": 0,
                "total": len(selected_tests),
                "cases": [],
                "error_type": "RuntimeError",
                "compilation_error": None,
                "runtime_error": err_msg,
                "error": err_msg,
                "execution_time_ms": elapsed_ms,
                "mode": mode
            }

        cases = json.loads(m.group(1))
        passed = sum(1 for x in cases if x.get("ok"))
        return {
            "passed": passed,
            "total": len(cases),
            "cases": cases,
            "error_type": None,
            "compilation_error": None,
            "runtime_error": None,
            "error": None,
            "execution_time_ms": elapsed_ms,
            "mode": mode
        }

    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "passed": 0,
            "total": len(selected_tests),
            "cases": [],
            "error_type": "TimeoutError",
            "compilation_error": None,
            "runtime_error": "Time limit exceeded (5.0s)",
            "error": "Time limit exceeded (5.0s)",
            "execution_time_ms": elapsed_ms,
            "mode": mode
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "passed": 0,
            "total": len(selected_tests),
            "cases": [],
            "error_type": "ExecutionError",
            "compilation_error": None,
            "runtime_error": str(e),
            "error": str(e),
            "execution_time_ms": elapsed_ms,
            "mode": mode
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)

class DiagReq(BaseModel):
    code: str
    language: str = "python"

def loop_depth(node, d=0):
    best = d
    for ch in ast.iter_child_nodes(node):
        best = max(best, loop_depth(ch, d + isinstance(ch, (ast.For, ast.While))))
    return best

@app.post("/api/diagnose")
def diagnose(r: DiagReq):
    """AST diagnostics. Python uses stdlib ast; Tree-sitter plugs in here for Java/C++/C#/JS/TS/Go/Rust."""
    issues = []
    try: tree = ast.parse(r.code)
    except SyntaxError as e:
        return {"issues": [{"level": "error", "msg": f"Syntax error line {e.lineno}: {e.msg}"}], "complexity": None}
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not fns: issues.append({"level": "warn", "msg": "No function defined."})
    for f in fns:
        if not any(isinstance(n, ast.Return) and n.value for n in ast.walk(f)):
            issues.append({"level": "warn", "msg": f"`{f.name}` never returns a value."})
        if any(isinstance(n, ast.Pass) for n in ast.walk(f)) and len(f.body) == 1:
            issues.append({"level": "info", "msg": f"`{f.name}` is still a stub."})
        names = {n.func.id for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if f.name in names: issues.append({"level": "info", "msg": f"`{f.name}` is recursive; check base case."})
    depth = loop_depth(tree)
    est = {0: "O(1)", 1: "O(n)", 2: "O(n²)"}.get(depth, f"O(n^{depth})")
    if depth >= 2: issues.append({"level": "warn", "msg": f"Nested loops (depth {depth}) → likely {est}. Can a hash map help?"})
    return {"issues": issues or [{"level": "ok", "msg": "No structural issues found."}], "complexity": est + " (heuristic estimate)"}

FILLERS = ["um", "uh", "like", "basically", "you know", "actually", "so", "i mean"]

class EvalReq(BaseModel):
    history: list[dict]
    tests_passed: int = 0
    tests_total: int = 0
    complexity: str = ""
    coding_cases: list[dict] = []
    duration_minutes: int = 15
    elapsed_seconds: float = 0
    visual_coaching_enabled: bool = False
    interview_language: str = "English"
    gaze_ratio: float | None = None
    posture_ok: float | None = None

@app.post("/api/evaluate")
def evaluate(r: EvalReq):
    text = " ".join(h.get("a", "") for h in r.history).lower()
    words = re.findall(r"[a-z']+", text)
    fill = sum(len(re.findall(rf"\b{f}\b", text)) for f in FILLERS)
    fill_rate = fill / max(len(words), 1)
    avg_len = len(words) / max(len(r.history), 1)

    # Heuristic Communication base
    comm_base = max(35, min(95, int(90 - fill_rate * 250 - max(0, 15 - avg_len) * 2)))

    # Coding performance score (0-100) based strictly on actual test case outcomes
    coding_score = int(100 * r.tests_passed / r.tests_total) if r.tests_total else None

    # Time management score based on pacing vs expected duration
    # Expected answers: approx 1-2 mins per turn
    target_secs = max(r.duration_minutes * 60, 60)
    actual_secs = max(r.elapsed_seconds, len(r.history) * 60)
    time_ratio = actual_secs / target_secs
    if 0.5 <= time_ratio <= 1.2:
        time_score = 90
    elif time_ratio < 0.5:
        time_score = max(50, int(90 - (0.5 - time_ratio) * 80))
    else:
        time_score = max(50, int(90 - (time_ratio - 1.2) * 50))

    # Optional visual score if enabled
    visual_score = None
    if r.visual_coaching_enabled:
        vis_parts = [100 * x for x in (r.gaze_ratio, r.posture_ok) if x is not None]
        visual_score = int(sum(vis_parts) / len(vis_parts)) if vis_parts else 85

    # Convo formatting for individual question evaluation
    qa_list = [{"question": h.get("q", ""), "answer": h.get("a", "")} for h in r.history]

    lang = r.interview_language or "English"
    system_instruction = (
        f"You are an expert senior engineering interviewer and hiring committee member conducting evaluation in {lang}. "
        "Evaluate the candidate's interview session thoroughly and objectively based ONLY on actual evidence. "
        f"Provide all qualitative feedback, questions analysis, strengths, improvements, and recommendations in {lang}. "
        "CRITICAL REQUIREMENT: Score Technical Knowledge and Problem-Solving strictly based on technical correctness, "
        "clarity of principles, and accuracy of concepts. Candidate answers in English, regional language, or code-mixed "
        "regional+English must be evaluated solely for technical substance. Score English Proficiency SEPARATELY without contaminating "
        "or penalizing technical marks for non-native phrasing or grammatical quirks."
    )

    prompt = f"""Evaluate this mock interview transcript and coding test results.
Interview Language: {lang}
Interview Length: {len(r.history)} questions answered
Target Duration: {r.duration_minutes} minutes (Elapsed time: {round(actual_secs/60, 1)} minutes)
Coding tests passed: {r.tests_passed}/{r.tests_total} (Complexity: {r.complexity or 'None'})

Interview Q&A:
{json.dumps(qa_list, indent=2)}

Provide an evaluation in JSON conforming EXACTLY to this schema (all strings in {lang}):
{{
  "technical_knowledge": <integer 0-100 based on technical depth and accuracy>,
  "problem_solving": <integer 0-100 based on reasoning, trade-offs, and logical decomposition>,
  "communication": <integer 0-100 based on clarity, structure, and conciseness>,
  "english_proficiency": <integer 0-100 based on language fluency, vocabulary, and grammar>,
  "question_evaluations": [
    {{
      "question": "<the question text>",
      "candidate_answer": "<the candidate answer text>",
      "verdict": "Strong" | "Adequate" | "Weak" | "Incorrect",
      "technical_feedback": "<1-2 sentences on technical correctness in {lang}>",
      "communication_feedback": "<1-2 sentences on phrasing or language clarity in {lang}>"
    }}
  ],
  "strengths": ["<strength 1 in {lang}>", "<strength 2 in {lang}>", ...],
  "areas_for_improvement": ["<improvement 1 in {lang}>", "<improvement 2 in {lang}>", ...],
  "incorrect_technical_concepts": ["<specific technical misconception identified in {lang}, or empty array if none>"],
  "recommendations": ["<actionable recommendation 1 in {lang}>", "<actionable recommendation 2 in {lang}>", ...]
}}"""

    llm_eval = None
    res_raw = llm(prompt, system_instruction=system_instruction, response_mime_type="application/json")
    if res_raw:
        try:
            llm_eval = json.loads(res_raw)
        except Exception as e:
            print("[evaluate] JSON parse error:", e)

    # Fallback or synthesis if LLM unavailable
    if not llm_eval:
        llm_eval = {
            "technical_knowledge": min(100, 45 + int(avg_len * 2)),
            "problem_solving": min(100, 50 + (coding_score or 50) // 3),
            "communication": comm_base,
            "english_proficiency": max(50, min(95, int(85 - fill_rate * 200))),
            "question_evaluations": [
                {
                    "question": h.get("q", ""),
                    "candidate_answer": h.get("a", ""),
                    "verdict": "Adequate",
                    "technical_feedback": "Core concept mentioned; elaborating further with concrete architectural trade-offs would improve the response.",
                    "communication_feedback": "Understood and answered directly."
                }
                for h in r.history
            ],
            "strengths": ["Clear participation in interview", "Answered foundational questions directly"],
            "areas_for_improvement": ["Provide deeper examples with technical specifics", "Minimize hesitation words"],
            "incorrect_technical_concepts": [],
            "recommendations": [
                "Practice explaining system concepts using the STAR (Situation, Task, Action, Result) format.",
                "Review edge cases and Big-O computational bounds for core data structures."
            ]
        }

    tech_k = int(llm_eval.get("technical_knowledge", 60))
    prob_s = int(llm_eval.get("problem_solving", 60))
    comm_s = int(llm_eval.get("communication", comm_base))
    eng_s = int(llm_eval.get("english_proficiency", 65))

    # Overall calculation: weighted balance across all core dimensions
    core_scores = [tech_k, prob_s, comm_s, eng_s, time_score]
    if coding_score is not None:
        core_scores.append(coding_score)
    if visual_score is not None:
        core_scores.append(visual_score)

    overall_score = int(sum(core_scores) / len(core_scores))

    return {
        "overall": overall_score,
        "scores": {
            "technical_knowledge": tech_k,
            "problem_solving": prob_s,
            "coding_performance": coding_score,
            "communication_skills": comm_s,
            "english_proficiency": eng_s,
            "time_management": time_score,
            "visual_presentation": visual_score
        },
        "coding_test_cases": {
            "passed": r.tests_passed,
            "total": r.tests_total,
            "cases": r.coding_cases
        },
        "question_evaluations": llm_eval.get("question_evaluations", []),
        "strengths": llm_eval.get("strengths", []),
        "areas_for_improvement": llm_eval.get("areas_for_improvement", []),
        "incorrect_technical_concepts": llm_eval.get("incorrect_technical_concepts", []),
        "recommendations": llm_eval.get("recommendations", []),
        "speech_stats": {
            "filler_words_total": fill,
            "filler_rate_pct": round(fill_rate * 100, 1),
            "words_per_answer": round(avg_len, 1)
        },
        "note": "Technical knowledge and problem-solving are scored strictly independent of English grammar or non-native phrasing."
    }

class TTSReq(BaseModel):
    text: str

@app.post("/api/tts")
async def tts(r: TTSReq):
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key: return Response(status_code=204)   # frontend falls back to browser speech
    vid = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    async with httpx.AsyncClient(timeout=30) as h:
        resp = await h.post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
            headers={"xi-api-key": key}, json={"text": r.text, "model_id": "eleven_turbo_v2_5"})
    if resp.status_code != 200: return Response(status_code=204)
    return Response(resp.content, media_type="audio/mpeg")

@app.get("/")
def index(): return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
