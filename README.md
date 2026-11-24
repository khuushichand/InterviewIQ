# InterviewIQ — AI Interview Prep with On-Demand Kannada Assistance (MVP)

UnlockED hackathon prototype. English-only AI interviewer + optional Kannada help + live coding with tests, AST diagnostics, and a split-score report.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY (and optionally ELEVENLABS_API_KEY)
uvicorn backend.main:app --reload
```
Open http://localhost:8000 in Chrome/Edge (mic needs Web Speech API). Works offline-demo without keys (fallback questions, browser TTS).

## Implemented in MVP
| Feature | Status |
|---|---|
| Gemini adaptive interviewer (technical/HR, role, level) | ✅ |
| ElevenLabs voice (falls back to browser TTS) | ✅ |
| Kannada explain / key terms / hints / English polish (on request only) | ✅ |
| Coding challenges + sandboxed test runner (Python) | ✅ |
| AST diagnostics + complexity heuristic (Python `ast`) | ✅ |
| Filler-word/communication analysis, technical vs language score split | ✅ |
| Report + progress history (browser storage) | ✅ |

## Roadmap (per proposal)
Tree-sitter for Java/C++/C#/JS/TS/Go/Rust, Monaco editor, Gemini Live audio streaming, MediaPipe gaze/posture, pitch/pace analysis, PostgreSQL + JWT, Docker/Judge0 sandbox.

Webcam-derived signals are coaching feedback, not measures of confidence or intent.
