"""
Tests for resume-grounded interview question generation in InterviewIQ.

Covers:
- Backward compatibility: QReq without resume_profile produces original behavior.
- Question generation when ResumeProfile with skills, projects, and experience is supplied.
- Question generation prioritizing resume projects and claimed skills.
- Gap-probing question formulation when identified_gaps are provided.
- Safe handling of incomplete or missing resume profile fields (e.g. empty projects, no experience).
- Offline fallback deterministically producing resume-grounded questions when LLM is unavailable.
- Integration via POST /api/question endpoint.
"""

import json
from unittest.mock import patch
import pytest

from backend.main import QReq, question, is_similar_question, get_non_repeating_fallback
from backend.resume.schemas import (
    CandidateContact,
    CategorizedSkills,
    Education,
    Project,
    ResumeProfile,
    WorkExperience,
)


def sample_resume_profile():
    return ResumeProfile(
        contact=CandidateContact(name="Alex Rivera", email="alex@example.com"),
        summary="Backend engineer specializing in distributed systems and FastAPI microservices.",
        skills=CategorizedSkills(
            languages=["Python", "Go", "TypeScript"],
            frameworks=["FastAPI", "React", "gRPC"],
            databases=["PostgreSQL", "Redis"],
            tools_and_cloud=["Docker", "Kubernetes", "AWS"]
        ),
        experience=[
            WorkExperience(
                company="Acme Corp",
                role="Software Engineer",
                duration="2023 - 2025",
                highlights=["Designed event-driven payment processing pipeline"],
                technologies_used=["Python", "FastAPI", "Redis"]
            )
        ],
        projects=[
            Project(
                name="StreamEngine",
                description="High-throughput distributed event streaming bus.",
                technologies=["Go", "Kafka", "Docker"],
                key_achievements=["Processed 50,000 events/sec with sub-5ms latency."]
            ),
            Project(
                name="TaskPulse",
                description="Real-time task synchronization dashboard.",
                technologies=["FastAPI", "React", "PostgreSQL"],
                key_achievements=["Built REST and WebSocket APIs serving 1,000 active users."]
            )
        ],
        education=[
            Education(
                institution="Tech University",
                degree="B.S. Computer Science",
                graduation_year="2023"
            )
        ],
        inferred_domain="Backend Engineering",
        inferred_experience_level="Junior"
    )


def test_backward_compatibility_no_resume():
    """Verify that requests without resume_profile preserve identical behavior."""
    req = QReq(
        role="Software Engineer",
        level="Fresher (0 years)",
        kind="Technical",
        domain="DSA",
        history=[]
    )
    assert req.resume_profile is None
    assert req.identified_gaps == []

    mock_llm_response = json.dumps({
        "question": "Explain how hash collisions are handled.",
        "next_difficulty": "Intermediate",
        "evaluation_summary": "Standard technical opening question.",
        "follow_up": False
    })

    with patch("backend.main.llm", return_value=mock_llm_response) as mock_llm:
        res = question(req)
        assert res["question"] == "Explain how hash collisions are handled."
        assert res["ai"] is True
        # Verify prompt did NOT contain resume section
        call_prompt = mock_llm.call_args[0][0]
        assert "CANDIDATE RESUME PROFILE" not in call_prompt


def test_question_with_resume_context_prompts_gemini():
    """Verify that when resume_profile is present, the prompt instructs Gemini to ground questions in resume claims."""
    profile = sample_resume_profile()
    req = QReq(
        role="Backend Engineer",
        level="Junior",
        kind="Technical",
        domain="Backend",
        history=[],
        resume_profile=profile,
        identified_gaps=["Kubernetes production deployment", "CI/CD automated testing"]
    )

    mock_llm_response = json.dumps({
        "question": "In StreamEngine, how did you achieve sub-5ms latency with Kafka in Go?",
        "next_difficulty": "Intermediate",
        "evaluation_summary": "Grounded in candidate's StreamEngine project.",
        "follow_up": False
    })

    with patch("backend.main.llm", return_value=mock_llm_response) as mock_llm:
        res = question(req)
        assert res["question"] == "In StreamEngine, how did you achieve sub-5ms latency with Kafka in Go?"
        assert res["ai"] is True

        call_prompt = mock_llm.call_args[0][0]
        # Verify prompt includes resume ground truth
        assert "CANDIDATE RESUME PROFILE (GROUND TRUTH):" in call_prompt
        assert "StreamEngine" in call_prompt
        assert "TaskPulse" in call_prompt
        assert "FastAPI" in call_prompt
        assert "Kubernetes production deployment" in call_prompt
        assert "RESUME-GROUNDED INTERVIEWING REQUIREMENTS:" in call_prompt


def test_resume_offline_fallback_project_grounding():
    """Verify that when Gemini is unavailable, fallback question specifically targets the candidate's actual projects."""
    profile = sample_resume_profile()
    req = QReq(
        role="Backend Engineer",
        kind="Technical",
        history=[],  # Turn 0 -> Should target first project
        resume_profile=profile
    )

    with patch("backend.main.llm", return_value=None):
        res = question(req)
        assert res["ai"] is False
        assert "StreamEngine" in res["question"]
        assert "architecture" in res["question"].lower()


def test_resume_offline_fallback_skill_and_gap_progression():
    """Verify that subsequent turns in offline mode cycle through skills, gaps, and past experience."""
    profile = sample_resume_profile()

    # Turn 1: should target skills
    req_turn1 = QReq(
        role="Backend Engineer",
        history=[{"q": "Q1", "a": "A1"}],
        resume_profile=profile
    )
    with patch("backend.main.llm", return_value=None):
        res1 = question(req_turn1)
        assert res1["ai"] is False
        assert "resume mentions hands-on experience" in res1["question"]

    # Turn 2: should probe gaps if present
    req_turn2 = QReq(
        role="Backend Engineer",
        history=[{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}],
        resume_profile=profile,
        identified_gaps=["Distributed Tracing (OpenTelemetry)"]
    )
    with patch("backend.main.llm", return_value=None):
        res2 = question(req_turn2)
        assert res2["ai"] is False
        assert "Distributed Tracing (OpenTelemetry)" in res2["question"]

    # Turn 3: should probe company experience
    req_turn3 = QReq(
        role="Backend Engineer",
        history=[{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}, {"q": "Q3", "a": "A3"}],
        resume_profile=profile
    )
    with patch("backend.main.llm", return_value=None):
        res3 = question(req_turn3)
        assert res3["ai"] is False
        assert "Acme Corp" in res3["question"]


def test_safe_handling_incomplete_resume():
    """Verify robust handling when resume profile has empty or None fields."""
    empty_profile = ResumeProfile(
        contact=CandidateContact(name="Minimal User"),
        skills=CategorizedSkills(),
        projects=[],
        experience=[],
        education=[]
    )
    req = QReq(
        role="Frontend Engineer",
        resume_profile=empty_profile
    )

    mock_llm_response = json.dumps({
        "question": "Can you share what technologies you have recently worked with?",
        "next_difficulty": "Beginner",
        "evaluation_summary": "Minimal profile fallback probe.",
        "follow_up": False
    })

    with patch("backend.main.llm", return_value=mock_llm_response):
        res = question(req)
        assert res["ai"] is True
        assert res["question"] is not None

    # Offline fallback test with empty profile
    with patch("backend.main.llm", return_value=None):
        res_offline = question(req)
        assert res_offline["ai"] is False
        assert res_offline["question"] is not None


def test_llm_duplicate_question_triggers_progressive_fallback():
    """Verify that if the LLM produces a duplicate/near-duplicate question, it is intercepted and replaced."""
    profile = sample_resume_profile()
    initial_q = "In your resume, you highlighted the project 'StreamEngine'. Can you walk me through its core architecture and the key technical trade-offs you made?"

    req = QReq(
        role="Backend Engineer",
        history=[
            {"q": initial_q, "a": "We used Go and Kafka to stream events with low latency."}
        ],
        resume_profile=profile
    )

    # LLM attempts to return the same question or a slight paraphrase
    duplicate_q = "Can you walk me through the core architecture and key technical trade-offs you made in StreamEngine?"
    mock_llm_response = json.dumps({
        "question": duplicate_q,
        "next_difficulty": "Intermediate",
        "evaluation_summary": "Attempting to repeat.",
        "follow_up": False
    })

    with patch("backend.main.llm", return_value=mock_llm_response):
        res = question(req)
        # Should detect similarity and not return the duplicate
        assert res["question"] != duplicate_q
        assert res["question"] != initial_q
        assert res["ai"] is False  # Fallback was used to guarantee non-repetition


def test_normal_interview_fallback_avoids_repetition():
    """Verify that normal interviews without resume do not repeat questions from the bank."""
    first_q = "Explain the difference between a process and a thread."
    req = QReq(
        role="Software Engineer",
        kind="Technical",
        history=[
            {"q": first_q, "a": "Processes have separate memory spaces, threads share memory."}
        ]
    )

    with patch("backend.main.llm", return_value=None):
        res = question(req)
        assert res["question"] != first_q
        assert "process and a thread" not in res["question"].lower()


def test_resume_offline_progression_across_multiple_turns():
    """Verify that offline resume fallback progressively covers distinct topics across 4+ turns without duplicates."""
    profile = sample_resume_profile()
    history = []
    asked = []

    for turn in range(4):
        req = QReq(
            role="Backend Engineer",
            history=history,
            resume_profile=profile,
            identified_gaps=["Kubernetes clustering"]
        )
        with patch("backend.main.llm", return_value=None):
            res = question(req)
            q_text = res["question"]
            assert q_text not in asked, f"Turn {turn} generated duplicate question: {q_text}"
            asked.append(q_text)
            history.append({"q": q_text, "a": f"Candidate answer for turn {turn}."})

    assert len(asked) == 4
    # All 4 questions must be distinct
    assert len(set(asked)) == 4


def test_third_turn_repetition_regression_resolved():
    """
    Regression test for the 3rd-turn repetition bug:
    Turn 0 asks about project architecture.
    Turn 1 asks about tech stack.
    Turn 2 (the 3rd question):
      Simulates an LLM attempting to re-ask about the same project's architecture,
      verifying that similarity detection intercepts it and produces a progressive fresh question.
    """
    profile = sample_resume_profile()
    q0 = "In your resume, you highlighted the project 'StreamEngine'. Can you walk me through its core architecture and the key technical trade-offs you made?"
    q1 = "How have you applied Go in production, and how did you handle edge cases or performance bottlenecks?"

    history = [
        {"q": q0, "a": "StreamEngine uses Go channels and Kafka brokers to ingest events."},
        {"q": q1, "a": "We tuned garbage collection thresholds and pooled worker goroutines."}
    ]

    req = QReq(
        role="Backend Engineer",
        history=history,
        resume_profile=profile
    )

    # Simulate LLM at turn 2 returning a paraphrased repeat of Turn 0
    paraphrased_repeat = "Can you describe the overall architecture of your StreamEngine project and how data flows through it?"
    progressive_retry = "In your TaskPulse project, how did you manage real-time WebSocket state synchronization?"

    calls = [
        json.dumps({"question": paraphrased_repeat, "next_difficulty": "Intermediate", "follow_up": False}),
        json.dumps({"question": progressive_retry, "next_difficulty": "Intermediate", "follow_up": False})
    ]

    with patch("backend.main.llm", side_effect=calls):
        res = question(req)
        assert res["question"] != q0
        assert res["question"] != q1
        assert res["question"] != paraphrased_repeat
        assert res["question"] == progressive_retry
        assert res["ai"] is True


def test_normal_interview_3rd_turn_progression():
    """Verify that normal interviews progress cleanly through turn 2 (the 3rd question) and beyond without repetition."""
    history = [
        {"q": "Tell me about yourself.", "a": "I am a backend developer."},
        {"q": "Describe a challenge you faced in a team project and how you handled it.", "a": "We had a conflicting API spec which we aligned on."}
    ]

    req = QReq(
        role="Software Engineer",
        kind="HR",
        history=history
    )

    with patch("backend.main.llm", return_value=None):
        res = question(req)
        assert res["question"] not in [h["q"] for h in history]
        assert "Tell me about yourself" not in res["question"]
        assert "challenge you faced" not in res["question"]

def test_session_wide_uniqueness_across_10_turns_normal_interview():
    """Verify that an entire 10-turn normal interview session has 10 unique, non-repeating questions."""
    history = []
    asked = []
    for turn in range(10):
        req = QReq(
            role="Software Engineer",
            kind="Technical",
            history=history,
            asked_questions=list(asked),
            interview_language="English"
        )
        with patch("backend.main.llm", return_value=None):
            res = question(req)
            q = res["question"]
            assert q not in asked, f"Turn {turn+1} repeated a question: {q}"
            for prev_q in asked:
                assert not is_similar_question(q, [prev_q]), f"Turn {turn+1} was similar to {prev_q}: {q}"
            asked.append(q)
            history.append({"q": q, "a": f"Answer to question {turn+1}"})
    assert len(set(asked)) == 10


def test_session_wide_uniqueness_across_10_turns_resume_interview():
    """Verify that an entire 10-turn resume-based interview session has 10 unique, non-repeating questions."""
    profile = sample_resume_profile()
    history = []
    asked = []
    for turn in range(10):
        req = QReq(
            role="Backend Engineer",
            kind="Technical",
            history=history,
            asked_questions=list(asked),
            resume_profile=profile,
            identified_gaps=["Distributed Tracing (OpenTelemetry)", "Kubernetes Operator Patterns"]
        )
        with patch("backend.main.llm", return_value=None):
            res = question(req)
            q = res["question"]
            assert q not in asked, f"Resume Turn {turn+1} repeated a question: {q}"
            for prev_q in asked:
                assert not is_similar_question(q, [prev_q]), f"Resume Turn {turn+1} was similar to {prev_q}: {q}"
            asked.append(q)
            history.append({"q": q, "a": f"Answer to resume question {turn+1}"})
    assert len(set(asked)) == 10


def test_session_wide_uniqueness_indic_languages():
    """Verify that multi-turn interviews in Indic languages (Kannada, Hindi) never repeat across 8 turns."""
    for lang in ["Kannada", "Hindi"]:
        history = []
        asked = []
        for turn in range(8):
            req = QReq(
                role="Software Engineer",
                kind="Technical",
                history=history,
                asked_questions=list(asked),
                interview_language=lang
            )
            with patch("backend.main.llm", return_value=None):
                res = question(req)
                q = res["question"]
                assert q not in asked, f"{lang} Turn {turn+1} repeated a question: {q}"
                asked.append(q)
                history.append({"q": q, "a": f"{lang} answer {turn+1}"})
        assert len(set(asked)) == 8, f"{lang} had duplicates in {asked}"


def test_llm_paraphrased_duplicate_rejection_and_fallback():
    """Verify that when the LLM keeps generating paraphrased variations of already asked questions,
    both initial response and retry are rejected, and a completely unasked question is safely returned."""
    history = [
        {"q": "Explain the difference between a process and a thread.", "a": "A process has isolated address space."}
    ]
    asked = ["Explain the difference between a process and a thread."]
    req = QReq(
        role="Software Engineer",
        kind="Technical",
        history=history,
        asked_questions=asked,
        interview_language="English"
    )

    # Both LLM calls return paraphrased duplicates
    call1 = json.dumps({"question": "Explain how a process differs from a thread in OS.", "next_difficulty": "Intermediate"})
    call2 = json.dumps({"question": "What is the primary difference between processes and threads?", "next_difficulty": "Intermediate"})

    with patch("backend.main.llm", side_effect=[call1, call2]):
        res = question(req)
        # Must NOT return either paraphrased duplicate
        assert not is_similar_question(res["question"], asked)
        assert res["question"] != asked[0]
        assert "process" not in res["question"].lower() or "thread" not in res["question"].lower()


def test_question_bank_exhaustion_returns_controlled_message():
    """Verify that if all available questions in a bank are exhausted, it never repeats any question
    and returns a controlled graceful completion message."""
    bank = ["Question Alpha", "Question Beta"]
    asked = ["Question Alpha", "Question Beta"]
    q = get_non_repeating_fallback(bank, 2, asked)
    assert q not in asked
    assert "conclude" in q.lower() or "finish" in q.lower()

def test_answer_validation_kannada_and_indic_languages():
    """Verify that pure Kannada, mixed Kannada-English, Hindi, and short valid Indic answers are accepted
    by answer validation and do not block answer submission."""
    from backend.main import validate_answer_offline, ValidateReq, validate_answer

    # 1. Pure Kannada full explanation
    kannada_ans = "ಪ್ರೊಸೆಸ್ ಸ್ವತಂತ್ರ ಮೆಮೊರಿಯನ್ನು ಹೊಂದಿದೆ ಮತ್ತು ಥ್ರೆಡ್ ಅದೇ ಮೆಮೊರಿಯನ್ನು ಹಂಚಿಕೊಳ್ಳುತ್ತದೆ."
    v1 = validate_answer_offline("ಪ್ರೊಸೆಸ್ ಮತ್ತು ಥ್ರೆಡ್ ವ್ಯತ್ಯಾಸವೇನು?", kannada_ans)
    assert v1["valid"] is True, f"Failed pure Kannada validation: {v1}"

    # 2. Mixed Kannada-English code-switching
    mixed_ans = "Process ಗೆ separate memory space ಇರುತ್ತೆ, ಆದರೆ threads same memory share ಮಾಡ್ತವೆ."
    v2 = validate_answer_offline("ಪ್ರೊಸೆಸ್ ಮತ್ತು ಥ್ರೆಡ್ ವ್ಯತ್ಯಾಸವೇನು?", mixed_ans)
    assert v2["valid"] is True, f"Failed mixed Kannada-English validation: {v2}"

    # 3. Short valid Kannada answer
    short_kannada = "ಹೌದು"
    v3 = validate_answer_offline("ಪ್ರಶ್ನೆ", short_kannada)
    assert v3["valid"] is True, f"Failed short Kannada validation: {v3}"

    # 4. Pure Hindi full explanation
    hindi_ans = "प्रोसेस के पास अपनी मेमोरी होती है जबकि थ्रेड एक ही मेमोरी शेयर करते हैं।"
    v4 = validate_answer_offline("Process और Thread में क्या अंतर है?", hindi_ans)
    assert v4["valid"] is True, f"Failed Hindi validation: {v4}"

    # 5. Endpoint call with Gemini offline (fallback)
    req = ValidateReq(
        question="ಪ್ರೊಸೆಸ್ ಮತ್ತು ಥ್ರೆಡ್ ನಡುವಿನ ವ್ಯತ್ಯಾಸವನ್ನು ವಿವರಿಸಿ.",
        answer="ಪ್ರೊಸೆಸ್ ಪ್ರತ್ಯೇಕ ಮೆಮೊರಿ ಸ್ಪೇಸ್ ಹೊಂದಿರುತ್ತದೆ ಆದರೆ ಥ್ರೆಡ್‌ಗಳು ಮೆಮೊರಿ ಹಂಚಿಕೊಳ್ಳುತ್ತವೆ.",
        role="Software Engineer",
        interview_language="Kannada"
    )
    with patch("backend.main.llm", return_value=None):
        endpoint_res = validate_answer(req)
        assert endpoint_res["valid"] is True, f"Endpoint rejected valid Kannada answer: {endpoint_res}"

    # 6. Verify empty answer is rejected with localized message
    empty_req = ValidateReq(
        question="Q",
        answer="   ",
        role="Software Engineer",
        interview_language="Kannada"
    )
    empty_res = validate_answer(empty_req)
    assert empty_res["valid"] is False
    assert "ದಯವಿಟ್ಟು" in empty_res["message"]

