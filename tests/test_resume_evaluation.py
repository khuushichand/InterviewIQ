"""
Targeted tests for resume-aware interview evaluation in InterviewIQ.

Covers:
- Backward compatibility: EvalReq without resume context retains identical schema and behavior.
- Resume-aware evaluation prompts Gemini with resume claims and instructions to verify claims via Q&A evidence.
- Verified skills and unsubstantiated claims are structured in resume_alignment.
- Offline fallback behavior when Gemini is unavailable, safely producing resume_alignment without crashing.
- Safe handling of incomplete or empty resume profiles.
"""

import json
from unittest.mock import patch
import pytest

from backend.main import EvalReq, evaluate
from backend.resume.schemas import (
    CandidateContact,
    CategorizedSkills,
    Education,
    Project,
    ResumeAnalysisResponse,
    ResumeProfile,
    ScoreBreakdown,
    SkillAnalysis,
    WorkExperience,
)


def sample_resume_profile():
    return ResumeProfile(
        contact=CandidateContact(name="Morgan Riley", email="morgan@example.com"),
        summary="Senior Backend Engineer building distributed high-concurrency systems.",
        skills=CategorizedSkills(
            languages=["Go", "Python"],
            frameworks=["FastAPI", "Gin"],
            databases=["PostgreSQL", "Redis"],
            tools_and_cloud=["Docker", "Kubernetes", "AWS"]
        ),
        experience=[
            WorkExperience(
                company="CloudScale",
                role="Backend Engineer",
                duration="2022 - 2025",
                highlights=["Scaled message queue consumers to 20k msg/sec"],
                technologies_used=["Go", "Kafka", "PostgreSQL"]
            )
        ],
        projects=[
            Project(
                name="PulseQueue",
                description="Distributed task queue using Redis and Go.",
                technologies=["Go", "Redis", "Docker"],
                key_achievements=["Zero data loss across node failovers."]
            )
        ],
        inferred_experience_level="Mid"
    )


def test_evaluation_backward_compatibility_no_resume():
    """Verify that evaluation requests without resume context retain original behavior and schema."""
    req = EvalReq(
        history=[
            {"q": "What is the difference between a process and a thread?", "a": "Processes have separate memory spaces, while threads share the memory of their parent process."}
        ],
        tests_passed=2,
        tests_total=2,
        duration_minutes=15,
        elapsed_seconds=900
    )

    mock_llm_eval = json.dumps({
        "technical_knowledge": 85,
        "problem_solving": 80,
        "communication": 82,
        "english_proficiency": 88,
        "question_evaluations": [
            {
                "question": "What is the difference between a process and a thread?",
                "candidate_answer": "Processes have separate memory spaces...",
                "verdict": "Strong",
                "technical_feedback": "Accurate distinction regarding memory isolation.",
                "communication_feedback": "Concise and clear."
            }
        ],
        "strengths": ["Accurate systems architecture fundamentals"],
        "areas_for_improvement": ["Discuss thread synchronization primitives"],
        "incorrect_technical_concepts": [],
        "recommendations": ["Review mutexes and semaphores"]
    })

    with patch("backend.main.llm", return_value=mock_llm_eval) as mock_llm:
        res = evaluate(req)
        assert res["overall"] > 0
        assert "resume_alignment" not in res
        assert res["scores"]["technical_knowledge"] == 85
        assert res["scores"]["coding_performance"] == 100
        # Ensure resume claims were not injected into prompt
        prompt_text = mock_llm.call_args[0][0]
        assert "CANDIDATE RESUME CLAIMS" not in prompt_text


def test_resume_aware_evaluation_prompts_and_scores_alignment():
    """Verify that when resume_profile is supplied, evaluation verifies claims and returns resume_alignment."""
    profile = sample_resume_profile()
    req = EvalReq(
        history=[
            {
                "q": "How did you design failover in your PulseQueue project?",
                "a": "In PulseQueue, we used Redis Sentinel with Go worker heartbeats and distributed lock leases to prevent split-brain during master failover."
            }
        ],
        resume_profile=profile
    )

    mock_llm_eval = json.dumps({
        "technical_knowledge": 90,
        "problem_solving": 88,
        "communication": 85,
        "english_proficiency": 86,
        "resume_alignment": {
            "claim_verification_score": 92,
            "verified_skills": ["Go", "Redis", "Distributed Systems"],
            "unsubstantiated_or_weak_claims": [],
            "resume_feedback": "Candidate demonstrated genuine, hands-on knowledge of Redis Sentinel and Go distributed locking matching their PulseQueue project."
        },
        "question_evaluations": [
            {
                "question": "How did you design failover in your PulseQueue project?",
                "candidate_answer": "In PulseQueue, we used Redis Sentinel...",
                "verdict": "Strong",
                "technical_feedback": "Solid practical reasoning on split-brain prevention.",
                "communication_feedback": "Well-structured response."
            }
        ],
        "strengths": ["Clear explanation of distributed state recovery"],
        "areas_for_improvement": ["Consider discussing Raft consensus alternatives"],
        "incorrect_technical_concepts": [],
        "recommendations": ["Explore etcd for strong consistency requirements"]
    })

    with patch("backend.main.llm", return_value=mock_llm_eval) as mock_llm:
        res = evaluate(req)
        prompt_text = mock_llm.call_args[0][0]
        assert "CANDIDATE RESUME CLAIMS (REFERENCE):" in prompt_text
        assert "PulseQueue" in prompt_text
        assert "CloudScale" in prompt_text
        assert "RESUME-AWARE EVALUATION RULES:" in prompt_text

        assert "resume_alignment" in res
        ra = res["resume_alignment"]
        assert ra["claim_verification_score"] == 92
        assert "Go" in ra["verified_skills"]
        assert "PulseQueue" in ra["resume_feedback"]


def test_resume_aware_evaluation_offline_fallback():
    """Verify safe fallback when Gemini is offline and resume context is present."""
    profile = sample_resume_profile()
    req = EvalReq(
        history=[
            {
                "q": "Tell me about Go in your project.",
                "a": "I wrote concurrent Go microservices using goroutines and channels."
            }
        ],
        resume_profile=profile
    )

    with patch("backend.main.llm", return_value=None):
        res = evaluate(req)
        assert res["overall"] > 0
        assert "resume_alignment" in res
        ra = res["resume_alignment"]
        assert 0 <= ra["claim_verification_score"] <= 100
        assert "Go" in ra["verified_skills"]


def test_resume_aware_evaluation_incomplete_profile():
    """Verify that an empty or incomplete resume profile does not fail evaluation."""
    empty_profile = ResumeProfile(
        contact=CandidateContact(),
        skills=CategorizedSkills(),
        projects=[],
        experience=[],
        education=[]
    )
    req = EvalReq(
        history=[{"q": "Explain BFS.", "a": "Breadth-first search traverses level by level using a queue."}],
        resume_profile=empty_profile
    )

    with patch("backend.main.llm", return_value=None):
        res = evaluate(req)
        assert res["overall"] > 0
        assert res["scores"]["technical_knowledge"] >= 0
