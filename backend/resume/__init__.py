"""
Resume Analyzer Module for InterviewIQ.

Provides:
- Resume extraction (.pdf, .docx)
- Structured resume profiling
- Target role / job description matching and gap detection
- Transparent 0-100 interview-readiness scoring
- Integration handoff contract for Developer 2 (personalized interview question generation)
"""
from backend.resume.schemas import (
    ResumeProfile,
    ResumeAnalysisResponse,
    SkillAnalysis,
    ScoreBreakdown,
    CandidateContact,
    CategorizedSkills,
    WorkExperience,
    Project,
    Education,
)
from backend.resume.router import resume_router

__all__ = [
    "resume_router",
    "ResumeProfile",
    "ResumeAnalysisResponse",
    "SkillAnalysis",
    "ScoreBreakdown",
    "CandidateContact",
    "CategorizedSkills",
    "WorkExperience",
    "Project",
    "Education",
]
