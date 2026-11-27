"""
Pydantic Schemas for the Resume Analyzer.

HANDOFF CONTRACT FOR DEVELOPER 2:
=================================
The models in this file, specifically `ResumeProfile` and `SkillAnalysis`,
form the contract that the personalized interview question generator will consume:

- resume_profile.skills: For generating deep technical questions on claimed languages/frameworks/tools.
- resume_profile.projects: For generating project-specific architecture, implementation, and trade-off questions.
- resume_profile.experience: For probing past job responsibilities, teamwork, and leadership (HR/behavioural).
- resume_profile.education: For contextualizing academic foundation and fresh graduate questions.
- resume_profile.inferred_experience_level: For calibrating question difficulty (Fresher, Junior, Mid, Senior).
- skills_analysis.matched_skills: High-confidence technical competencies to validate.
- skills_analysis.missing_critical_skills: Target skills missing from resume, for adaptive probing questions.
- identified_gaps: Resume gaps (e.g. missing testing, concurrency, system design) for adaptive follow-ups.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateContact(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class CategorizedSkills(BaseModel):
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    tools_and_cloud: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    company: str
    role: str
    duration: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)
    technologies_used: List[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str
    description: str
    technologies: List[str] = Field(default_factory=list)
    key_achievements: List[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    graduation_year: Optional[str] = None
    grade_or_gpa: Optional[str] = None


class ResumeProfile(BaseModel):
    """
    Structured representation of candidate's profile extracted faithfully from the resume.
    Missing information is represented as None or empty list, never hallucinated.
    """
    contact: CandidateContact = Field(default_factory=CandidateContact)
    summary: Optional[str] = None
    skills: CategorizedSkills = Field(default_factory=CategorizedSkills)
    experience: List[WorkExperience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    inferred_domain: Optional[str] = None
    inferred_experience_level: Optional[str] = None  # e.g., "Fresher", "Junior", "Mid", "Senior"


class SkillAnalysis(BaseModel):
    """
    Skill matching against target role / job description.
    For general (non-targeted) analysis, matched/missing skill lists remain empty.
    """
    matched_skills: List[str] = Field(default_factory=list)
    missing_critical_skills: List[str] = Field(default_factory=list)
    missing_recommended_skills: List[str] = Field(default_factory=list)
    bonus_skills: List[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """
    Transparent scoring breakdown (0-100).
    Represents resume quality and interview readiness; NOT a hiring prediction.
    """
    resume_quality: int = Field(..., ge=0, le=100, description="Overall resume formatting, depth, and clarity")
    skill_match: Optional[int] = Field(None, ge=0, le=100, description="Coverage of target role / JD skills (null if general)")
    experience_relevance: Optional[int] = Field(None, ge=0, le=100, description="Relevance of work history (null if general)")
    project_depth: Optional[int] = Field(None, ge=0, le=100, description="Quality and relevance of projects (null if general)")
    education_fit: Optional[int] = Field(None, ge=0, le=100, description="Alignment of academic credentials (null if general)")
    role_fit: Optional[int] = Field(None, ge=0, le=100, description="Overall fit against target role / JD (null if general)")


class ResumeAnalysisResponse(BaseModel):
    """
    Top-level API response from POST /api/resume/analyze.
    """
    overall_score: int = Field(..., ge=0, le=100, description="Composite interview-readiness score (0-100)")
    target_role: Optional[str] = None
    analysis_type: str = Field("general", description="'targeted' if target_role or JD provided; else 'general'")
    score_breakdown: ScoreBreakdown
    skills_analysis: SkillAnalysis
    key_strengths: List[str] = Field(default_factory=list)
    identified_gaps: List[str] = Field(default_factory=list)
    improvement_recommendations: List[str] = Field(default_factory=list)
    resume_profile: ResumeProfile
