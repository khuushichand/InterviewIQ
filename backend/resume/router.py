"""
FastAPI Router for Resume Analyzer.

DEVELOPER 2 CONTRACT:
=====================
This router exposes POST /api/resume/analyze which returns `ResumeAnalysisResponse`.
Developer 2 can consume this structured response to generate personalized interview questions:

1. Technical questions:
   Use `response.resume_profile.skills` (languages, frameworks, tools)
   and `response.skills_analysis.matched_skills`.

2. Project-specific questions:
   Iterate over `response.resume_profile.projects` (name, description, technologies, key_achievements).

3. HR / Behavioural questions:
   Use `response.resume_profile.experience` (companies, roles, highlights)
   and `response.resume_profile.inferred_experience_level`.

4. Adaptive follow-up probing:
   Use `response.skills_analysis.missing_critical_skills`
   and `response.identified_gaps` to formulate questions assessing how the candidate addresses gaps.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.resume.extractor import (
    EmptyResumeError,
    FileTooLargeError,
    ResumeExtractionError,
    UnsupportedFileFormatError,
    extract_resume_text,
)
from backend.resume.schemas import ResumeAnalysisResponse
from backend.resume.service import analyze_resume

resume_router = APIRouter(
    prefix="/api/resume",
    tags=["Resume Analyzer"]
)


@resume_router.post(
    "/analyze",
    response_model=ResumeAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and analyze a resume document (.pdf / .docx)",
    description=(
        "Extracts text from PDF or DOCX resume, builds a structured profile without hallucination, "
        "and evaluates interview readiness. If target_role or job_description is provided, performs "
        "role-specific skill matching and scoring; otherwise performs general resume-quality analysis."
    )
)
def analyze_resume_endpoint(
    file: UploadFile = File(..., description="Resume file (.pdf or .docx, max 5 MB)"),
    target_role: Optional[str] = Form(None, description="Optional target job title (e.g. 'Backend Engineer')"),
    job_description: Optional[str] = Form(None, description="Optional job description / requirements text")
) -> ResumeAnalysisResponse:
    """
    Accepts a PDF or DOCX resume upload via multipart/form-data.
    Returns a structured `ResumeAnalysisResponse` JSON payload.
    """
    filename = file.filename or "resume.pdf"

    # Read binary stream in memory without persisting to filesystem
    try:
        file_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # Validate file and extract text
    try:
        resume_text = extract_resume_text(file_bytes=file_bytes, filename=filename)
    except FileTooLargeError as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(e)
        )
    except UnsupportedFileFormatError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(e)
        )
    except EmptyResumeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ResumeExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while parsing the resume document."
        )

    # Perform analysis with anti-hallucination guarantees and optional role matching
    try:
        response = analyze_resume(
            resume_text=resume_text,
            target_role=target_role,
            job_description=job_description
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume analysis failed: {str(e)}"
        )
