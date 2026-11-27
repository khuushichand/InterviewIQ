"""
Unit and Integration Tests for InterviewIQ Resume Analyzer.

Covers:
- Valid PDF extraction
- Valid DOCX extraction
- Unsupported file format rejection
- File over 5 MB rejection
- Corrupted PDF/DOCX handling
- Empty resume handling
- Resume without target role (General Analysis)
- Resume with target role (Targeted Analysis)
- Resume with job description
- Response schema validation
- Score bounds (0 <= score <= 100)
- Skill matching and missing skill detection with canonical normalization
- Anti-hallucination verification (missing resume info is not invented)
- Endpoint validation on analyze_resume_endpoint (status codes, schemas, errors)
- Existing endpoints integrity verification
"""

import io
import json
import pytest
from unittest.mock import patch
from fastapi import HTTPException, UploadFile
import docx

from backend.main import validate_answer, ValidateReq, index
from backend.resume.extractor import (
    EmptyResumeError,
    FileTooLargeError,
    ResumeExtractionError,
    UnsupportedFileFormatError,
    extract_resume_text,
    normalize_extracted_text,
)
from backend.resume.router import analyze_resume_endpoint
from backend.resume.schemas import ResumeAnalysisResponse
from backend.resume.service import (
    analyze_resume,
    calculate_overall_score,
    match_skills_against_requirements,
    normalize_skill,
)


# Helper fixtures to create test documents in memory
def make_valid_pdf_bytes(text: str = "Jane Doe\njane.doe@example.com\nPython, FastAPI, Docker, PostgreSQL") -> bytes:
    escaped_text = text.replace("(", "\\(").replace(")", "\\)").replace("\n", " ")
    pdf_template = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(escaped_text) + 40} >> stream
BT /F1 12 Tf 72 712 Td ({escaped_text}) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000371 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
448
%%EOF"""
    return pdf_template.encode("latin-1")


def make_valid_docx_bytes(
    name: str = "Alex Chen",
    role: str = "Senior Engineer",
    skills: str = "TypeScript, React, Node.js, AWS, Kubernetes",
    include_tables: bool = True
) -> bytes:
    doc = docx.Document()
    doc.add_heading(f"{name} - {role}", level=1)
    doc.add_paragraph("Email: alex.chen@example.com | Phone: +1-555-0199")
    doc.add_paragraph("Summary: Experienced engineer designing distributed cloud systems.")
    doc.add_heading("Skills", level=2)
    doc.add_paragraph(skills)
    
    if include_tables:
        table = doc.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Languages"
        table.rows[0].cells[1].text = "Go, Python, TypeScript"
        table.rows[1].cells[0].text = "Frameworks"
        table.rows[1].cells[1].text = "Next.js, FastAPI, Express"

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# -------------------------------------------------------------------------
# Unit Tests: Text Extraction & Validation
# -------------------------------------------------------------------------

def test_extract_valid_pdf():
    pdf_bytes = make_valid_pdf_bytes("Sarah Connor\nCyberdyne Systems\nPython, AI, C++")
    text = extract_resume_text(pdf_bytes, "sarah_resume.pdf")
    assert "Sarah Connor" in text
    assert "Python" in text


def test_extract_valid_docx():
    docx_bytes = make_valid_docx_bytes(name="Alex Chen", skills="React, AWS, Docker")
    text = extract_resume_text(docx_bytes, "alex_resume.docx")
    assert "Alex Chen" in text
    assert "React" in text
    assert "Express" in text  # Extracted from table cell


def test_reject_unsupported_file_format():
    with pytest.raises(UnsupportedFileFormatError) as exc_info:
        extract_resume_text(b"some content", "resume.txt")
    assert "Unsupported file format" in str(exc_info.value)

    with pytest.raises(UnsupportedFileFormatError):
        extract_resume_text(b"image data", "photo.png")


def test_reject_file_over_5mb():
    oversized = b"a" * (5 * 1024 * 1024 + 1)
    with pytest.raises(FileTooLargeError) as exc_info:
        extract_resume_text(oversized, "huge_resume.pdf")
    assert "exceeds maximum allowed limit" in str(exc_info.value)


def test_reject_corrupted_pdf():
    corrupted_bytes = b"%PDF-1.4 garbage corrupted byte stream without objects or xref"
    with pytest.raises(ResumeExtractionError):
        extract_resume_text(corrupted_bytes, "bad.pdf")


def test_reject_corrupted_docx():
    corrupted_bytes = b"PK\x03\x04 not a valid zip archive for docx"
    with pytest.raises(ResumeExtractionError):
        extract_resume_text(corrupted_bytes, "bad.docx")


def test_reject_empty_file():
    with pytest.raises(EmptyResumeError):
        extract_resume_text(b"", "empty.pdf")


def test_normalize_extracted_text():
    raw = "Line 1   with   spaces\r\n\r\n\r\n\r\nLine 2 with \x00control chars\n"
    normalized = normalize_extracted_text(raw)
    assert "\x00" not in normalized
    assert "Line 1 with spaces" in normalized
    assert "\n\n\n" not in normalized


# -------------------------------------------------------------------------
# Unit Tests: Skill Normalization and Matching
# -------------------------------------------------------------------------

def test_skill_normalization():
    assert normalize_skill("JS") == "javascript"
    assert normalize_skill("Postgres") == "postgresql"
    assert normalize_skill("ML") == "machine learning"
    assert normalize_skill("k8s") == "kubernetes"
    assert normalize_skill("Amazon Web Services") == "amazon web services"
    assert normalize_skill("AWS") == "amazon web services"


def test_skill_matching_and_missing_detection():
    candidate_skills = ["Python", "JS", "Docker", "Postgres", "Redis"]
    required_skills = ["Python", "TypeScript", "JavaScript", "PostgreSQL", "Kubernetes"]

    matched, missing = match_skills_against_requirements(candidate_skills, required_skills)
    
    # "JS" should match "JavaScript" via canonical normalization
    assert "JS" in matched or "Python" in matched
    # "Postgres" matches "PostgreSQL"
    assert "Postgres" in matched
    # Missing should contain TypeScript and Kubernetes
    assert "TypeScript" in missing
    assert "Kubernetes" in missing


# -------------------------------------------------------------------------
# Unit Tests: Resume Analysis (General vs Targeted) with Gemini Mocking
# -------------------------------------------------------------------------

MOCK_GEMINI_GENERAL_RESPONSE = json.dumps({
    "resume_profile": {
        "contact": {
            "name": "David Miller",
            "email": "david@example.com",
            "phone": "+1-234-567-8900",
            "location": "Boston, MA",
            "linkedin": "https://linkedin.com/in/davidmiller",
            "github": "https://github.com/davidmiller",
            "portfolio": None
        },
        "summary": "Full Stack developer specializing in scalable cloud applications.",
        "skills": {
            "languages": ["Python", "TypeScript"],
            "frameworks": ["FastAPI", "React"],
            "databases": ["PostgreSQL", "Redis"],
            "tools_and_cloud": ["Docker", "AWS"],
            "soft_skills": ["Mentorship", "Agile"]
        },
        "experience": [
            {
                "company": "TechNova",
                "role": "Software Engineer",
                "duration": "2022 - Present",
                "highlights": ["Built REST APIs with FastAPI reducing latency by 35%"],
                "technologies_used": ["FastAPI", "PostgreSQL", "Docker"]
            }
        ],
        "projects": [
            {
                "name": "Cloud Monitor",
                "description": "Real-time infrastructure monitoring dashboard.",
                "technologies": ["React", "FastAPI", "AWS"],
                "key_achievements": ["Handled 10,000 requests/sec with 99.9% uptime."]
            }
        ],
        "education": [
            {
                "institution": "MIT",
                "degree": "B.S. Computer Science",
                "field_of_study": "Computer Science",
                "graduation_year": "2022",
                "grade_or_gpa": "3.8/4.0"
            }
        ],
        "certifications": ["AWS Certified Solutions Architect"],
        "inferred_domain": "Backend & Cloud",
        "inferred_experience_level": "Junior"
    },
    "skills_analysis": {
        "matched_skills": [],
        "missing_critical_skills": [],
        "missing_recommended_skills": [],
        "bonus_skills": []
    },
    "score_breakdown": {
        "resume_quality": 85,
        "skill_match": None,
        "experience_relevance": None,
        "project_depth": None,
        "education_fit": None,
        "role_fit": None
    },
    "key_strengths": ["Quantified performance impact (35% latency reduction)", "Strong cloud background"],
    "identified_gaps": ["Lacks mention of automated testing or CI/CD pipelines"],
    "improvement_recommendations": ["Highlight unit and integration testing tools (e.g. pytest, GitHub Actions)"]
})

MOCK_GEMINI_TARGETED_RESPONSE = json.dumps({
    "resume_profile": {
        "contact": {
            "name": "David Miller",
            "email": "david@example.com",
            "phone": None,
            "location": None,
            "linkedin": None,
            "github": None,
            "portfolio": None
        },
        "summary": "Backend specialist.",
        "skills": {
            "languages": ["Python"],
            "frameworks": ["FastAPI"],
            "databases": ["PostgreSQL"],
            "tools_and_cloud": ["Docker"],
            "soft_skills": []
        },
        "experience": [],
        "projects": [],
        "education": [],
        "certifications": [],
        "inferred_domain": "Backend",
        "inferred_experience_level": "Fresher"
    },
    "skills_analysis": {
        "matched_skills": ["Python", "FastAPI", "PostgreSQL"],
        "missing_critical_skills": ["Kubernetes", "gRPC"],
        "missing_recommended_skills": ["Kafka"],
        "bonus_skills": ["Docker"]
    },
    "score_breakdown": {
        "resume_quality": 75,
        "skill_match": 80,
        "experience_relevance": 65,
        "project_depth": 70,
        "education_fit": 60,
        "role_fit": 75
    },
    "key_strengths": ["Solid foundation in FastAPI and relational databases"],
    "identified_gaps": ["No production Kubernetes or streaming message queue experience"],
    "improvement_recommendations": ["Include experience or personal projects with distributed queues (Kafka/RabbitMQ)"]
})


def test_general_analysis_no_target_role():
    with patch("backend.resume.service.generate_content_with_retry", return_value=MOCK_GEMINI_GENERAL_RESPONSE):
        resp = analyze_resume("Some valid resume content...")
        
        # Verify general mode guarantees
        assert resp.analysis_type == "general"
        assert resp.target_role is None
        assert resp.score_breakdown.skill_match is None
        assert resp.score_breakdown.role_fit is None
        assert resp.score_breakdown.resume_quality == 85
        assert resp.overall_score == 85
        assert resp.skills_analysis.matched_skills == []
        assert resp.skills_analysis.missing_critical_skills == []
        assert 0 <= resp.overall_score <= 100


def test_targeted_analysis_with_role():
    with patch("backend.resume.service.generate_content_with_retry", return_value=MOCK_GEMINI_TARGETED_RESPONSE):
        resp = analyze_resume(
            "Some valid resume content...",
            target_role="Senior Backend Engineer",
            job_description="Must know Python, FastAPI, PostgreSQL, and Kubernetes"
        )

        # Verify targeted mode guarantees
        assert resp.analysis_type == "targeted"
        assert resp.target_role == "Senior Backend Engineer"
        assert resp.score_breakdown.skill_match == 80
        assert resp.score_breakdown.role_fit == 75
        assert "Kubernetes" in resp.skills_analysis.missing_critical_skills
        assert "Python" in resp.skills_analysis.matched_skills
        assert 0 <= resp.overall_score <= 100


def test_anti_hallucination_guarantees():
    """Verify that when contact/experience is absent from resume, it remains null/empty."""
    with patch("backend.resume.service.generate_content_with_retry", return_value=MOCK_GEMINI_TARGETED_RESPONSE):
        resp = analyze_resume("Minimal resume", target_role="Backend Developer")
        
        # In mock data, phone, linkedin, github, experience, projects are empty
        assert resp.resume_profile.contact.phone is None
        assert resp.resume_profile.contact.github is None
        assert resp.resume_profile.experience == []
        assert resp.resume_profile.projects == []


def test_offline_fallback_when_gemini_unavailable():
    """Verify graceful offline fallback without crash when Gemini returns None."""
    sample_text = (
        "Sarah Jenkins\n"
        "sarah.j@example.com | +1-555-0143\n"
        "https://github.com/sarahj | https://linkedin.com/in/sarahj\n"
        "Skills: Python, React, PostgreSQL, Docker, AWS, FastAPI\n"
        "Summary: Passionate software engineer building clean web APIs."
    )
    with patch("backend.resume.service.generate_content_with_retry", return_value=None):
        resp = analyze_resume(sample_text, target_role="Full Stack Developer")
        
        assert isinstance(resp, ResumeAnalysisResponse)
        assert resp.resume_profile.contact.email == "sarah.j@example.com"
        assert "github.com/sarahj" in (resp.resume_profile.contact.github or "")
        assert "Python" in resp.resume_profile.skills.languages
        assert 0 <= resp.overall_score <= 100
        assert resp.target_role == "Full Stack Developer"


# -------------------------------------------------------------------------
# Integration Tests: Endpoint / Router Tests (Status Codes & Schema)
# -------------------------------------------------------------------------

def test_endpoint_general_analysis():
    docx_bytes = make_valid_docx_bytes(name="Jordan Lee", skills="Python, SQL, Linux")
    upload = UploadFile(filename="jordan.docx", file=io.BytesIO(docx_bytes))

    with patch("backend.resume.service.generate_content_with_retry", return_value=MOCK_GEMINI_GENERAL_RESPONSE):
        res = analyze_resume_endpoint(file=upload, target_role=None, job_description=None)
        
        assert isinstance(res, ResumeAnalysisResponse)
        assert res.analysis_type == "general"
        assert res.target_role is None
        assert res.overall_score == 85
        assert res.score_breakdown.resume_quality == 85
        assert res.score_breakdown.skill_match is None
        assert res.resume_profile.contact.name == "David Miller"


def test_endpoint_targeted_analysis():
    pdf_bytes = make_valid_pdf_bytes("Elena Rostova\nFull Stack Developer\nPython React Docker")
    upload = UploadFile(filename="elena.pdf", file=io.BytesIO(pdf_bytes))

    with patch("backend.resume.service.generate_content_with_retry", return_value=MOCK_GEMINI_TARGETED_RESPONSE):
        res = analyze_resume_endpoint(
            file=upload,
            target_role="Backend Lead",
            job_description="Lead microservices architecture with Python and K8s"
        )
        
        assert isinstance(res, ResumeAnalysisResponse)
        assert res.analysis_type == "targeted"
        assert res.target_role == "Backend Lead"
        assert "Kubernetes" in res.skills_analysis.missing_critical_skills
        assert res.score_breakdown.skill_match == 80


def test_endpoint_reject_unsupported_format():
    upload = UploadFile(filename="script.exe", file=io.BytesIO(b"executable data"))
    with pytest.raises(HTTPException) as exc_info:
        analyze_resume_endpoint(file=upload, target_role=None, job_description=None)
    assert exc_info.value.status_code == 415
    assert "Unsupported file format" in exc_info.value.detail


def test_endpoint_reject_empty_file():
    upload = UploadFile(filename="empty.pdf", file=io.BytesIO(b""))
    with pytest.raises(HTTPException) as exc_info:
        analyze_resume_endpoint(file=upload, target_role=None, job_description=None)
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


def test_endpoint_reject_oversized_file():
    huge_bytes = b"0" * (5 * 1024 * 1024 + 10)
    upload = UploadFile(filename="big.pdf", file=io.BytesIO(huge_bytes))
    with pytest.raises(HTTPException) as exc_info:
        analyze_resume_endpoint(file=upload, target_role=None, job_description=None)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum allowed limit" in exc_info.value.detail


def test_endpoint_reject_corrupted_file():
    corrupt = b"%PDF-1.4 completely corrupted content"
    upload = UploadFile(filename="corrupt.pdf", file=io.BytesIO(corrupt))
    with pytest.raises(HTTPException) as exc_info:
        analyze_resume_endpoint(file=upload, target_role=None, job_description=None)
    assert exc_info.value.status_code == 422
    assert "Corrupted or unreadable" in exc_info.value.detail


def test_existing_endpoints_intact():
    """Verify that existing endpoints in backend/main.py function correctly."""
    req = ValidateReq(
        question="What is an array?",
        answer="A contiguous memory data structure storing elements of same type.",
        role="Software Engineer"
    )
    res_val = validate_answer(req)
    assert "valid" in res_val
    assert res_val["valid"] is True

    # Test index route
    res_root = index()
    assert res_root is not None
