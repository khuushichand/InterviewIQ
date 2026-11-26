"""
Resume Analysis Service for InterviewIQ.

Handles:
- LLM-powered structured resume profiling via Gemini (reusing backend/gemini_service.py)
- Optional target role / job description matching vs general resume quality analysis
- Conservative, anti-hallucination extraction
- Skill normalization and missing skill detection
- Transparent, weighted 0-100 scoring (interview readiness, not hiring prediction)
- Robust offline fallback when Gemini is unavailable
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple

from backend.gemini_service import generate_content_with_retry, MODEL_NAME
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

# Configurable transparent scoring weights for targeted role / JD analysis
TARGETED_SCORING_WEIGHTS = {
    "skill_match": 0.30,         # Coverage of required/core competencies
    "role_fit": 0.20,            # Overall alignment with the role's scope
    "experience_relevance": 0.20,# Relevance of work background
    "project_depth": 0.15,       # Relevance and technical depth of projects
    "resume_quality": 0.10,      # Structural clarity, action metrics, formatting
    "education_fit": 0.05,       # Alignment of academic credentials
}

# Skill synonyms for non-aggressive, canonical equivalence matching
SKILL_SYNONYMS: Dict[str, str] = {
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "ml": "machine learning",
    "machine learning": "machine learning",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "dl": "deep learning",
    "deep learning": "deep learning",
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",
    "react": "react",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "node.js": "node.js",
    "nodejs": "node.js",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "aws": "amazon web services",
    "amazon web services": "amazon web services",
    "gcp": "google cloud platform",
    "google cloud": "google cloud platform",
    "google cloud platform": "google cloud platform",
    "azure": "microsoft azure",
    "microsoft azure": "microsoft azure",
    "golang": "go",
    "go": "go",
    "py": "python",
    "python": "python",
    "c#": "c-sharp",
    "c-sharp": "c-sharp",
    "csharp": "c-sharp",
    "c++": "cpp",
    "cpp": "cpp",
    "ror": "ruby on rails",
    "rails": "ruby on rails",
    "ruby on rails": "ruby on rails",
    "vue": "vue.js",
    "vue.js": "vue.js",
    "vuejs": "vue.js",
    "docker": "docker",
    "containerization": "docker",
    "rest": "rest api",
    "restful": "rest api",
    "rest api": "rest api",
    "graphql": "graphql",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "cicd": "cicd",
}


def normalize_skill(skill: str) -> str:
    """Normalize skill name to lowercase alphanumeric canonical representation."""
    cleaned = skill.lower().strip()
    cleaned = re.sub(r"[\s\-_]+", " ", cleaned)
    return SKILL_SYNONYMS.get(cleaned, cleaned)


def match_skills_against_requirements(
    candidate_skills: List[str],
    target_requirements: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Compares candidate skills against required skills using safe canonical normalization.
    Returns: (matched_skills, missing_skills)
    """
    candidate_norm_map = {normalize_skill(s): s for s in candidate_skills}
    matched = []
    missing = []

    for req in target_requirements:
        req_norm = normalize_skill(req)
        if req_norm in candidate_norm_map:
            matched.append(candidate_norm_map[req_norm])
        else:
            missing.append(req)

    return matched, missing


def build_gemini_extraction_prompt(
    resume_text: str,
    target_role: Optional[str] = None,
    job_description: Optional[str] = None
) -> Tuple[str, str]:
    """
    Constructs system instruction and prompt for Gemini JSON extraction.
    """
    system_instruction = (
        "You are an expert technical recruiter, engineering leader, and objective resume analyst. "
        "Your task is to analyze candidate resume text with 100% factual accuracy and zero hallucination. "
        "\nCRITICAL RULES:"
        "\n1. Rely STRICTLY on facts explicitly stated in the resume text. Do NOT assume, extrapolate, or invent details."
        "\n2. If any field (e.g. contact details, dates, companies, project details) is not mentioned in the resume, return null or an empty list []. NEVER invent placeholder text or example companies."
        "\n3. If NO target role or job description is provided, do NOT assume 'Software Engineer' or any default. Perform a general resume-quality assessment only."
        "\n4. Scoring represents interview-readiness and quality of technical evidence (0-100), NOT a hiring prediction or job offer guarantee."
        "\n5. Return strictly valid JSON conforming to the requested schema."
    )

    is_targeted = bool(target_role or job_description)

    prompt = f"""Analyze the following resume text and output a structured JSON analysis.

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"
"""
    if target_role:
        prompt += f"\nTARGET ROLE: {target_role.strip()}\n"
    if job_description:
        prompt += f"\nJOB DESCRIPTION / REQUIREMENTS:\n{job_description.strip()}\n"

    if is_targeted:
        prompt += f"""
TARGETED ANALYSIS INSTRUCTIONS:
- Evaluate the candidate strictly against the Target Role and/or Job Description provided.
- Identify skills that match, critical skills that are missing, and recommended skills that are missing.
- Bonus skills are skills the candidate possesses that add valuable domain depth beyond the basic requirements.
- Calculate transparent integer scores (0-100) for:
  * skill_match: direct overlap with role/JD skills
  * experience_relevance: relevance of past roles to target requirements
  * project_depth: technical complexity and relevance of projects
  * education_fit: relevance of academic background to the role
  * role_fit: overall alignment with the target position
  * resume_quality: structural clarity, formatting, quantifiable metrics, and impact statements

Return ONLY JSON matching this EXACT structure:
{{
  "resume_profile": {{
    "contact": {{
      "name": "<name or null>",
      "email": "<email or null>",
      "phone": "<phone or null>",
      "location": "<location or null>",
      "linkedin": "<url or null>",
      "github": "<url or null>",
      "portfolio": "<url or null>"
    }},
    "summary": "<summary/objective if present or null>",
    "skills": {{
      "languages": ["<language>", ...],
      "frameworks": ["<framework>", ...],
      "databases": ["<database>", ...],
      "tools_and_cloud": ["<tool/cloud>", ...],
      "soft_skills": ["<soft skill>", ...]
    }},
    "experience": [
      {{
        "company": "<company name>",
        "role": "<job title>",
        "duration": "<dates/duration or null>",
        "highlights": ["<highlight/bullet 1>", ...],
        "technologies_used": ["<tech 1>", ...]
      }}
    ],
    "projects": [
      {{
        "name": "<project title>",
        "description": "<project overview>",
        "technologies": ["<tech 1>", ...],
        "key_achievements": ["<bullet/achievement 1>", ...]
      }}
    ],
    "education": [
      {{
        "institution": "<school/university>",
        "degree": "<degree>",
        "field_of_study": "<major/branch or null>",
        "graduation_year": "<year or null>",
        "grade_or_gpa": "<gpa/percentage or null>"
      }}
    ],
    "certifications": ["<cert 1>", ...],
    "inferred_domain": "<primary domain e.g. Frontend, Backend, Machine Learning, Embedded, or null>",
    "inferred_experience_level": "<Fresher | Junior | Mid | Senior or null>"
  }},
  "skills_analysis": {{
    "matched_skills": ["<skill>", ...],
    "missing_critical_skills": ["<critical skill missing for target role>", ...],
    "missing_recommended_skills": ["<recommended skill missing>", ...],
    "bonus_skills": ["<extra skill from resume that adds value>", ...]
  }},
  "score_breakdown": {{
    "resume_quality": <integer 0-100>,
    "skill_match": <integer 0-100>,
    "experience_relevance": <integer 0-100>,
    "project_depth": <integer 0-100>,
    "education_fit": <integer 0-100>,
    "role_fit": <integer 0-100>
  }},
  "key_strengths": ["<concise factual strength 1>", ...],
  "identified_gaps": ["<factual gap against target role 1>", ...],
  "improvement_recommendations": ["<actionable recommendation 1>", ...]
}}
"""
    else:
        prompt += """
GENERAL ANALYSIS INSTRUCTIONS:
- NO target role or job description was provided.
- Do NOT assume 'Software Engineer' or calculate role-specific match scores.
- Perform a thorough general resume quality analysis: structure, clarity, quantification of achievements, technical skills organization, and interview readiness.
- In score_breakdown, provide 'resume_quality' (0-100). Keep skill_match, experience_relevance, project_depth, education_fit, and role_fit as null.
- In skills_analysis, keep matched_skills, missing_critical_skills, missing_recommended_skills, and bonus_skills as empty lists [].

Return ONLY JSON matching this EXACT structure:
{{
  "resume_profile": {{
    "contact": {{
      "name": "<name or null>",
      "email": "<email or null>",
      "phone": "<phone or null>",
      "location": "<location or null>",
      "linkedin": "<url or null>",
      "github": "<url or null>",
      "portfolio": "<url or null>"
    }},
    "summary": "<summary/objective if present or null>",
    "skills": {{
      "languages": ["<language>", ...],
      "frameworks": ["<framework>", ...],
      "databases": ["<database>", ...],
      "tools_and_cloud": ["<tool/cloud>", ...],
      "soft_skills": ["<soft skill>", ...]
    }},
    "experience": [
      {{
        "company": "<company name>",
        "role": "<job title>",
        "duration": "<dates/duration or null>",
        "highlights": ["<highlight/bullet 1>", ...],
        "technologies_used": ["<tech 1>", ...]
      }}
    ],
    "projects": [
      {{
        "name": "<project title>",
        "description": "<project overview>",
        "technologies": ["<tech 1>", ...],
        "key_achievements": ["<bullet/achievement 1>", ...]
      }}
    ],
    "education": [
      {{
        "institution": "<school/university>",
        "degree": "<degree>",
        "field_of_study": "<major/branch or null>",
        "graduation_year": "<year or null>",
        "grade_or_gpa": "<gpa/percentage or null>"
      }}
    ],
    "certifications": ["<cert 1>", ...],
    "inferred_domain": "<primary domain e.g. Frontend, Backend, Machine Learning, Embedded, or null>",
    "inferred_experience_level": "<Fresher | Junior | Mid | Senior or null>"
  }},
  "skills_analysis": {{
    "matched_skills": [],
    "missing_critical_skills": [],
    "missing_recommended_skills": [],
    "bonus_skills": []
  }},
  "score_breakdown": {{
    "resume_quality": <integer 0-100>,
    "skill_match": null,
    "experience_relevance": null,
    "project_depth": null,
    "education_fit": null,
    "role_fit": null
  }},
  "key_strengths": ["<general structural/content strength 1>", ...],
  "identified_gaps": ["<resume presentation or technical detail gap 1>", ...],
  "improvement_recommendations": ["<actionable recommendation for resume improvement 1>", ...]
}}
"""
    return system_instruction, prompt


def calculate_overall_score(breakdown: ScoreBreakdown, is_targeted: bool) -> int:
    """
    Computes overall score from breakdown based on transparent weights.
    Clamps result strictly to [0, 100].
    """
    if not is_targeted:
        return max(0, min(100, int(breakdown.resume_quality)))

    # For targeted analysis, compute weighted score
    sm = breakdown.skill_match if breakdown.skill_match is not None else breakdown.resume_quality
    rf = breakdown.role_fit if breakdown.role_fit is not None else breakdown.resume_quality
    er = breakdown.experience_relevance if breakdown.experience_relevance is not None else breakdown.resume_quality
    pd = breakdown.project_depth if breakdown.project_depth is not None else breakdown.resume_quality
    rq = breakdown.resume_quality
    ef = breakdown.education_fit if breakdown.education_fit is not None else breakdown.resume_quality

    weighted = (
        TARGETED_SCORING_WEIGHTS["skill_match"] * sm
        + TARGETED_SCORING_WEIGHTS["role_fit"] * rf
        + TARGETED_SCORING_WEIGHTS["experience_relevance"] * er
        + TARGETED_SCORING_WEIGHTS["project_depth"] * pd
        + TARGETED_SCORING_WEIGHTS["resume_quality"] * rq
        + TARGETED_SCORING_WEIGHTS["education_fit"] * ef
    )
    return max(0, min(100, int(round(weighted))))


def extract_offline_fallback(
    resume_text: str,
    target_role: Optional[str] = None,
    job_description: Optional[str] = None
) -> ResumeAnalysisResponse:
    """
    Conservative, zero-hallucination heuristic fallback when Gemini is unavailable.
    Rely strictly on deterministic regex extraction without fabricating any details.
    """
    is_targeted = bool(target_role or job_description)

    # Email
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", resume_text)
    email = email_match.group(0) if email_match else None

    # Phone
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", resume_text)
    phone = phone_match.group(0) if phone_match else None

    # Links
    linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?", resume_text, re.IGNORECASE)
    linkedin = linkedin_match.group(0) if linkedin_match else None

    github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?", resume_text, re.IGNORECASE)
    github = github_match.group(0) if github_match else None

    # Candidate Name (heuristic: first clean non-empty line with letters)
    name = None
    lines = [ln.strip() for ln in resume_text.split("\n") if ln.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) <= 4 and re.match(r"^[A-Za-z\s.'-]+$", first_line):
            name = first_line

    # Known technology vocabulary search
    known_tech = {
        "python": "languages", "javascript": "languages", "typescript": "languages",
        "java": "languages", "c++": "languages", "c#": "languages", "go": "languages",
        "ruby": "languages", "rust": "languages", "sql": "languages", "html": "languages",
        "css": "languages", "react": "frameworks", "angular": "frameworks", "vue": "frameworks",
        "django": "frameworks", "fastapi": "frameworks", "flask": "frameworks",
        "spring boot": "frameworks", "express": "frameworks", "next.js": "frameworks",
        "postgresql": "databases", "mysql": "databases", "mongodb": "databases",
        "redis": "databases", "sqlite": "databases", "oracle": "databases",
        "docker": "tools_and_cloud", "kubernetes": "tools_and_cloud", "aws": "tools_and_cloud",
        "azure": "tools_and_cloud", "gcp": "tools_and_cloud", "git": "tools_and_cloud",
        "linux": "tools_and_cloud", "jira": "tools_and_cloud", "jenkins": "tools_and_cloud"
    }

    found_skills: Dict[str, List[str]] = {
        "languages": [], "frameworks": [], "databases": [], "tools_and_cloud": [], "soft_skills": []
    }
    resume_lower = resume_text.lower()
    for tech, category in known_tech.items():
        if re.search(rf"\b{re.escape(tech)}\b", resume_lower):
            found_skills[category].append(tech.title())

    # Build profile
    profile = ResumeProfile(
        contact=CandidateContact(
            name=name,
            email=email,
            phone=phone,
            linkedin=linkedin,
            github=github,
            location=None,
            portfolio=None
        ),
        summary=None,
        skills=CategorizedSkills(**found_skills),
        experience=[],
        projects=[],
        education=[],
        certifications=[],
        inferred_domain=None,
        inferred_experience_level="Fresher" if "fresher" in resume_lower or "intern" in resume_lower else None
    )

    # Base quality score heuristics
    quality_score = 40
    if email: quality_score += 10
    if phone: quality_score += 10
    if github or linkedin: quality_score += 10
    total_skills = sum(len(v) for v in found_skills.values())
    if total_skills >= 5: quality_score += 15
    if len(resume_text.split()) >= 150: quality_score += 15
    quality_score = min(85, quality_score)  # Heuristic capped at 85

    strengths = []
    if email or phone:
        strengths.append("Direct contact details are present.")
    if total_skills > 0:
        strengths.append(f"Identified {total_skills} verified technical tools and languages.")

    gaps = []
    if not (github or linkedin):
        gaps.append("Professional profile links (LinkedIn/GitHub) were not detected.")

    recommendations = [
        "Include measurable impact metrics (percentages, speedups, user count) in project and experience descriptions.",
        "Ensure clear section headings (Skills, Experience, Projects, Education) for machine readability."
    ]

    if is_targeted:
        role_label = (target_role or "Specified Role").strip()
        all_candidate_skills = [s for sublist in found_skills.values() for s in sublist]

        # Extract basic keywords from role / JD
        target_corpus = f"{target_role or ''} {job_description or ''}".lower()
        target_skills = [t.title() for t in known_tech.keys() if re.search(rf"\b{re.escape(t)}\b", target_corpus)]
        
        matched, missing = match_skills_against_requirements(all_candidate_skills, target_skills)
        bonus = [s for s in all_candidate_skills if s not in matched]

        match_pct = int(100 * len(matched) / max(len(target_skills), 1)) if target_skills else 60

        breakdown = ScoreBreakdown(
            resume_quality=quality_score,
            skill_match=match_pct,
            experience_relevance=55,
            project_depth=50,
            education_fit=60,
            role_fit=int((match_pct + quality_score) / 2)
        )
        overall = calculate_overall_score(breakdown, is_targeted=True)

        if missing:
            gaps.append(f"Key skills mentioned in target requirements not detected in resume: {', '.join(missing[:4])}")

        return ResumeAnalysisResponse(
            overall_score=overall,
            target_role=target_role,
            analysis_type="targeted",
            score_breakdown=breakdown,
            skills_analysis=SkillAnalysis(
                matched_skills=matched,
                missing_critical_skills=missing[:3],
                missing_recommended_skills=missing[3:6],
                bonus_skills=bonus[:5]
            ),
            key_strengths=strengths or ["Structured resume format."],
            identified_gaps=gaps,
            improvement_recommendations=recommendations,
            resume_profile=profile
        )
    else:
        breakdown = ScoreBreakdown(
            resume_quality=quality_score,
            skill_match=None,
            experience_relevance=None,
            project_depth=None,
            education_fit=None,
            role_fit=None
        )
        return ResumeAnalysisResponse(
            overall_score=quality_score,
            target_role=None,
            analysis_type="general",
            score_breakdown=breakdown,
            skills_analysis=SkillAnalysis(),
            key_strengths=strengths or ["Clear textual presentation."],
            identified_gaps=gaps,
            improvement_recommendations=recommendations,
            resume_profile=profile
        )


def analyze_resume(
    resume_text: str,
    target_role: Optional[str] = None,
    job_description: Optional[str] = None
) -> ResumeAnalysisResponse:
    """
    Analyzes resume text using Gemini, with a robust heuristic fallback.
    Target role is strictly optional. If absent, performs general resume analysis.
    """
    clean_role = target_role.strip() if target_role and target_role.strip() else None
    clean_jd = job_description.strip() if job_description and job_description.strip() else None
    is_targeted = bool(clean_role or clean_jd)

    system_instruction, prompt = build_gemini_extraction_prompt(
        resume_text=resume_text,
        target_role=clean_role,
        job_description=clean_jd
    )

    raw_response = generate_content_with_retry(
        contents=prompt,
        system_instruction=system_instruction,
        response_mime_type="application/json"
    )

    if raw_response:
        try:
            data = json.loads(raw_response)
            
            # Parse profile safely
            prof_data = data.get("resume_profile", {})
            contact_data = prof_data.get("contact", {})
            skills_data = prof_data.get("skills", {})

            contact = CandidateContact(
                name=contact_data.get("name"),
                email=contact_data.get("email"),
                phone=contact_data.get("phone"),
                location=contact_data.get("location"),
                linkedin=contact_data.get("linkedin"),
                github=contact_data.get("github"),
                portfolio=contact_data.get("portfolio")
            )

            skills = CategorizedSkills(
                languages=skills_data.get("languages", []) or [],
                frameworks=skills_data.get("frameworks", []) or [],
                databases=skills_data.get("databases", []) or [],
                tools_and_cloud=skills_data.get("tools_and_cloud", []) or [],
                soft_skills=skills_data.get("soft_skills", []) or []
            )

            experience = [
                WorkExperience(
                    company=str(exp.get("company", "Company")),
                    role=str(exp.get("role", "Role")),
                    duration=exp.get("duration"),
                    highlights=exp.get("highlights", []) or [],
                    technologies_used=exp.get("technologies_used", []) or []
                )
                for exp in prof_data.get("experience", []) or []
                if isinstance(exp, dict)
            ]

            projects = [
                Project(
                    name=str(proj.get("name", "Project")),
                    description=str(proj.get("description", "")),
                    technologies=proj.get("technologies", []) or [],
                    key_achievements=proj.get("key_achievements", []) or []
                )
                for proj in prof_data.get("projects", []) or []
                if isinstance(proj, dict)
            ]

            education = [
                Education(
                    institution=str(edu.get("institution", "Institution")),
                    degree=str(edu.get("degree", "Degree")),
                    field_of_study=edu.get("field_of_study"),
                    graduation_year=edu.get("graduation_year"),
                    grade_or_gpa=edu.get("grade_or_gpa")
                )
                for edu in prof_data.get("education", []) or []
                if isinstance(edu, dict)
            ]

            profile = ResumeProfile(
                contact=contact,
                summary=prof_data.get("summary"),
                skills=skills,
                experience=experience,
                projects=projects,
                education=education,
                certifications=prof_data.get("certifications", []) or [],
                inferred_domain=prof_data.get("inferred_domain"),
                inferred_experience_level=prof_data.get("inferred_experience_level")
            )

            # Parse score breakdown
            raw_scores = data.get("score_breakdown", {})
            rq = max(0, min(100, int(raw_scores.get("resume_quality", 70))))

            if is_targeted:
                def safe_score(val):
                    return max(0, min(100, int(val))) if val is not None else None

                breakdown = ScoreBreakdown(
                    resume_quality=rq,
                    skill_match=safe_score(raw_scores.get("skill_match")),
                    experience_relevance=safe_score(raw_scores.get("experience_relevance")),
                    project_depth=safe_score(raw_scores.get("project_depth")),
                    education_fit=safe_score(raw_scores.get("education_fit")),
                    role_fit=safe_score(raw_scores.get("role_fit"))
                )
            else:
                breakdown = ScoreBreakdown(
                    resume_quality=rq,
                    skill_match=None,
                    experience_relevance=None,
                    project_depth=None,
                    education_fit=None,
                    role_fit=None
                )

            overall = calculate_overall_score(breakdown, is_targeted)

            # Parse skills analysis
            raw_sa = data.get("skills_analysis", {})
            if is_targeted:
                sa = SkillAnalysis(
                    matched_skills=raw_sa.get("matched_skills", []) or [],
                    missing_critical_skills=raw_sa.get("missing_critical_skills", []) or [],
                    missing_recommended_skills=raw_sa.get("missing_recommended_skills", []) or [],
                    bonus_skills=raw_sa.get("bonus_skills", []) or []
                )
            else:
                sa = SkillAnalysis()

            return ResumeAnalysisResponse(
                overall_score=overall,
                target_role=clean_role,
                analysis_type="targeted" if is_targeted else "general",
                score_breakdown=breakdown,
                skills_analysis=sa,
                key_strengths=data.get("key_strengths", []) or [],
                identified_gaps=data.get("identified_gaps", []) or [],
                improvement_recommendations=data.get("improvement_recommendations", []) or [],
                resume_profile=profile
            )

        except Exception as e:
            print(f"[ResumeService] LLM JSON parsing warning: {e}. Falling back to heuristic extraction.")

    # Graceful fallback when Gemini is unavailable or JSON fails to parse
    return extract_offline_fallback(
        resume_text=resume_text,
        target_role=clean_role,
        job_description=clean_jd
    )
