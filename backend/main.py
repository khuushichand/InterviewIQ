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
from backend.resume import resume_router
from backend.resume.schemas import ResumeProfile, ResumeAnalysisResponse

app = FastAPI(title="InterviewIQ")
app.include_router(resume_router)
MODEL = MODEL_NAME

def llm(prompt: str, system_instruction: str | None = None, response_mime_type: str | None = None) -> str | None:
    return generate_content_with_retry(
        contents=prompt,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        model=MODEL
    )

FALLBACK_Q = {'English': {'Technical': ['Explain the difference between a process and a thread.',
                           'What is the time complexity of searching in a hash map, and why?',
                           'How would you detect a cycle in a linked list?',
                           'What is the difference between an array and a linked list?',
                           'Explain the ACID properties in database transaction management.',
                           'How does asynchronous I/O differ from synchronous multi-threading?',
                           'What strategies would you use to design a cache eviction policy?',
                           'Describe the CAP theorem and the trade-offs it imposes on distributed systems.',
                           'How does garbage collection work in modern managed runtimes, and what triggers pause '
                           'times?',
                           'Explain how database indexing using B-trees or LSM trees optimizes query retrieval.',
                           'What is the difference between optimistic and pessimistic locking in concurrent systems?',
                           'How would you design a rate limiter for a high-throughput microservices architecture?'],
             'HR': ['Tell me about yourself.',
                    'Describe a challenge you faced in a team project and how you handled it.',
                    'Why do you want to join our company?',
                    'Where do you see yourself in three years?',
                    'How do you prioritize competing deadlines when multiple tasks arrive simultaneously?',
                    'Can you describe an environment where you do your best work?',
                    'What is an engineering decision you made that you would do differently in hindsight?',
                    'How do you stay motivated and focused when working on repetitive or ambiguous requirements?',
                    'What criteria do you use to evaluate whether a company culture is a good fit for you?',
                    'Tell me about a time you gave or received constructive feedback that led to positive change.'],
             'Behavioural': ['Tell me about a time you had a disagreement with a team member and how you resolved it.',
                             'Describe a situation where you had to meet a tight deadline under pressure.',
                             'Can you share an example of a mistake you made and what you learned from it?',
                             'Tell me about an instance where you stepped outside your role to solve an issue.',
                             'Describe a scenario where requirements changed midway and how you adapted.',
                             'How do you navigate working with a colleague whose communication style differs from '
                             'yours?',
                             'Tell me about a situation where you advocated for code quality or technical debt '
                             'reduction.',
                             'Describe a project where you took ownership of a critical failure during a production '
                             'release.',
                             'Can you describe how you managed work-life balance during a demanding sprint cycle?',
                             'Tell me about a time you mentored or helped an unblocked peer to achieve a team goal.'],
             'Communication': ['How would you explain a complex technical concept to a non-technical stakeholder?',
                               'Can you walk me through your recent project architecture in under two minutes?',
                               'How do you handle delivering critical feedback to a colleague?',
                               'Describe a situation where miscommunication caused a bug and how you fixed it.',
                               'How do you write technical documentation or pull request descriptions so peers review '
                               'quickly?',
                               'Explain how you conduct a post-mortem meeting without blaming individuals.',
                               'How do you communicate an unexpected delay or engineering setback to leadership?',
                               'Describe how you present trade-offs between speed-to-market and architectural purity '
                               'to product managers.'],
             'Technical + HR': ['Tell me about yourself and your primary technical stack.',
                                'Describe how you design scalable APIs and how you manage collaboration in your team.',
                                'What was the most challenging bug you resolved, and how did you communicate the '
                                'impact?',
                                'How do you keep your technical skills current while balancing team deliverables?',
                                'Walk me through a system you designed and how you handled code reviews with peers.',
                                'How do you evaluate new libraries or technologies before recommending them to your '
                                'team?',
                                'Describe a scenario where you had to debug a live production issue while keeping '
                                'stakeholders informed.',
                                'What are your core architectural principles when building maintainable software?',
                                'How do you handle disagreement during code reviews or architecture RFC discussions?',
                                'Tell me about a technical accomplishment that gave you the greatest sense of '
                                'engineering pride.'],
             'Full Placement Interview': ['Introduce yourself and give a brief overview of your technical background.',
                                          'Walk me through how indexing works in a relational database or your primary '
                                          'domain.',
                                          'Tell me about a time you led an initiative or took ownership of an '
                                          'engineering issue.',
                                          'How do you balance writing quick prototypes versus production-grade '
                                          'resilient code?',
                                          'Can you explain how you design monitoring and observability for a live '
                                          'microservice?',
                                          'What strategies do you use when refactoring legacy code without introducing '
                                          'regressions?',
                                          'How would you break down a complex monolith into independent domain '
                                          'services?',
                                          'Describe how you write automated end-to-end and integration tests to ensure '
                                          'system reliability.',
                                          'What security practices do you follow to prevent common vulnerabilities '
                                          'like injection or auth bypass?',
                                          'Where do you see software engineering evolving over the next five years, '
                                          'and how are you preparing?']},
 'Kannada': {'Technical': ['ಪ್ರೊಸೆಸ್ ಮತ್ತು ಥ್ರೆಡ್ ನಡುವಿನ ವ್ಯತ್ಯಾಸವನ್ನು ವಿವರಿಸಿ.',
                           'ಹ್ಯಾಶ್ ಮ್ಯಾಪ್\u200cನಲ್ಲಿ ಹುಡುಕುವ ಸಮಯ ಸಂಕೀರ್ಣತೆ (Time Complexity) ಏನು ಮತ್ತು ಏಕೆ?',
                           'ಲಿಂಕ್ಡ್ ಲಿಸ್ಟ್\u200cನಲ್ಲಿ ಸೈಕಲ್ ಅನ್ನು ಹೇಗೆ ಕಂಡುಹಿಡಿಯುತ್ತೀರಿ?',
                           'ಅರೇ ಮತ್ತು ಲಿಂಕ್ಡ್ ಲಿಸ್ಟ್ ನಡುವಿನ ಪ್ರಮುಖ ವ್ಯತ್ಯಾಸಗಳು ಯಾವುವು?',
                           'ಡೇಟಾಬೇಸ್\u200cನಲ್ಲಿ ACID ಗುಣಲಕ್ಷಣಗಳು ಏಕೆ ಮುಖ್ಯವಾಗಿವೆ?',
                           'ಸಿಂಕ್ರೋನಸ್ ಮತ್ತು ಅಸಿಂಕ್ರೋನಸ್ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ನಡುವಿನ ವ್ಯತ್ಯಾಸವೇನು?',
                           'ಕ್ಯಾಶಿಂಗ್ (Caching) ತಂತ್ರಜ್ಞಾನವನ್ನು ಸಿಸ್ಟಮ್\u200cನಲ್ಲಿ ಹೇಗೆ ಅಳವಡಿಸುತ್ತೀರಿ?',
                           'ಡಿಸ್ಟ್ರಿಬ್ಯೂಟೆಡ್ ಸಿಸ್ಟಮ್\u200cಗಳಲ್ಲಿ CAP ಪ್ರಮೇಯ (CAP Theorem) ಏನು ತಿಳಿಸುತ್ತದೆ?',
                           'ಡೇಟಾಬೇಸ್ ಇಂಡೆಕ್ಸಿಂಗ್ ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ ಮತ್ತು ಕ್ವೆರಿ ವೇಗವನ್ನು ಹೇಗೆ ಹೆಚ್ಚಿಸುತ್ತದೆ?',
                           'ರೆಸ್ಟ್ ಎಪಿಐ (REST API) ವಿನ್ಯಾಸಗೊಳಿಸುವಾಗ ನೀವು ಅನುಸರಿಸುವ ಅತ್ಯುತ್ತಮ ಅಭ್ಯಾಸಗಳಾವುವು?'],
             'HR': ['ನಿಮ್ಮ ಬಗ್ಗೆ ಪರಿಚಯಿಸಿಕೊಳ್ಳಿ.',
                    'ತಂಡದ ಪ್ರಾಜೆಕ್ಟ್\u200cನಲ್ಲಿ ನೀವು ಎದುರಿಸಿದ ಸವಾಲು ಮತ್ತು ಅದನ್ನು ಹೇಗೆ ಪರಿಹರಿಸಿದ್ದೀರಿ ಎಂದು ತಿಳಿಸಿ.',
                    'ನಮ್ಮ ಕಂಪನಿಗೆ ಸೇರಲು ನೀವು ಏಕೆ ಇಷ್ಟಪಡುತ್ತೀರಿ?',
                    'ಮುಂದಿನ ಮೂರು ವರ್ಷಗಳಲ್ಲಿ ನಿಮ್ಮನ್ನು ನೀವು ಎಲ್ಲಿ ನೋಡಲು ಬಯಸುತ್ತೀರಿ?',
                    'ಒಂದೇ ಸಮಯದಲ್ಲಿ ಅನೇಕ ಜವಾಬ್ದಾರಿಗಳು ಬಂದಾಗ ನೀವು ಹೇಗೆ ಆದ್ಯತೆ ನೀಡುತ್ತೀರಿ?',
                    'ನಿಮ್ಮ ಸಾಮರ್ಥ್ಯ ಮತ್ತು ಸುಧಾರಿಸಬೇಕಾದ ಕ್ಷೇತ್ರಗಳ ಬಗ್ಗೆ ತಿಳಿಸಿ.',
                    'ನಿಮ್ಮ ಪ್ರಕಾರ ಯಶಸ್ವಿ ತಂಡದ ಕೆಲಸ (Teamwork) ಎಂದರೆ ಏನು?',
                    'ಒತ್ತಡದ ಸಂದರ್ಭದಲ್ಲಿ ನೀವು ಕೆಲಸವನ್ನು ಹೇಗೆ ನಿರ್ವಹಿಸುತ್ತೀರಿ?'],
             'Behavioural': ['ತಂಡದ ಸದಸ್ಯರೊಂದಿಗೆ ಭಿನ್ನಾಭಿಪ್ರಾಯ ಬಂದಾಗ ಅದನ್ನು ಹೇಗೆ ಸರಿಪಡಿಸಿಕೊಂಡಿದ್ದೀರಿ?',
                             'ಕಠಿಣ ಗಡುವಿನಲ್ಲಿ (Tight Deadline) ಕೆಲಸ ಮಾಡಿ ಯಶಸ್ವಿಯಾದ ಅನುಭವವನ್ನು ಹಂಚಿಕೊಳ್ಳಿ.',
                             'ನೀವು ಮಾಡಿದ ತಪ್ಪು ಮತ್ತು ಅದರಿಂದ ಕಲಿತ ಪಾಠವೇನು?',
                             'ನಿಮ್ಮ ಜವಾಬ್ದಾರಿ ಮೀರಿ ತಂಡಕ್ಕೆ ನೆರವಾದ ಸಂದರ್ಭವನ್ನು ವಿವರಿಸಿ.',
                             'ಪ್ರಾಜೆಕ್ಟ್ ಮಧ್ಯದಲ್ಲಿ ಅವಶ್ಯಕತೆಗಳು ಬದಲಾದಾಗ ನೀವು ಹೇಗೆ ಸ್ಪಂದಿಸಿದ್ದೀರಿ?',
                             'ನಿಮ್ಮ ಕಾರ್ಯಕ್ಷಮತೆ ಸುಧಾರಿಸಲು ನೀಡಿದ ಸಲಹೆಯನ್ನು ಹೇಗೆ ಸ್ವೀಕರಿಸಿದ್ದೀರಿ?',
                             'ಹೊಸ ತಂತ್ರಜ್ಞಾನವನ್ನು ಅಲ್ಪಾವಧಿಯಲ್ಲಿ ಕಲಿತು ಅಳವಡಿಸಿಕೊಂಡ ಉದಾಹರಣೆ ನೀಡಿ.',
                             'ತಂಡದ ಗುರಿಯನ್ನು ತಲುಪಲು ಬೇರೆಯವರಿಗೆ ಮಾರ್ಗದರ್ಶನ ನೀಡಿದ ಅನುಭವವನ್ನು ಹಂಚಿಕೊಳ್ಳಿ.'],
             'Communication': ['ತಾಂತ್ರಿಕ ವಿಷಯವನ್ನು ತಾಂತ್ರಿಕೇತರ ವ್ಯಕ್ತಿಗೆ ಹೇಗೆ ಸರಳವಾಗಿ ವಿವರಿಸುತ್ತೀರಿ?',
                               'ನಿಮ್ಮ ಇತ್ತೀಚಿನ ಪ್ರಾಜೆಕ್ಟ್ ಆರ್ಕಿಟೆಕ್ಚರ್ ಅನ್ನು ಎರಡು ನಿಮಿಷಗಳಲ್ಲಿ ಹೇಗೆ ವಿವರಿಸುತ್ತೀರಿ?',
                               'ತಂಡದ ಸದಸ್ಯರಿಗೆ ಕನ್ಸ್ಟ್ರಕ್ಟಿವ್ ಫೀಡ್\u200cಬ್ಯಾಕ್ ನೀಡುವಾಗ ನೀವು ಯಾವ ವಿಧಾನ ಬಳಸುತ್ತೀರಿ?',
                               'ಸಂವಹನದ ಕೊರತೆಯಿಂದ ಉಂಟಾದ ಸಮಸ್ಯೆಯನ್ನು ನೀವು ಹೇಗೆ ಪರಿಹರಿಸಿದ್ದೀರಿ?',
                               'ಪ್ರಾಜೆಕ್ಟ್ ತಡವಾಗುವ ಸಾಧ್ಯತೆ ಇದ್ದಾಗ ಮ್ಯಾನೇಜ್\u200cಮೆಂಟ್\u200cಗೆ ಹೇಗೆ ತಿಳಿಸುತ್ತೀರಿ?',
                               'ಕೋಡ್ ರಿವ್ಯೂ ಪ್ರಕ್ರಿಯೆಯಲ್ಲಿ ನಿಮ್ಮ ಅಭಿಪ್ರಾಯಗಳನ್ನು ಹೇಗೆ ಸ್ಪಷ್ಟವಾಗಿ ತಿಳಿಸುತ್ತೀರಿ?'],
             'Technical + HR': ['ನಿಮ್ಮ ಬಗ್ಗೆ ಮತ್ತು ನಿಮ್ಮ ತಾಂತ್ರಿಕ ಹಿನ್ನೆಲೆಯ ಬಗ್ಗೆ ಸಂಕ್ಷಿಪ್ತವಾಗಿ ತಿಳಿಸಿ.',
                                'ಸ್ಕೆಲೇಬಲ್ ಸಾಫ್ಟ್\u200cವೇರ್ ಸಿಸ್ಟಮ್ ವಿನ್ಯಾಸಗೊಳಿಸುವಾಗ ತಂಡದ ಸಹಯೋಗ ಹೇಗೆ ಮಾಡುತ್ತೀರಿ?',
                                'ನೀವು ಪರಿಹರಿಸಿದ ಅತ್ಯಂತ ಕಠಿಣ ಬಗ್ (Bug) ಯಾವುದು ಮತ್ತು ಅದರ ಅನುಭವವೇನು?',
                                'ಹೊಸ ತಂತ್ರಜ್ಞಾನಗಳನ್ನು ಕಲಿಯುತ್ತಾ ಪ್ರಾಜೆಕ್ಟ್ ಕೆಲಸಗಳನ್ನು ಹೇಗೆ ನಿಭಾಯಿಸುತ್ತೀರಿ?',
                                'ನಿಮ್ಮ ಕೋಡ್ ಕ್ವಾಲಿಟಿ ಮತ್ತು ಪೀರ್ ರಿವ್ಯೂಗಳನ್ನು ಹೇಗೆ ನಿರ್ವಹಿಸುತ್ತೀರಿ?',
                                'ಉತ್ಪಾದನಾ ಹಂತದಲ್ಲಿ (Production) ಉಂಟಾದ ಸಮಸ್ಯೆಯನ್ನು ಹೇಗೆ ಬಗೆಹರಿಸಿದ್ದೀರಿ?',
                                'ಸಾಫ್ಟ್\u200cವೇರ್ ಇಂಜಿನಿಯರಿಂಗ್\u200cನಲ್ಲಿ ನಿಮ್ಮ ಪ್ರಮುಖ ತತ್ವಗಳು ಯಾವುವು?'],
             'Full Placement Interview': ['ನಿಮ್ಮ ಪರಿಚಯ ನೀಡಿ ಮತ್ತು ಡೇಟಾಬೇಸ್ ಇಂಡೆಕ್ಸಿಂಗ್ ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ ಎಂದು '
                                          'ವಿವರಿಸಿ.',
                                          'ಸಾಫ್ಟ್\u200cವೇರ್ ಆರ್ಕಿಟೆಕ್ಚರ್\u200cನಲ್ಲಿ ಮೈಕ್ರೋಸರ್ವಿಸಸ್ ಮತ್ತು ಮೊನೊಲಿತ್ '
                                          'ನಡುವಿನ ವ್ಯತ್ಯಾಸವೇನು?',
                                          'ತಾಂತ್ರಿಕ ಸವಾಲನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಮುನ್ನಡೆಸಿದ ಅನುಭವವನ್ನು ಹಂಚಿಕೊಳ್ಳಿ.',
                                          'ಕೋಡ್ ಟೆಸ್ಟಿಂಗ್ ಮತ್ತು ಕ್ವಾಲಿಟಿ ಅಶ್ಯುರೆನ್ಸ್\u200cನಲ್ಲಿ ನಿಮ್ಮ ಅನುಭವವೇನು?',
                                          'ಭವಿಷ್ಯದ ತಾಂತ್ರಿಕ ಬೆಳವಣಿಗೆಗಳಿಗೆ ನೀವು ಹೇಗೆ ಸಿದ್ಧರಾಗುತ್ತಿದ್ದೀರಿ?',
                                          'ಒಂದು ದೊಡ್ಡ ಪ್ರಮಾಣದ ಅಪ್ಲಿಕೇಶನ್\u200cನಲ್ಲಿ ಡೇಟಾ ನಷ್ಟವಾಗದಂತೆ ಹೇಗೆ '
                                          'ರಕ್ಷಿಸುತ್ತೀರಿ?',
                                          'ನಿಮ್ಮ ಇಂಜಿನಿಯರಿಂಗ್ ಪ್ರಯಾಣದಲ್ಲಿ ನೀವು ಅತ್ಯಂತ ಹೆಮ್ಮೆಪಡುವ ಸಾಧನೆ ಯಾವುದು?']},
 'Hindi': {'Technical': ['Process और Thread के बीच क्या अंतर है, समझाइए।',
                         'Hash Map में सर्च करने की Time Complexity क्या होती है और क्यों?',
                         'Linked List में cycle का पता कैसे लगाते हैं?',
                         'Array और Linked List के बीच प्रमुख अंतर क्या हैं?',
                         'Database में ACID properties क्यों महत्वपूर्ण हैं?',
                         'Synchronous और Asynchronous programming में क्या अंतर है?',
                         'Caching रणनीतियों को high-load सिस्टम में कैसे लागू किया जाता है?',
                         'Distributed systems में CAP theorem क्या समझाता है?',
                         'Database indexing कैसे काम करती है और क्वेरी प्रदर्शन को कैसे बढ़ाती है?',
                         'RESTful API डिजाइन करते समय किन बातों का ध्यान रखना चाहिए?'],
           'HR': ['अपने बारे में बताइए।',
                  'किसी टीम प्रोजेक्ट में आई चुनौती और उसके समाधान के बारे में बताएं।',
                  'आप हमारी कंपनी क्यों जॉइन करना चाहते हैं?',
                  'अगले तीन वर्षों में आप खुद को कहाँ देखते हैं?',
                  'एक साथ कई कार्य आने पर आप प्राथमिकताओं को कैसे तय करते हैं?',
                  'अपनी सबसे बड़ी ताकत और सुधार के क्षेत्रों के बारे में बताएं।',
                  'आपके अनुसार एक सफल टीम वर्क का क्या अर्थ है?',
                  'तनाव और दबाव की स्थिति में आप काम को कैसे संभालते हैं?'],
           'Behavioural': ['जब टीम में मतभेद हुआ तो आपने उसे कैसे सुलझाया?',
                           'कठिन समय-सीमा (Tight Deadline) के दबाव में आपने काम कैसे पूरा किया?',
                           'अपने किसी अनुभव के बारे में बताएं जहाँ आपसे गलती हुई और आपने उससे क्या सीखा?',
                           'क्या आपने कभी अपनी सामान्य ज़िम्मेदारी से आगे बढ़कर किसी समस्या को हल किया है?',
                           'प्रोजेक्ट के बीच में आवश्यकताएं बदलने पर आपने कैसे सामंजस्य बिठाया?',
                           'किसी सहकर्मी से मिली रचनात्मक प्रतिक्रिया (Feedback) को आपने कैसे लागू किया?',
                           'कम समय में कोई नई तकनीक सीखकर लागू करने का अनुभव साझा करें।',
                           'टीम के लक्ष्य को हासिल करने के लिए आपने किसी साथी की मदद कैसे की?'],
           'Communication': ['एक जटिल तकनीकी विषय को किसी गैर-तकनीकी व्यक्ति को कैसे समझाएंगे?',
                             'अपने हालिया प्रोजेक्ट आर्किटेक्चर को दो मिनट में संक्षेप में कैसे समझाएंगे?',
                             'किसी सहकर्मी को कंस्ट्रक्टिव फीडबैक देते समय आप किन बातों का ध्यान रखते हैं?',
                             'संवाद की कमी (Miscommunication) से उत्पन्न समस्या को आपने कैसे ठीक किया?',
                             'प्रोजेक्ट में देरी होने की संभावना पर आप मैनेजमेंट को कैसे सूचित करते हैं?',
                             'कोड समीक्षा (Code Review) में अपने विचारों को स्पष्ट रूप से कैसे प्रस्तुत करते हैं?'],
           'Technical + HR': ['अपने और अपने तकनीकी कौशल के बारे में संक्षेप में बताएं।',
                              'स्केलेबल एपीआई बनाते समय आप टीम के साथ समन्वय कैसे करते हैं?',
                              'अब तक का सबसे चुनौतीपूर्ण बग कौन सा था जिसे आपने सुलझाया?',
                              'दैनिक कार्यों के साथ-साथ आप नए तकनीकी कौशल कैसे सीखते हैं?',
                              'कोड गुणवत्ता और पीयर रिव्यू को आप किस तरह प्राथमिकता देते हैं?',
                              'प्रोडक्शन एनवायरनमेंट में आई किसी समस्या को आपने कैसे डीबग किया?',
                              'सॉफ्टवेयर डेवलपमेंट में आपके मुख्य सिद्धांत क्या हैं?'],
           'Full Placement Interview': ['अपना परिचय दें और रिलेशनल डेटाबेस में इंडेक्सिंग कैसे काम करती है बताएं।',
                                        'माइक्रोसर्विसेज और मोनोलिथिक आर्किटेक्चर के बीच क्या अंतर है?',
                                        'किसी तकनीकी प्रोजेक्ट का नेतृत्व करने का अपना अनुभव बताएं।',
                                        'सिस्टम की विश्वसनीयता बनाए रखने के लिए आप टेस्टिंग कैसे करते हैं?',
                                        'तकनीकी क्षेत्र में हो रहे बदलावों के साथ आप खुद को कैसे अपडेट रखते हैं?',
                                        'हाई ट्रैफिक सिस्टम में डेटा लॉस रोकने के लिए क्या उपाय करते हैं?',
                                        'अपने करियर की उस उपलब्धि के बारे में बताएं जिस पर आपको सबसे ज्यादा गर्व है।']},
 'Tamil': {'Technical': ['Process மற்றும் Thread இடையேயான வித்தியாசத்தை விளக்குங்கள்.',
                         'Hash Map-ல் தேடுவதற்கான Time Complexity என்ன மற்றும் ஏன்?',
                         'Linked List-ல் cycle-ஐ எவ்வாறு கண்டறிவீர்கள்?',
                         'Array மற்றும் Linked List இடையேயான முக்கிய வேறுபாடுகள் என்ன?',
                         'Database-ல் ACID பண்புகள் ஏன் மிக முக்கியமானவை?',
                         'Synchronous மற்றும் Asynchronous programming-ன் வேறுபாடுகள் யாவை?',
                         'கணினி கட்டமைப்பில் Caching உத்திகளை எவ்வாறு செயல்படுத்துவீர்கள்?',
                         'Distributed systems-ல் CAP தேற்றம் (CAP Theorem) எதை விளக்குகிறது?',
                         'Database Indexing எவ்வாறு இயங்குகிறது மற்றும் வினவல் வேகத்தை எவ்வாறு கூட்டுகிறது?',
                         'REST API வடிவமைக்கும் போது பின்பற்ற வேண்டிய சிறந்த நடைமுறைகள் யாவை?'],
           'HR': ['உங்களைப் பற்றி சுருக்கமாக கூறுங்கள்.',
                  'ஒரு குழு திட்டத்தில் நீங்கள் எதிர்கொண்ட சவாலை எவ்வாறு கையாண்டீர்கள்?',
                  'எங்கள் நிறுவனத்தில் இணைய நீங்கள் ஏன் விரும்புகிறீர்கள்?',
                  'அடுத்த மூன்று ஆண்டுகளில் உங்களை எந்த நிலையில் பார்க்க விரும்புகிறீர்கள்?',
                  'ஒரே நேரத்தில் பல பணிகள் வரும்போது முன்னுரிமைகளை எவ்வாறு தீர்மானிக்கிறீர்கள்?',
                  'உங்கள் பலம் மற்றும் மேம்படுத்த வேண்டிய திறன்களைப் பற்றி கூறுங்கள்.',
                  'உங்கள் பார்வையில் சிறந்த குழுப்பணி (Teamwork) என்பது என்ன?',
                  'அழுத்தமான சூழ்நிலைகளில் பணியை எவ்வாறு திறம்பட நிர்வகிக்கிறீர்கள்?'],
           'Behavioural': ['குழு உறுப்பினருடன் கருத்து வேறுபாடு ஏற்பட்டபோது அதை எவ்வாறு தீர்த்தீர்கள்?',
                           'குறுகிய காலக்கெடுவில் (Tight Deadline) ஒரு திட்டத்தை வெற்றிகரமாக முடித்த அனுபவம் உள்ளதா?',
                           'நீங்கள் செய்த ஒரு தவறு மற்றும் அதிலிருந்து நீங்கள் கற்றுக்கொண்ட பாடம் என்ன?',
                           'உங்கள் வழக்கமான பொறுப்பைத் தாண்டி ஒரு சிக்கலைத் தீர்த்த நிகழ்வைப் பகிருங்கள்.',
                           'திட்டத்தின் நடுவில் தேவைகள் மாறியபோது நீங்கள் எவ்வாறு மாற்றியமைத்துக் கொண்டீர்கள்?',
                           'உங்களுக்கு அளிக்கப்பட்ட ஆக்கபூர்வமான கருத்துக்களை (Feedback) எவ்வாறு செயல்படுத்தினீர்கள்?',
                           'குறைந்த காலத்தில் ஒரு புதிய தொழில்நுட்பத்தைக் கற்று செயல்படுத்திய அனுபவத்தைக் கூறுங்கள்.',
                           'குழு இலக்கை அடைய சக ஊழியருக்கு வழிகாட்டிய அனுபவத்தைப் பகிருங்கள்.'],
           'Communication': ['ஒரு தொழில்நுட்ப கருத்தை தொழில்நுட்பம் அல்லாத ஒருவருக்கு எவ்வாறு விளக்குவீர்கள்?',
                             'உங்கள் சமீபத்திய ப்ராஜெக்ட் கட்டமைப்பை இரண்டு நிமிடங்களில் எவ்வாறு விளக்குவீர்கள்?',
                             'சக ஊழியருக்கு ஃபீட்பேக் (Feedback) வழங்கும் போது நீங்கள் கையாளும் முறை என்ன?',
                             'தகவல் தொடர்புக் குறைபாட்டினால் ஏற்பட்ட சிக்கலை எவ்வாறு சரிசெய்தீர்கள்?',
                             'ப்ராஜெக்ட் காலதாமதம் ஏற்படும் வாய்ப்பு இருந்தால் நிர்வாகத்திற்கு எவ்வாறு '
                             'தெரிவிப்பீர்கள்?',
                             'கோட் ரிவியூவில் (Code Review) உங்கள் கருத்துக்களை எவ்வாறு தெளிவாக முன்வைப்பீர்கள்?'],
           'Technical + HR': ['உங்களைப் பற்றியும் உங்கள் தொழில்நுட்ப திறன்களைப் பற்றியும் கூறுங்கள்.',
                              'ஸ்கேலபிள் ஏபிஐ அமைக்கும் போது குழுவுடன் எவ்வாறு ஒருங்கிணைந்து செயல்படுகிறீர்கள்?',
                              'நீங்கள் சரிசெய்த மிகக் கடினமான பக் (Bug) எது மற்றும் அதன் அனுபவம் என்ன?',
                              'பணிச்சுமைக்கு இடையே புதிய தொழில்நுட்பங்களை எவ்வாறு கற்றுக் கொள்கிறீர்கள்?',
                              'கோட் தரம் மற்றும் பியர் ரிவியூக்களை எவ்வாறு பராமரிக்கிறீர்கள்?',
                              'லைவ் புரொடக்ஷனில் ஏற்பட்ட சிக்கலை எவ்வாறு தீர்த்தீர்கள்?',
                              'மென்பொருள் உருவாக்கத்தில் உங்கள் முக்கிய கொள்கைகள் யாவை?'],
           'Full Placement Interview': ['உங்களை அறிமுகப்படுத்தி, டேட்டாபேஸ் இன்டெக்சிங் எவ்வாறு செயல்படுகிறது என்பதை '
                                        'விளக்குங்கள்.',
                                        'மைக்ரோசர்வீசஸ் மற்றும் மோனோலிதிக் அமைப்புகளுக்கு இடையேயான வேறுபாடு என்ன?',
                                        'ஒரு தொழில்நுட்ப சவாலை முன்னின்று நடத்திய அனுபவத்தைப் பகிருங்கள்.',
                                        'மென்பொருளின் நம்பகத்தன்மையை உறுதி செய்ய நீங்கள் மேற்கொள்ளும் சோதனைகள் என்ன?',
                                        'எதிர்கால தொழில்நுட்ப மாற்றங்களுக்கு நீங்கள் எவ்வாறு தயாராகி வருகிறீர்கள்?',
                                        'அதிக டிராஃபிக் கொண்ட அமைப்பில் டேட்டா இழப்பை எவ்வாறு தடுக்கிறீர்கள்?',
                                        'உங்கள் மென்பொருள் பயணத்தில் நீங்கள் மிகவும் பெருமைப்படும் சாதனை எது?']},
 'Telugu': {'Technical': ['Process మరియు Thread మధ్య తేడాను వివరించండి.',
                          'Hash Map లో వెతకడానికి Time Complexity ఎంత మరియు ఎందుకు?',
                          'Linked List లో cycle ని ఎలా గుర్తిస్తారు?',
                          'Array మరియు Linked List మధ్య ప్రధాన తేడాలు ఏమిటి?',
                          'Database లో ACID లక్షణాలు ఎందుకు ముఖ్యమైనవి?',
                          'Synchronous మరియు Asynchronous programming మధ్య వ్యత్యాసం ఏమిటి?',
                          'సిస్టమ్ లో Caching వ్యూహాలను ఎలా అమలు చేస్తారు?',
                          'Distributed systems లో CAP సిద్ధాంతం ఏమి చెబుతుంది?',
                          'Database Indexing ఎలా పనిచేస్తుంది మరియు క్వెరీ వేగాన్ని ఎలా పెంచుతుంది?',
                          'REST API డిజైన్ చేసేటప్పుడు పాటించాల్సిన ఉత్తమ పద్ధతులు ఏమిటి?'],
            'HR': ['మీ గురించి క్లుప్తంగా చెప్పండి.',
                   'టీమ్ ప్రాజెక్ట్\u200cలో మీరు ఎదుర్కొన్న సవాలు మరియు దానిని ఎలా పరిష్కరించారో చెప్పండి.',
                   'మా కంపెనీలో ఎందుకు చేరాలనుకుంటున్నారు?',
                   'రాబోయే మూడేళ్లలో మిమ్మల్ని మీరు ఎక్కడ చూడాలనుకుంటున్నారు?',
                   'ఒకే సమయంలో అనేక పనులు వచ్చినప్పుడు ప్రాధాన్యతలను ఎలా నిర్ణయిస్తారు?',
                   'మీ బలాలు మరియు మెరుగుపరుచుకోవలసిన అంశాల గురించి చెప్పండి.',
                   'మీ దృష్టిలో విజయవంతమైన టీమ్\u200cవర్క్ అంటే ఏమిటి?',
                   'ఒత్తిడితో కూడిన పరిస్థితులలో పనిని ఎలా నిర్వహిస్తారు?'],
            'Behavioural': ['టీమ్ సభ్యుడితో భేదాభిప్రాయం వచ్చినప్పుడు మీరు దానిని ఎలా పరిష్కరించారు?',
                            'పరిమిత గడువులో (Tight Deadline) ప్రాజెక్ట్\u200cను విజయవంతంగా పూర్తి చేసిన అనుభవం '
                            'చెప్పండి.',
                            'మీరు చేసిన ఏదైనా తప్పు మరియు దాని నుండి నేర్చుకున్న పాఠం ఏమిటి?',
                            'మీ సాధారణ బాధ్యతలకు మించి సమస్యను పరిష్కరించిన సందర్భాన్ని వివరించండి.',
                            'ప్రాజెక్ట్ మధ్యలో అవసరాలు మారినప్పుడు మీరు ఎలా సర్దుబాటు చేసుకున్నారు?',
                            'మీ పనితీరుపై ఇచ్చిన ఫీడ్\u200cబ్యాక్\u200cను ఎలా స్వీకరించి మెరుగుపరుచుకున్నారు?',
                            'తక్కువ సమయంలో కొత్త టెక్నాలజీని నేర్చుకుని అమలు చేసిన అనుభవం తెలపండి.',
                            'టీమ్ లక్ష్యాన్ని సాధించడానికి సహోద్యోగికి మార్గనిర్దేశం చేసిన అనుభవాన్ని పంచుకోండి.'],
            'Communication': ['ఒక సంక్లిష్టమైన సాంకేతిక భావనను సాంకేతికేతర వ్యక్తికి ఎలా వివరిస్తారు?',
                              'మీ ఇటీవలి ప్రాజెక్ట్ ఆర్కిటెక్చర్\u200cను రెండు నిమిషాల్లో ఎలా వివరిస్తారు?',
                              'సహోద్యోగికి కన్స్ట్రక్టివ్ ఫీడ్\u200cబ్యాక్ ఇచ్చేటప్పుడు మీరు ఎలాంటి విధానాన్ని '
                              'అనుసరిస్తారు?',
                              'కమ్యూనికేషన్ లోపం వల్ల వచ్చిన సమస్యను ఎలా సరిదిద్దారు?',
                              'ప్రాజెక్ట్ ఆలస్యమయ్యే అవకాశం ఉన్నప్పుడు యాజమాన్యానికి ఎలా తెలియజేస్తారు?',
                              'కోడ్ రివ్యూలో మీ అభిప్రాయాలను స్పష్టంగా ఎలా వ్యక్తీకరిస్తారు?'],
            'Technical + HR': ['మీ గురించి మరియు మీ సాంకేతిక నైపుణ్యాల గురించి క్లుప్తంగా చెప్పండి.',
                               'స్కేలబుల్ ఏపీఐలను రూపొందించేటప్పుడు టీమ్\u200cతో ఎలా సమన్వయం చేసుకుంటారు?',
                               'మీరు పరిష్కరించిన అత్యంత క్లిష్టమైన బగ్ (Bug) ఏది మరియు దాని అనుభవం ఏమిటి?',
                               'పనితో పాటు కొత్త టెక్నాలజీలను ఎలా నేర్చుకుంటారు?',
                               'కోడ్ నాణ్యత మరియు పీర్ రివ్యూలను ఎలా నిర్వహిస్తారు?',
                               'లైవ్ ప్రొడక్షన్ లో తలెత్తిన సమస్యను ఎలా పరిష్కరించారు?',
                               'సాఫ్ట్\u200cవేర్ ఇంజనీరింగ్\u200cలో మీ ప్రధాన సిద్ధాంతాలు ఏమిటి?'],
            'Full Placement Interview': ['మీ పరిచయం ఇవ్వండి మరియు డేటాబేస్ ఇండెక్సింగ్ ఎలా పనిచేస్తుందో వివరించండి.',
                                         'మైక్రోసర్వీసెస్ మరియు మోనోలిథిక్ ఆర్కిటెక్చర్ మధ్య తేడా ఏమిటి?',
                                         'సాంకేతిక సవాలును విజయవంతంగా నడిపించిన అనుభవాన్ని పంచుకోండి.',
                                         'సిస్టమ్ విశ్వసనీయతను కాపాడటానికి మీరు ఎలాంటి టెస్టింగ్ నిర్వహిస్తారు?',
                                         'భవిష్యత్తు సాంకేతిక మార్పులకు మీరు ఎలా సిద్ధమవుతున్నారు?',
                                         'హై ట్రాఫిక్ సిస్టమ్\u200cలో డేటా నష్టపోకుండా ఎలాంటి జాగ్రత్తలు తీసుకుంటారు?',
                                         'మీ ఇంజనీరింగ్ కెరీర్\u200cలో మీరు అత్యంత గర్వపడే విజయం ఏది?']},
 'Malayalam': {'Technical': ['Process ഉം Thread ഉം തമ്മിലുള്ള വ്യത്യാസം വിശദീകരിക്കുക.',
                             'Hash Map ൽ തിരയുന്നതിനുള്ള Time Complexity എന്താണ്, എന്തുകൊണ്ട്?',
                             'Linked List ൽ ഒരു cycle എങ്ങനെ കണ്ടെത്താം?',
                             'Array യും Linked List ഉം തമ്മിലുള്ള പ്രധാന വ്യത്യാസങ്ങൾ എന്തൊക്കെയാണ്?',
                             'Database ൽ ACID ഗുണവിശേഷങ്ങൾ എന്തുകൊണ്ട് പ്രധാനമാണ്?',
                             'Synchronous ഉം Asynchronous ഉം പ്രോഗ്രാമിങ് തമ്മിലുള്ള വ്യത്യാസമെന്താണ്?',
                             'സിസ്റ്റത്തിൽ Caching തന്ത്രങ്ങൾ എങ്ങനെ നടപ്പിലാക്കാം?',
                             'Distributed systems ൽ CAP തിയറം എന്തിനെക്കുറിച്ചാണ് വ്യക്തമാക്കുന്നത്?',
                             'Database Indexing എങ്ങനെ പ്രവർത്തിക്കുന്നു, അത് ക്വറി വേഗത എങ്ങനെ വർദ്ധിപ്പിക്കുന്നു?',
                             'ഒരു REST API ഡിസൈൻ ചെയ്യുമ്പോൾ ശ്രദ്ധിക്കേണ്ട പ്രധാന കാര്യങ്ങൾ എന്തൊക്കെയാണ്?'],
               'HR': ['നിങ്ങളെക്കുറിച്ച് സ്വയം പരിചയപ്പെടുത്തുക.',
                      'ഒരു പ്രോജക്റ്റിൽ നിങ്ങൾ നേരിട്ട പ്രധാന വെല്ലുവിളി എന്തായിരുന്നു?',
                      'ഞങ്ങളുടെ കമ്പനിയിൽ ചേരാൻ നിങ്ങൾ ആഗ്രഹിക്കുന്നത് എന്തുകൊണ്ടാണ്?',
                      'അടുത്ത മൂന്ന് വർഷത്തിനുള്ളിൽ നിങ്ങളെ എവിടെ കാണാനാണ് ആഗ്രഹിക്കുന്നത്?',
                      'ഒന്നിലധികം ജോലികൾ ഒരേസമയം വരുമ്പോൾ മുൻഗണനകൾ എങ്ങനെ നിശ്ചയിക്കും?',
                      'നിങ്ങളുടെ പ്രധാന കരുത്തുകളും മെച്ചപ്പെടുത്തേണ്ട കാര്യങ്ങളും എന്തൊക്കെയാണ്?',
                      'നിങ്ങളുടെ അഭിപ്രായത്തിൽ മികച്ച ടീംവർക്ക് എന്നാൽ എന്താണ്?',
                      'സമ്മർദ്ദകരമായ സാഹചര്യങ്ങളിൽ ജോലി എങ്ങനെ കൈകാര്യം ചെയ്യുന്നു?'],
               'Behavioural': ['ഒരു ടീം അംഗവുമായി അഭിപ്രായവ്യത്യാസം ഉണ്ടായപ്പോൾ നിങ്ങളത് എങ്ങനെ പരിഹരിച്ചു?',
                               'കുറഞ്ഞ സമയപരിധിക്കുള്ളിൽ (Tight Deadline) ഒരു പ്രോജക്റ്റ് വിജയകരമായി പൂർത്തിയാക്കിയ '
                               'അനുഭവം പങ്കുവെക്കുക.',
                               'നിങ്ങൾക്ക് സംഭവിച്ച ഒരു തെറ്റും അതിൽ നിന്ന് പഠിച്ച പാഠവും എന്താണ്?',
                               'നിങ്ങളുടെ സാധാരണ ഉത്തരവാദിത്തത്തിനപ്പുറം ടീമിനെ സഹായിച്ച ഒരു സന്ദർഭം വിവരിക്കുക.',
                               'പ്രോജക്റ്റിനിടയിൽ ആവശ്യകതകൾ മാറിയപ്പോൾ നിങ്ങളത് എങ്ങനെ കൈകാര്യം ചെയ്തു?',
                               'നിങ്ങൾക്ക് ലഭിച്ച ഫീഡ്ബാക്ക് എങ്ങനെ ക്രിയാത്മകമായി ഉപയോഗിച്ചു?',
                               'കുറഞ്ഞ സമയം കൊണ്ട് പുതിയ സാങ്കേതികവിദ്യ പഠിച്ച് നടപ്പിലാക്കിയ അനുഭവം വ്യക്തമാക്കുക.',
                               'ടീമിന്റെ ലക്ഷ്യം കൈവരിക്കാൻ സഹപ്രവർത്തകനെ സഹായിച്ച അനുഭവം പങ്കുവെക്കുക.'],
               'Communication': ['ഒരു സങ്കീർണ്ണമായ സാങ്കേതിക ആശയം സാങ്കേതികേതര വ്യക്തിക്ക് എങ്ങനെ ലളിതമായി '
                                 'വിശദീകരിക്കും?',
                                 'നിങ്ങളുടെ സമീപകാല പ്രോജക്റ്റ് ആർക്കിടെക്ചർ രണ്ട് മിനിറ്റിൽ എങ്ങനെ വിശദീകരിക്കും?',
                                 'ഒരു സഹപ്രവർത്തകന് കൺസ്ട്രക്റ്റീവ് ഫീഡ്ബാക്ക് നൽകുമ്പോൾ ശ്രദ്ധിക്കുന്ന കാര്യങ്ങൾ '
                                 'എന്തൊക്കെയാണ്?',
                                 'ആശയവിനിമയത്തിലെ പോരായ്മ കാരണം ഉണ്ടായ ഒരു പ്രശ്നം നിങ്ങൾ എങ്ങനെ പരിഹരിച്ചു?',
                                 'പ്രോജക്റ്റ് വൈകാൻ സാധ്യതയുണ്ടെങ്കിൽ മാനേജ്\u200cമെന്റിനെ എങ്ങനെ അറിയിക്കും?',
                                 'കോഡ് റിവ്യൂവിൽ നിങ്ങളുടെ നിർദ്ദേശങ്ങൾ എങ്ങനെ വ്യക്തമായി അവതരിപ്പിക്കും?'],
               'Technical + HR': ['നിങ്ങളെക്കുറിച്ചും നിങ്ങളുടെ പ്രധാന സാങ്കേതിക കഴിവുകളെക്കുറിച്ചും വ്യക്തമാക്കുക.',
                                  'സ്കേലബിൾ ആയ സിസ്റ്റങ്ങൾ നിർമ്മിക്കുമ്പോൾ ടീമുമായി എങ്ങനെ ഏകോപിപ്പിക്കുന്നു?',
                                  'നിങ്ങൾ പരിഹരിച്ച ഏറ്റവും വെല്ലുവിളി നിറഞ്ഞ ബഗ് (Bug) ഏതാണ്?',
                                  'ജോലിക്കൊപ്പം പുതിയ സാങ്കേതികവിദ്യകൾ എങ്ങനെ പഠിച്ചെടുക്കുന്നു?',
                                  'കോഡ് ക്വാളിറ്റിയും പിയർ റിവ്യൂകളും എങ്ങനെ പരിപാലിക്കുന്നു?',
                                  'പ്രൊഡക്ഷൻ എൻവയോൺമെന്റിൽ ഉണ്ടായ ഒരു പ്രശ്നം എങ്ങനെ പരിഹരിച്ചു?',
                                  'സോഫ്റ്റ്\u200cവെയർ എഞ്ചിനീയറിംഗിൽ നിങ്ങൾ പിന്തുടരുന്ന പ്രധാന തത്വങ്ങൾ '
                                  'എന്തൊക്കെയാണ്?'],
               'Full Placement Interview': ['നിങ്ങളുടെ സ്വയം പരിചയപ്പെടുത്തലും ഡാറ്റാബേസ് ഇൻഡക്സിംഗ് എങ്ങനെ '
                                            'പ്രവർത്തിക്കുന്നുവെന്നും വ്യക്തമാക്കുക.',
                                            'മൈക്രോസർവീസസും മോണോലിത്തിക് ആർക്കിടെക്ചറും തമ്മിലുള്ള വ്യത്യാസമെന്താണ്?',
                                            'ഒരു സാങ്കേതിക വെല്ലുവിളി വിജയകരമായി നയിച്ച അനുഭവം പങ്കുവെക്കുക.',
                                            'സിസ്റ്റത്തിന്റെ വിശ്വാസ്യത ഉറപ്പാക്കാൻ നിങ്ങൾ എന്തൊക്കെ ടെസ്റ്റിംഗ് '
                                            'നടത്തുന്നു?',
                                            'ഭാവി സാങ്കേതിക മാറ്റങ്ങളെ നേരിടാൻ നിങ്ങൾ എങ്ങനെ സജ്ജമാകുന്നു?',
                                            'ഹൈ ട്രാഫിക് ഉള്ള സിസ്റ്റത്തിൽ ഡാറ്റ നഷ്ടപ്പെടാതെ എങ്ങനെ സംരക്ഷിക്കാം?',
                                            'നിങ്ങളുടെ എഞ്ചിനീയറിംഗ് കരിയറിൽ ഏറ്റവും അഭിമാനം തോന്നിയ നേട്ടം ഏതാണ്?']}}

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
    asked_questions: list[str] = []   # Session-wide list of all asked questions
    resume_profile: ResumeProfile | None = None
    identified_gaps: list[str] = []

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
    
    # Extract words/tokens across both Latin and non-Latin Indic scripts
    tokens = [re.sub(r'^[^\w]+|[^\w]+$', '', tok) for tok in cleaned.split()]
    tokens = [t for t in tokens if t]
    all_alnum = [ch for ch in cleaned if ch.isalnum()]

    if len(tokens) < 2 or len(all_alnum) < 3:
        common_short_ok = {
            "yes", "no", "true", "false", "o(1)", "o(n)", "o(n^2)", "o(logn)", "o(nlogn)",
            "tcp", "udp", "stack", "queue", "bfs", "dfs", "tree", "graph", "hashmap", "array",
            "ಹೌದು", "ಸರಿ", "ಇಲ್ಲ", "हाँ", "सही", "नहीं", "ஆம்", "இல்லை", "అవును", "కాదు", "അതെ", "അല്ല"
        }
        cleaned_lower = cleaned.lower()
        if cleaned_lower not in common_short_ok and not any(t.lower() in common_short_ok for t in tokens):
            return {
                "valid": False,
                "status": "insufficient",
                "technical_accuracy": "insufficient",
                "message": "Your answer is too short or incomplete. Please elaborate a bit more.",
                "retry_allowed": True
            }

    # Check for keyboard spam / repetitive sequences / lack of natural vowels
    # Check for repeated characters like 'aaaa', 'zzzz', '1111'
    has_repeated = bool(re.search(r"(.)\1{3,}", cleaned.lower()))

    # Check Latin keyboard spam only when substantial Latin characters are present
    latin_letters = re.findall(r"[A-Za-z]", cleaned)
    latin_vowels = len(re.findall(r"[aeiouAEIOU]", cleaned))
    has_low_vowels = len(latin_letters) >= 8 and (latin_vowels / len(latin_letters)) < 0.18
    has_keyboard_spam = any(seq in cleaned.lower() for seq in ["asdf", "qwerty", "zxcv", "ghjkl"])

    if has_repeated or (has_low_vowels and len(latin_letters) / max(len(all_alnum), 1) > 0.6) or has_keyboard_spam:
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

def stem_word(w: str) -> str:
    """Lightweight English stemmer for morphological variants."""
    w = re.sub(r'(?:ing|tion|tions|ment|ments|ed|es|s)$', '', w)
    w = re.sub(r'differ.*', 'differ', w)
    w = re.sub(r'arch.*', 'architect', w)
    w = re.sub(r'concurr.*', 'concurr', w)
    return w

def is_similar_question(new_q: str, prev_questions: list[str]) -> bool:
    """
    Checks if new_q is identical, substantially overlapping, or targeting the exact same concept/project/skill
    as any previously asked question in the session.
    Guarantees strict session-wide uniqueness against paraphrasing, containment, and near-duplicates.
    """
    if not new_q or not prev_questions:
        return False

    def normalize(text: str) -> tuple[set[str], set[str]]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        stopwords = {
            "the", "a", "an", "in", "on", "at", "of", "and", "or", "to", "for", "is", "are",
            "was", "were", "what", "how", "can", "you", "your", "tell", "me", "about",
            "explain", "describe", "with", "do", "does", "did", "could", "would", "please",
            "also", "when", "which", "give", "example", "mention", "mentioned", "talk",
            "from", "between", "into", "over", "under", "through", "where", "why", "regarding"
        }
        filtered = {w for w in words if w not in stopwords and len(w) > 2}
        base_words = filtered if filtered else set(words)
        stemmed = {stem_word(w) for w in base_words}
        return base_words, stemmed

    words_new, stems_new = normalize(new_q)
    cleaned_new = re.sub(r"[^\w\s]", "", new_q.lower()).strip()

    for prev in prev_questions:
        if not prev:
            continue
        cleaned_prev = re.sub(r"[^\w\s]", "", prev.lower()).strip()
        # Direct exact match
        if cleaned_new == cleaned_prev:
            return True
        # Direct containment for phrases > 15 chars
        if len(cleaned_new) > 15 and len(cleaned_prev) > 15:
            if cleaned_new in cleaned_prev or cleaned_prev in cleaned_new:
                return True

        words_prev, stems_prev = normalize(prev)
        if words_new and words_prev:
            # Check stem overlap
            intersection_stem = stems_new.intersection(stems_prev)
            union_stem = stems_new.union(stems_prev)
            jaccard_stem = len(intersection_stem) / len(union_stem) if union_stem else 0.0
            overlap_min_stem = len(intersection_stem) / min(len(stems_new), len(stems_prev)) if min(len(stems_new), len(stems_prev)) > 0 else 0.0

            # Sensitive threshold for near-duplicate or paraphrased questions
            if jaccard_stem >= 0.40 or overlap_min_stem >= 0.55:
                return True

            # If 3 or more content stems overlap, treat as duplicate
            generic_ignore = {"project", "system", "using", "work", "technical", "experience", "approach", "problem"}
            distinctive_common = [w for w in intersection_stem if w not in generic_ignore]
            if len(distinctive_common) >= 3:
                return True

            # If 2 distinctive keywords overlap and both mention core concepts
            core_concepts = {"process", "thread", "differ", "architect", "bottleneck", "challeng", "trade", "scale", "index", "cach", "cycl", "lock"}
            if len(distinctive_common) >= 2 and any(k in intersection_stem for k in core_concepts):
                return True

    return False


def get_non_repeating_fallback(
    bank: list[str],
    turn_index: int,
    prev_questions: list[str]
) -> str:
    """Selects a question from bank that has not been asked in this session. Returns clear session-exhausted message if fully exhausted."""
    if not bank:
        fallback_msg = "Could you introduce your technical background and primary domain experience?"
        return fallback_msg if not is_similar_question(fallback_msg, prev_questions) else "We have covered all primary topics for this session. Please proceed to the final evaluation."

    unasked = [q for q in bank if not is_similar_question(q, prev_questions)]
    if unasked:
        return unasked[turn_index % len(unasked)]

    # If the bank is completely exhausted, do NOT repeat any previous question.
    return "You have answered all available questions for this round. Please conclude the interview using 'Finish & get report'."


def generate_resume_fallback_question(
    profile: ResumeProfile,
    kind: str,
    turn_index: int,
    gaps: list[str] | None = None,
    lang: str = "English",
    prev_questions: list[str] | None = None
) -> str:
    """
    Deterministically generates progressive resume-grounded questions when LLM is offline or unavailable.
    Guarantees progression across:
      - Multiple distinct projects (avoiding the same project twice)
      - Distinct technical skills (languages, frameworks, databases, cloud tools)
      - Work experience and production challenges
      - Specific identified gaps and real-world trade-offs
    Ensures newly generated questions NEVER duplicate any question already asked in the session.
    """
    gaps = gaps or []
    prev_questions = prev_questions or []

    valid_projects = [p for p in profile.projects if p and p.name]
    skills = profile.skills
    tech_skills = []
    if skills:
        for s in (skills.languages or []):
            if s and s not in tech_skills: tech_skills.append(s)
        for s in (skills.frameworks or []):
            if s and s not in tech_skills: tech_skills.append(s)
        for s in (skills.databases or []):
            if s and s not in tech_skills: tech_skills.append(s)
        for s in (skills.tools_and_cloud or []):
            if s and s not in tech_skills: tech_skills.append(s)

    exp = [e for e in profile.experience if e and e.company]

    # Interleave categories across cycles:
    # cycle 0: Project architecture & scaling
    # cycle 1: Skill deep dive & performance
    # cycle 2: Gap probe / Project bottleneck
    # cycle 3: Experience / Production challenge
    project_candidates = []
    for p in valid_projects:
        tech_str = f" using {', '.join(p.technologies[:3])}" if p.technologies else ""
        project_candidates.append(
            f"In your resume, you highlighted the project '{p.name}'{tech_str}. Can you walk me through its core architecture and the key technical trade-offs you made?"
        )
        project_candidates.append(
            f"Regarding your project '{p.name}', what was the most complex technical hurdle or bottleneck you faced during implementation, and how did you resolve it?"
        )
        project_candidates.append(
            f"For '{p.name}', if you were to redesign the system today for 10x scale, what components would you modify?"
        )
        project_candidates.append(
            f"How did you design automated testing, validation, or observability for '{p.name}'?"
        )

    skill_candidates = []
    for s in tech_skills:
        skill_candidates.append(
            f"Your resume mentions hands-on experience with {s}. How have you applied {s} in production or complex tasks, and how did you handle edge cases or performance bottlenecks?"
        )
    for s in tech_skills:
        skill_candidates.append(
            f"When building systems with {s}, what architectural patterns or testing strategies do you rely on to ensure maintainability and reliability?"
        )
    for s in tech_skills:
        skill_candidates.append(
            f"What are the major trade-offs between {s} and its leading industry alternatives in enterprise systems?"
        )

    exp_candidates = []
    for e in exp:
        role_title = e.role or "Engineer"
        exp_candidates.append(
            f"During your time at {e.company} as {role_title}, what was the most difficult engineering challenge you encountered, and how did you resolve it?"
        )
        exp_candidates.append(
            f"At {e.company}, how did you collaborate with teammates and stakeholders when delivering technical milestones under tight deadlines?"
        )
        exp_candidates.append(
            f"While working at {e.company}, how did you handle code reviews, technical debt, and architectural alignment with other engineers?"
        )

    gap_candidates = []
    for g in gaps:
        gap_candidates.append(
            f"Regarding your technical background, how would you approach working with {g} if required in this role, and have you had exposure to related workflows?"
        )
        gap_candidates.append(
            f"If our engineering team asked you to ramp up quickly on {g} for a mission-critical feature, what would your step-by-step learning and execution strategy look like?"
        )

    # Determine desired bucket based on cycle
    cycle = turn_index % 4
    if cycle == 0:
        bucket_order = [project_candidates, skill_candidates, gap_candidates, exp_candidates]
    elif cycle == 1:
        bucket_order = [skill_candidates, gap_candidates, project_candidates, exp_candidates]
    elif cycle == 2:
        bucket_order = [gap_candidates, exp_candidates, project_candidates, skill_candidates] if gaps else [exp_candidates, project_candidates, skill_candidates]
    else:
        bucket_order = [exp_candidates, project_candidates, skill_candidates, gap_candidates]

    for bucket in bucket_order:
        fresh = [c for c in bucket if not is_similar_question(c, prev_questions)]
        if fresh:
            return fresh[turn_index % len(fresh)]

    # If priority buckets exhausted, check all candidates
    all_candidates = project_candidates + skill_candidates + gap_candidates + exp_candidates
    fresh_all = [c for c in all_candidates if not is_similar_question(c, prev_questions)]
    if fresh_all:
        return fresh_all[turn_index % len(fresh_all)]

    # Dynamic fallback probing general problem solving or domain trade-offs
    dynamic_options = [
        "Could you describe an engineering problem from your resume where your initial approach failed and how you adapted?",
        "When balancing speed-of-delivery versus architectural purity on your resume projects, how did you make that compromise?",
        "What was the most impactful performance optimization or refactoring you executed in any of your projects?",
        "How do you approach documenting complex architectural designs and API specifications for peer developers?"
    ]
    for opt in dynamic_options:
        if not is_similar_question(opt, prev_questions):
            return opt

    # If completely exhausted, controlled non-repeating message
    return "You have comprehensively covered all projects and skills from your resume for this interview. Please click 'Finish & get report'."


@app.post("/api/question")
def question(r: QReq):
    n = len(r.history)
    lang = r.interview_language or "English"
    # Consolidate all session questions from both history and explicit session-wide tracker
    prev_questions = []
    for h in r.history:
        if isinstance(h, dict) and h.get("q"):
            q_str = h["q"].strip()
            if q_str and q_str not in prev_questions:
                prev_questions.append(q_str)
    for q_str in (r.asked_questions or []):
        if q_str and q_str.strip() and q_str.strip() not in prev_questions:
            prev_questions.append(q_str.strip())

    convo = "\n".join(f"Interviewer: {h['q']}\nCandidate: {h['a']}" for h in r.history if isinstance(h, dict) and h.get('q') and h.get('a'))
    domain_clause = f"Technical domain focus: '{r.domain}'." if r.kind in ("Technical", "Technical + HR", "Full Placement Interview") and r.domain else ""

    # Check if structured resume profile is supplied
    has_resume = False
    resume_context_clause = ""
    if r.resume_profile:
        prof = r.resume_profile
        all_skills = []
        if prof.skills:
            if prof.skills.languages: all_skills.append(f"Languages: {', '.join(prof.skills.languages)}")
            if prof.skills.frameworks: all_skills.append(f"Frameworks: {', '.join(prof.skills.frameworks)}")
            if prof.skills.databases: all_skills.append(f"Databases: {', '.join(prof.skills.databases)}")
            if prof.skills.tools_and_cloud: all_skills.append(f"Tools/Cloud: {', '.join(prof.skills.tools_and_cloud)}")

        projects_summary = []
        for p in (prof.projects or []):
            if p and p.name:
                p_tech = f" (Tech: {', '.join(p.technologies)})" if p.technologies else ""
                p_desc = f" - {p.description}" if p.description else ""
                projects_summary.append(f"* {p.name}{p_tech}{p_desc}")

        exp_summary = []
        for e in (prof.experience or []):
            if e and e.company:
                e_role = e.role or "Engineer"
                e_dur = f" ({e.duration})" if e.duration else ""
                exp_summary.append(f"* {e_role} at {e.company}{e_dur}")

        edu_summary = []
        for ed in (prof.education or []):
            if ed and ed.institution:
                deg = ed.degree or "Degree"
                edu_summary.append(f"* {deg} from {ed.institution}")

        gaps_list = r.identified_gaps or []
        has_resume = bool(all_skills or projects_summary or exp_summary or gaps_list)

        if has_resume:
            resume_sections = []
            if prof.contact and prof.contact.name:
                resume_sections.append(f"Candidate Name: {prof.contact.name}")
            if prof.inferred_experience_level:
                resume_sections.append(f"Resume Experience Level: {prof.inferred_experience_level}")
            if prof.summary:
                resume_sections.append(f"Professional Summary: {prof.summary}")
            if all_skills:
                resume_sections.append("Verified Skills from Resume:\n  " + "\n  ".join(all_skills))
            if projects_summary:
                resume_sections.append("Candidate Projects from Resume:\n  " + "\n  ".join(projects_summary))
            if exp_summary:
                resume_sections.append("Past Experience from Resume:\n  " + "\n  ".join(exp_summary))
            if edu_summary:
                resume_sections.append("Education:\n  " + "\n  ".join(edu_summary))
            if gaps_list:
                resume_sections.append("Identified Skill / Knowledge Gaps to probe:\n  " + "\n  ".join(f"* {g}" for g in gaps_list))

            resume_context_clause = "\n\nCANDIDATE RESUME PROFILE (GROUND TRUTH):\n" + "\n\n".join(resume_sections)

    def compute_fallback():
        if has_resume and r.resume_profile:
            return generate_resume_fallback_question(
                profile=r.resume_profile,
                kind=r.kind,
                turn_index=n,
                gaps=r.identified_gaps,
                lang=lang,
                prev_questions=prev_questions
            )
        lang_bank = FALLBACK_Q.get(lang, FALLBACK_Q["English"])
        bank = lang_bank.get(r.kind, lang_bank.get(r.kind.capitalize(), lang_bank.get("Technical", FALLBACK_Q["English"]["Technical"])))
        return get_non_repeating_fallback(bank, n, prev_questions)

    # Structured prompt for Gemini to evaluate previous answer (if any) and adaptively produce the next question in the target language
    system_instruction = (
        f"You are an expert, professional technical and HR interviewer conducting an adaptive interview in {lang}. "
        f"You maintain interview context, ask questions natively in {lang} (using standard English technical terms where appropriate), "
        "evaluate answers based STRICTLY on technical correctness (never penalizing grammar, English fluency, or regional code-mixing), "
        "and adjust question difficulty dynamically."
    )

    prev_q_list_clause = ""
    if prev_questions:
        prev_q_list_clause = "\nPREVIOUSLY ASKED QUESTIONS IN THIS SESSION (STRICTLY FORBIDDEN TO REPEAT OR PARAPHRASE):\n" + "\n".join(f"- {q}" for q in prev_questions)

    resume_instructions = ""
    if has_resume:
        resume_instructions = f"""
RESUME-GROUNDED INTERVIEWING REQUIREMENTS:
1. Ground questions directly in the candidate's actual projects, claimed skills, work experience, and identified gaps shown in the resume profile.
2. PROGRESSION & DIVERSITY MANDATE:
   - Do NOT stay stuck on the same project, tool, or question topic.
   - If a previous question asked about one project (e.g. Project A), PROGRESS to a different project (e.g. Project B), a different claimed language/framework, past work experience, or an identified gap.
   - Across the interview turns, ensure a balanced distribution:
     * Turn 1: Opening / Core architecture of primary project
     * Turn 2: Deep dive into a claimed language or database (e.g. Go concurrency, PostgreSQL indexing, API latency)
     * Turn 3: Technical challenge / trade-off in a second project or past work experience
     * Turn 4: Problem-solving scenario or probing an identified skill gap
3. Prioritize projects and technologies ACTUALLY PRESENT in the resume profile over generic textbook examples.
4. Do NOT ask generic, disconnected questions (e.g., standard generic definitions) when specific resume projects or tech stacks can be explored.
5. Under NO circumstances repeat or paraphrase any previously asked question listed above.
"""

    prompt = f"""Conduct the next turn of an adaptive interview.
Target Role: {r.role}
Candidate Experience: {r.level}
Interview Type: {r.kind}
Interview Language: {lang}
{domain_clause}
Base Difficulty Setting: {r.difficulty}
Current Dynamic Difficulty: {r.current_difficulty or 'Intermediate'}
Target Interview Duration: {r.duration} minutes (Pacing question #{n+1})
{resume_context_clause}
{prev_q_list_clause}

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
   - If the candidate's answer was incomplete or vague: ask a RELEVANT FOLLOW-UP question in {lang} probing that specific gap, BUT do not repeat what was already asked.
3. If this is the start of the interview (history is empty):
   - Ask an appropriate opening question in {lang} tailored to the role, domain, experience level ({r.level}), and candidate's resume (if provided).
4. ABSOLUTE CONSTRAINT: Never repeat, rephrase, or ask a question substantively similar to any question already present in the history or the PREVIOUSLY ASKED QUESTIONS list.
5. Keep questions appropriate for the candidate's experience ({r.level}).
6. Ask exactly ONE concise question (1-2 sentences). Do not provide answers or preambles.
{resume_instructions}
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
            candidate_q = (data.get("question") or "").strip()
            # Verify that the generated question is NOT a duplicate/paraphrase of an earlier question
            if candidate_q and not is_similar_question(candidate_q, prev_questions):
                return {
                    "question": candidate_q,
                    "next_difficulty": data.get("next_difficulty") or r.current_difficulty or "Intermediate",
                    "evaluation_summary": data.get("evaluation_summary") or "",
                    "follow_up": bool(data.get("follow_up", False)),
                    "ai": True
                }
            print(f"[question] Warning: LLM generated duplicate question '{candidate_q}'. Attempting progressive retry.")
            # Immediate retry with explicit rejection in prompt
            retry_prompt = prompt + f"\n\nCRITICAL RETRY REQUIREMENT: Your previous response was rejected because '{candidate_q}' was already asked. You MUST pick a completely different topic, project, or skill."
            res_retry = llm(retry_prompt, system_instruction=system_instruction, response_mime_type="application/json")
            if res_retry:
                data_retry = json.loads(res_retry)
                q_retry = (data_retry.get("question") or "").strip()
                if q_retry and not is_similar_question(q_retry, prev_questions):
                    return {
                        "question": q_retry,
                        "next_difficulty": data_retry.get("next_difficulty") or r.current_difficulty or "Intermediate",
                        "evaluation_summary": data_retry.get("evaluation_summary") or "Progressive retry question",
                        "follow_up": bool(data_retry.get("follow_up", False)),
                        "ai": True
                    }
        except Exception as e:
            print("[question] JSON parse error:", e)

    # Compute fallback lazily so prev_questions incorporates everything
    fallback_q = compute_fallback()
    return {
        "question": fallback_q,
        "next_difficulty": r.current_difficulty or "Intermediate",
        "evaluation_summary": f"Offline fallback question ({lang})." if not has_resume else f"Offline resume fallback question ({lang}).",
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
    resume_profile: ResumeProfile | None = None
    resume_analysis: ResumeAnalysisResponse | None = None

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

    # Compile resume evaluation context if provided
    has_resume = False
    resume_eval_clause = ""
    resume_scoring_prompt = ""
    if r.resume_profile or r.resume_analysis:
        prof = r.resume_profile or (r.resume_analysis.resume_profile if r.resume_analysis else None)
        if prof:
            all_skills = []
            if prof.skills:
                if prof.skills.languages: all_skills.append(f"Languages: {', '.join(prof.skills.languages)}")
                if prof.skills.frameworks: all_skills.append(f"Frameworks: {', '.join(prof.skills.frameworks)}")
                if prof.skills.databases: all_skills.append(f"Databases: {', '.join(prof.skills.databases)}")
                if prof.skills.tools_and_cloud: all_skills.append(f"Tools/Cloud: {', '.join(prof.skills.tools_and_cloud)}")

            proj_list = []
            for p in (prof.projects or []):
                if p and p.name:
                    p_tech = f" (Tech: {', '.join(p.technologies)})" if p.technologies else ""
                    p_desc = f" - {p.description}" if p.description else ""
                    proj_list.append(f"* {p.name}{p_tech}{p_desc}")

            exp_list = []
            for e in (prof.experience or []):
                if e and e.company:
                    e_role = e.role or "Engineer"
                    e_dur = f" ({e.duration})" if e.duration else ""
                    exp_list.append(f"* {e_role} at {e.company}{e_dur}")

            gaps_list = []
            if r.resume_analysis and r.resume_analysis.identified_gaps:
                gaps_list.extend(r.resume_analysis.identified_gaps)

            resume_parts = []
            if prof.contact and prof.contact.name:
                resume_parts.append(f"Candidate: {prof.contact.name}")
            if prof.inferred_experience_level:
                resume_parts.append(f"Experience Level: {prof.inferred_experience_level}")
            if all_skills:
                resume_parts.append("Claimed Technical Skills:\n  " + "\n  ".join(all_skills))
            if proj_list:
                resume_parts.append("Claimed Projects:\n  " + "\n  ".join(proj_list))
            if exp_list:
                resume_parts.append("Claimed Work History:\n  " + "\n  ".join(exp_list))
            if gaps_list:
                resume_parts.append("Identified Gaps for Context:\n  " + "\n  ".join(f"* {g}" for g in gaps_list))

            if resume_parts:
                has_resume = True
                resume_eval_clause = "\n\nCANDIDATE RESUME CLAIMS (REFERENCE):\n" + "\n\n".join(resume_parts)
                resume_scoring_prompt = f"""
  "resume_alignment": {{
    "claim_verification_score": <integer 0-100 indicating how authentically and accurately candidate answers substantiate their resume claims>,
    "verified_skills": ["<skill demonstrated with genuine technical understanding in Q&A>", ...],
    "unsubstantiated_or_weak_claims": ["<specific project, tool, or claim from resume where answers lacked depth, precision, or practical reasoning>", ...],
    "resume_feedback": "<2-3 sentences evaluating candidate depth on claimed projects and skills in {lang}>"
  }},"""

    system_instruction = (
        f"You are an expert senior engineering interviewer and hiring committee member conducting evaluation in {lang}. "
        "Evaluate the candidate's interview session thoroughly and objectively based ONLY on actual evidence. "
        f"Provide all qualitative feedback, questions analysis, strengths, improvements, and recommendations in {lang}. "
        "CRITICAL REQUIREMENT: Score Technical Knowledge and Problem-Solving strictly based on technical correctness, "
        "clarity of principles, and accuracy of concepts. Candidate answers in English, regional language, or code-mixed "
        "regional+English must be evaluated solely for technical substance. Score English Proficiency SEPARATELY without contaminating "
        "or penalizing technical marks for non-native phrasing or grammatical quirks."
    )

    resume_eval_instructions = ""
    if has_resume:
        resume_eval_instructions = """
RESUME-AWARE EVALUATION RULES:
1. Cross-examine the interview transcript against the candidate's claimed resume profile.
2. Evaluate whether the candidate demonstrates genuine, authentic technical understanding of the projects, technologies, and skills they claim.
3. Base your evaluation strictly on the interview Q&A evidence:
   - Do NOT give unearned credit for keywords merely mentioned on the resume if the candidate's explanation is superficial, incorrect, or evasive.
   - Do NOT penalize the candidate for limited past years of experience or natural gaps; evaluate the quality, depth, and practical reasoning shown in their answers.
   - If the candidate provided clear, concrete architectural trade-offs, implementation details, or debugging stories for their projects, reward that technical depth.
4. If a question probed an identified gap or missing skill, evaluate how thoughtfully the candidate navigated the question (e.g. foundational problem-solving, willingness to learn, reasoning from first principles).
"""

    prompt = f"""Evaluate this mock interview transcript and coding test results.
Interview Language: {lang}
Interview Length: {len(r.history)} questions answered
Target Duration: {r.duration_minutes} minutes (Elapsed time: {round(actual_secs/60, 1)} minutes)
Coding tests passed: {r.tests_passed}/{r.tests_total} (Complexity: {r.complexity or 'None'})
{resume_eval_clause}
{resume_eval_instructions}
Interview Q&A:
{json.dumps(qa_list, indent=2)}

Provide an evaluation in JSON conforming EXACTLY to this schema (all strings in {lang}):
{{
  "technical_knowledge": <integer 0-100 based on technical depth and accuracy>,
  "problem_solving": <integer 0-100 based on reasoning, trade-offs, and logical decomposition>,
  "communication": <integer 0-100 based on clarity, structure, and conciseness>,
  "english_proficiency": <integer 0-100 based on language fluency, vocabulary, and grammar>,{resume_scoring_prompt}
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

    # Parse resume alignment if present
    resume_alignment_data = None
    if has_resume:
        raw_ra = llm_eval.get("resume_alignment")
        if isinstance(raw_ra, dict):
            resume_alignment_data = {
                "claim_verification_score": max(0, min(100, int(raw_ra.get("claim_verification_score", tech_k)))),
                "verified_skills": [str(s) for s in raw_ra.get("verified_skills", [])],
                "unsubstantiated_or_weak_claims": [str(c) for c in raw_ra.get("unsubstantiated_or_weak_claims", [])],
                "resume_feedback": str(raw_ra.get("resume_feedback", ""))
            }
        else:
            # Deterministic heuristic fallback when offline
            verified = []
            weak = []
            claimed_skills = []
            if prof and prof.skills:
                claimed_skills = (prof.skills.languages or []) + (prof.skills.frameworks or [])
            all_answers_lower = " ".join(h.get("a", "") for h in r.history).lower()
            for s in claimed_skills[:4]:
                if s.lower() in all_answers_lower:
                    verified.append(s)
            resume_alignment_data = {
                "claim_verification_score": min(95, max(50, int((tech_k + prob_s) / 2))),
                "verified_skills": verified,
                "unsubstantiated_or_weak_claims": weak,
                "resume_feedback": "Candidate responses were evaluated against claimed technical background and project experience."
            }

    # Overall calculation: weighted balance across all core dimensions
    core_scores = [tech_k, prob_s, comm_s, eng_s, time_score]
    if coding_score is not None:
        core_scores.append(coding_score)
    if visual_score is not None:
        core_scores.append(visual_score)

    overall_score = int(sum(core_scores) / len(core_scores))

    eval_result = {
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

    if resume_alignment_data is not None:
        eval_result["resume_alignment"] = resume_alignment_data

    return eval_result

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
