from dotenv import load_dotenv
from google import genai
import os

load_dotenv("backend/.env")

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

if not api_key:
    raise ValueError("API key not found. Check backend/.env")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model=model_name,
    contents="ನಮಸ್ಕಾರ! ನಿಮ್ಮ ಬಗ್ಗೆ ಕನ್ನಡದಲ್ಲಿ ಪರಿಚಯಿಸಿಕೊಳ್ಳಿ."
)

print("\nGemini Response:\n")
print(response.text)