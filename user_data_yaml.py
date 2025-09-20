from PyPDF2 import PdfReader
from docx import Document
from config import API_KEY

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

resume_text = extract_text_from_pdf("C:\\Users\\Downloads\\SampleResume.pdf")  # or from .docx

prompt = f"""
Extract the following fields from the resume below:
- Full name
- Email
- Phone number
- Address, Location
- LinkedIn Id
- Github Id
- Current company
- Previous companies
- Experience details from different employers
- Summary of education
- Skills

Return the result in YAML format.
If you can't find Linkedin id, then use this. https://www.linkedin.com/sample-john-doe/
If you can't find Github id, then use this. https://github.com/sample-johndoe
Add this as the resume path, C:\\Users\\Downloads\\SampleResume.pdf

Resume:
{resume_text}
"""

from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
    )

models = [
    "z-ai/glm-4.5-air:free",                 
    # "deepseek/deepseek-v3:free",  
    "deepseek/deepseek-chat-v3.1:free",
    "deepseek/deepseek-r1:free",             
    "deepseek/deepseek-r1-0528:free",        
    # "tng/deepseek-r1t2-chimera:free",  
    "tngtech/deepseek-r1t2-chimera:free",      
    "qwen/qwen3-14b:free",
    "qwen/qwen3-8b:free"
    # "qwen/qwen3-coder:free"

]

def ask_with_fallback(messages):
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages
            )
            print(f"✅ Used model: {model}")
            return resp.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Model {model} failed: {e}")
    raise RuntimeError("All models failed")

answer = ask_with_fallback([

    {"role": "system", "content": "You are an expert resume parser."},
    {"role": "user", "content": prompt}
])
# print(answer)

import pickle
with open("userdata_yaml.pkl", "wb") as f:
    pickle.dump(answer, f)