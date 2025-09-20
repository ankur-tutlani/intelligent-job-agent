import os
from extractors import extract_text_from_pdf
from prompt_builder import build_resume_prompt
from resume_parser import ask_with_fallback
from save_yaml import save_yaml_to_file

def run_resume_pipeline(resume_path="SampleResume.pdf",filename="userdata_yaml.pkl"):
    resume_text = extract_text_from_pdf(resume_path)
    prompt = build_resume_prompt(resume_text, resume_path)
    answer = ask_with_fallback([
        {"role": "system", "content": "You are an expert resume parser."},
        {"role": "user", "content": prompt}
    ])
    save_yaml_to_file(answer,filename)
    return answer  # Optional: return the result if needed