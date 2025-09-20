def build_resume_prompt(resume_text, resume_path):
    return f"""
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
If you can't find Linkedin id, then use this: https://www.linkedin.com/sample-john-doe/
If you can't find Github id, then use this: https://github.com/sample-johndoe
Add this as the resume path: {resume_path}

Resume:
{resume_text}
"""