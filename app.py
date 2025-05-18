import gradio as gr
from pdfminer.high_level import extract_text
from huggingface_hub import InferenceClient
import fitz  # PyMuPDF
import json
import os
import re

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

extract_info_function = {
    "name": "extract_candidate_info",
    "description": "Extracts key personal and professional information from any PDF resume or profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "mobile_number": {"type": "string"},
            "address": {"type": "string"},
            "email": {"type": "string"},
            "linkedin": {"type": "string"},
            "github": {"type": "string"},
            "experience_years": {"type": "string"}
        },
        "required": ["name", "mobile_number", "address", "email", "linkedin", "github", "experience_years"]
    }
}

def build_extraction_prompt(text, schema):
    return f"""
You are an expert data extractor. Carefully read the following text and extract the following fields as accurately as possible.

For each field, output ONLY the direct value asked for. For LinkedIn and GitHub, output only the profile URL. If a value is not present, output NA.

{json.dumps(schema["parameters"]["properties"], indent=2)}

Return your answer as a JSON object matching this schema.

--- BEGIN TEXT ---
{text[:2000]}
--- END TEXT ---
    """.strip()

def extract_pdf_links(pdf_path):
    """Extract all hyperlinks from a PDF file."""
    links = []
    doc = fitz.open(pdf_path)
    for page in doc:
        for link in page.get_links():
            uri = link.get("uri", "")
            if uri:
                links.append(uri)
    doc.close()
    return links

def find_profile_url_from_links(links, platform):
    for link in links:
        if platform in link.lower():
            return link
    return "NA"

def find_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else "NA"

def extract_info_from_pdf(pdf_file):
    pdf_text = extract_text(pdf_file.name)
    links = extract_pdf_links(pdf_file.name)

    extraction_prompt = build_extraction_prompt(pdf_text, extract_info_function)
    response = client.text_generation(
        model=MODEL_ID,
        prompt=extraction_prompt,
        max_new_tokens=512,
        temperature=0.01,
        do_sample=False,
        return_full_text=False
    )

    json_matches = list(re.finditer(r'\{.*?\}', response, re.DOTALL))
    if not json_matches:
        return "Could not find a JSON block in the model output."

    json_str = json_matches[0].group()
    json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', json_str)
    json_str = re.sub(r'\\[nt]', ' ', json_str)

    try:
        data = json.loads(json_str)
    except Exception as e:
        return f"Failed to parse JSON: {e}\n\nModel Output:\n{response}"

    def safe(field):
        val = data.get(field, "")
        if isinstance(val, dict):
            val = val.get("value", "")
        if not val or val in ["NA", "None", "null"]:
            if field == "linkedin" or field == "github":
                return find_profile_url_from_links(links, field)
            if field == "email":
                return find_email(pdf_text)
            return "NA"
        if field == "linkedin" or field == "github":
            url = find_profile_url_from_links(links, field)
            if url != "NA":
                return url
        if field == "email":
            if re.match(r".+@.+\..+", val):
                return val
            return find_email(pdf_text)
        return str(val).strip() if val else "NA"

    pretty = (
        f"Name: {safe('name')}\n"
        f"Mobile Number: {safe('mobile_number')}\n"
        f"Address: {safe('address')}\n"
        f"Email: {safe('email')}\n"
        f"Linkedin: {safe('linkedin')}\n"
        f"Github: {safe('github')}\n"
        f"Experience Years: {safe('experience_years')}"
    )
    return pretty

demo = gr.Interface(
    fn=extract_info_from_pdf,
    inputs=gr.File(type="filepath", label="Upload PDF Resume"),
    outputs="text",
    title="Quebec Solutions, Inc - PDF Resume Info Extractor",
    description="Upload a PDF resume/profile. Extracts Name, Mobile Number, Address, Email, LinkedIn, GitHub, and Experience in Years using open-source AI. Any missing value is shown as NA.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
