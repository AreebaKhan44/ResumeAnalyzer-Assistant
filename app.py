import os
import streamlit as st
import openai
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Get OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📄 Resume Analyzer Assistant")

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''.join(page.extract_text() or '' for page in pdf.pages)
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def get_resume_text(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Only PDF and DOCX formats are supported.")
        return ""

def calculate_similarity(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def get_missing_keywords(jd_text, resume_text):
    jd_words = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    missing = list(jd_words - resume_words)
    return missing[:20]

jd_text = st.text_area("📋 Paste the Job Description here", height=250)
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF or DOCX)", type=['pdf', 'docx'])

if jd_text and uploaded_file:
    resume_text = get_resume_text(uploaded_file)
    score = calculate_similarity(jd_text, resume_text)
    missing_keywords = get_missing_keywords(jd_text, resume_text)

    st.markdown(f"### ✅ Match Score: `{score}%`")
    
    if score > 75:
        st.success("Great match! Your resume is well aligned with the job description.")
    elif score > 50:
        st.warning("Decent match. You can improve your resume.")
    else:
        st.error("Low match. Consider revising your resume to better align with the job description.")

    st.markdown("### ❌ Missing Keywords in Resume:")
    st.write(", ".join(missing_keywords))
