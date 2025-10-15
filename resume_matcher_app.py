import streamlit as st
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- 1. SECURE KEY LOADING (for Streamlit Cloud) ---
try:
    openai_api_key = st.secrets["openai_key"]
except KeyError:
    st.error("🚨 OpenAI API key not found! Please add 'openai_key' in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=openai_api_key)
# --- END SECURE KEY LOADING ---


# --- 2. FIXED NLTK DOWNLOADS FOR STREAMLIT ---
nltk_packages = ['stopwords', 'punkt', 'averaged_perceptron_tagger']

for pkg in nltk_packages:
    try:
        if pkg == "punkt":
            nltk.data.find(f"tokenizers/{pkg}")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)
# --- END FIXED NLTK DOWNLOADS ---


# --- Helper functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

def extract_keywords(text):
    return list(set(text.split()))

def compare_texts(resume, jd):
    vectorizer = CountVectorizer().fit_transform([resume, jd])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return round(similarity[0][1] * 100, 2)

def find_missing_keywords(resume_words, jd_words):
    return [word for word in jd_words if word not in resume_words]

def generate_suggested_keywords(jd_text, resume_text):
    tokens = word_tokenize(jd_text)
    tagged = pos_tag(tokens)
    important_words = [word.lower() for word, tag in tagged if tag.startswith("NN") or tag.startswith("JJ")]
    clean_jd = clean_text(" ".join(important_words))
    jd_keywords = set(clean_jd.split())
    clean_resume = clean_text(resume_text)
    resume_keywords = set(clean_resume.split())
    suggested = [w for w in jd_keywords if w not in resume_keywords and len(w) > 2]
    return suggested[:15]

def ai_feedback(resume, jd):
    """Use OpenAI API to give recruiter-style feedback"""
    prompt = f"""
    You are an expert recruiter and resume analyst.
    Compare the following resume with the job description and give:
    1. A short summary of alignment.
    2. 3–5 specific suggestions to improve the resume.
    3. Any missing skills, tools, or phrases you recommend adding.

    --- Resume ---
    {resume}

    --- Job Description ---
    {jd}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Smart Resume Matcher", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered Smart Resume Matcher")
st.write("Compare your resume with a job description using both keyword analysis and OpenAI intelligence!")

resume_text = st.text_area("📄 Paste your Resume Text here:", height=200)
jd_text = st.text_area("💼 Paste the Job Description here:", height=200)

if st.button("🔍 Analyze & Suggest"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please fill in both fields.")
    else:
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(jd_text)
        resume_keywords = extract_keywords(clean_resume)
        jd_keywords = extract_keywords(clean_jd)

        match_score = compare_texts(clean_resume, clean_jd)
        missing = find_missing_keywords(resume_keywords, jd_keywords)
        matched = [w for w in jd_keywords if w in resume_keywords]
        suggestions = generate_suggested_keywords(jd_text, resume_text)
        
        with st.spinner('🤔 Thinking like an expert recruiter...'):
            feedback = ai_feedback(resume_text, jd_text)

        st.subheader("📊 Results:")
        st.metric("Resume Match Score", f"{match_score}%")

        st.markdown("### ✅ Matching Keywords")
        st.success(", ".join(matched) if matched else "No major keyword matches found.")

        st.markdown("### ❌ Missing Keywords")
        st.warning(", ".join(missing) if missing else "No missing keywords — great match!")

        st.markdown("### 💡 Suggested Keywords")
        st.info(", ".join(suggestions) if suggestions else "Your resume already includes most relevant keywords!")

        st.markdown("### 🤖 AI Recruiter Feedback")
        st.write(feedback)