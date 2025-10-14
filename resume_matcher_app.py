import streamlit as st
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --- 1. SECURE KEY LOADING (Required for Streamlit Cloud) ---
# Use st.secrets to securely access keys stored in .streamlit/secrets.toml
# This is much safer than hardcoding or using os.environ directly for deployment.
try:
    openai_api_key = st.secrets["openai_key"]
except KeyError:
    st.error("OpenAI API key not found. Please ensure 'openai_key' is set in your Streamlit secrets.")
    st.stop() # Stop the app if the key is missing

client = OpenAI(api_key=openai_api_key)
# --- END SECURE KEY LOADING ---


# --- 2. EFFICIENT NLTK DATA DOWNLOADS (Fixes slow restarts) ---
# A helper function to check and download NLTK data only if it's missing.
def check_and_download_nltk_data(resource):
    """Checks for an NLTK resource and downloads it if missing."""
    try:
        nltk.data.find(resource)
    except (nltk.downloader.DownloadError, LookupError):
        # Extract the resource name (e.g., 'stopwords' from 'corpora/stopwords')
        resource_name = resource.split('/')[-1]
        # print(f"NLTK resource '{resource_name}' not found. Downloading...") # Optional: good for debugging
        nltk.download(resource_name, quiet=True) # quiet=True prevents excessive output

# Call the function for every resource your app uses:
check_and_download_nltk_data('corpora/stopwords')
check_and_download_nltk_data('tokenizers/punkt')
# You only need one of the taggers, 'averaged_perceptron_tagger' is standard
check_and_download_nltk_data('taggers/averaged_perceptron_tagger') 
# check_and_download_nltk_data('corpora/punkt_tab') # Not a standard NLTK path, removed
# check_and_download_nltk_data('corpora/averaged_perceptron_tagger_eng') # Not a standard NLTK path, removed
# --- END EFFICIENT NLTK DOWNLOADS ---


# --- Helper functions ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # The stopwords list is now guaranteed to be downloaded
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
    # NLTK data (punkt, averaged_perceptron_tagger) is now guaranteed to be downloaded
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
    2. 3-5 specific suggestions to improve the resume.
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

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Resume Matcher", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered Smart Resume Matcher")
st.write("Compare your resume with a job description using both *keyword analysis* and *OpenAI intelligence*!")

resume_text = st.text_area("📄 Paste your Resume Text here:", height=200)
jd_text = st.text_area("💼 Paste the Job Description here:", height=200)

if st.button("🔍 Analyze & Suggest"):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please fill in both fields.")
    else:
        # Proceed with analysis
        clean_resume = clean_text(resume_text)
        clean_jd = clean_text(jd_text)
        resume_keywords = extract_keywords(clean_resume)
        jd_keywords = extract_keywords(clean_jd)

        match_score = compare_texts(clean_resume, clean_jd)
        missing = find_missing_keywords(resume_keywords, jd_keywords)
        matched = [w for w in jd_keywords if w in resume_keywords]
        suggestions = generate_suggested_keywords(jd_text, resume_text)
        
        # Add a spinner while generating AI feedback
        with st.spinner('Thinking like an expert recruiter...'):
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