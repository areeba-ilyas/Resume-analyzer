import streamlit as st
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter
import matplotlib.pyplot as plt
import re
from docx import Document
from PyPDF2 import PdfReader
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import textwrap
import os

# Download NLTK resources with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up the app
st.set_page_config(
    page_title="Resume Analyzer Pro",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    :root {
        --primary: #4361ee;
        --secondary: #3a0ca3;
        --accent: #4cc9f0;
        --light: #f8f9fa;
        --dark: #212529;
        --success: #2ec4b6;
        --warning: #ff9f1c;
        --danger: #e63946;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #f0f2f5 0%, #e6e9ef 100%);
        color: var(--dark);
    }
    
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .header {
        text-align: center;
        padding: 30px 0;
        margin-bottom: 40px;
        background: linear-gradient(to right, var(--primary), var(--secondary));
        color: white;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(67, 97, 238, 0.2);
        animation: gradientAnimation 10s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 30px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
    }
    
    .match-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
        border-top: 5px solid var(--primary);
    }
    
    .progress-container {
        height: 35px;
        background: #e9ecef;
        border-radius: 20px;
        margin: 25px 0;
        overflow: hidden;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(to right, var(--accent), var(--primary));
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        transition: width 1.5s ease-in-out;
        box-shadow: 0 3px 10px rgba(67, 97, 238, 0.3);
    }
    
    .keyword-list {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 20px 0;
    }
    
    .keyword {
        background: linear-gradient(to right, #e0f7fa, #b2ebf2);
        color: #006064;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
    }
    
    .missing-keyword {
        background: linear-gradient(to right, #ffebee, #ffcdd2);
        color: #b71c1c;
        padding: 8px 18px;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
    }
    
    .suggestion-card {
        background: linear-gradient(to right, #e8f5e9, #c8e6c9);
        border-left: 5px solid var(--success);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 25px 0;
        text-align: center;
    }
    
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        flex: 1;
        margin: 0 10px;
        border-top: 4px solid var(--accent);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #64748b;
    }
    
    .feature-badge {
        background: var(--accent);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    
    @media (max-width: 768px) {
        .stats-container {
            flex-direction: column;
        }
        .stat-card {
            margin: 10px 0;
        }
        .header h1 {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from files
def extract_text(file):
    if file.type == "application/pdf":
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = Document(io.BytesIO(file.getvalue()))
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    elif file.type == "text/plain":
        try:
            return file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    return ""

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-]', '', text)
    return text.strip().lower()

# Function to analyze resume
def analyze_resume(resume_text, job_description):
    # Load spaCy model with error handling
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.warning("Downloading spaCy model... This may take a few minutes.")
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None
    
    # Process texts
    resume_doc = nlp(clean_text(resume_text))
    job_doc = nlp(clean_text(job_description))
    
    # Extract important keywords
    job_keywords = [token.lemma_.lower() for token in job_doc 
                   if not token.is_stop and not token.is_punct 
                   and len(token.text) > 2 and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']]
    
    # Count keyword frequency
    keyword_counter = Counter(job_keywords)
    top_keywords = [word for word, count in keyword_counter.most_common(30)]
    
    # Create phrase matcher
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    patterns = [nlp(keyword) for keyword in top_keywords]
    matcher.add("JOB_KEYWORDS", None, *patterns)
    
    # Find matches in resume
    matches = matcher(resume_doc)
    matched_keywords = set()
    for match_id, start, end in matches:
        matched_keywords.add(resume_doc[start:end].lemma_.lower())
    
    # Calculate match percentage
    match_percentage = min(100, round(len(matched_keywords) / len(top_keywords) * 100)) if top_keywords else 0
    
    # Find missing keywords
    missing_keywords = [keyword for keyword in top_keywords if keyword.lower() not in matched_keywords]
    
    # Generate suggestions
    suggestions = []
    if match_percentage < 70:
        suggestions.append(f"Add these keywords to your resume: {', '.join(missing_keywords[:5])}")
    if len(resume_text.split()) < 300:
        suggestions.append("Your resume seems too short. Consider adding more details.")
    if len(resume_text.split()) > 800:
        suggestions.append("Your resume might be too long. Try to keep it concise.")
    
    # Extract skills section
    skills_section = ""
    for sent in resume_doc.sents:
        if "skill" in sent.text.lower() or "expertise" in sent.text.lower():
            skills_section = sent.text
            break
    
    # Calculate resume stats
    word_count = len(resume_text.split())
    sentence_count = len(list(resume_doc.sents))
    avg_sentence_length = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
    
    return {
        "match_percentage": match_percentage,
        "matched_keywords": list(matched_keywords),
        "missing_keywords": missing_keywords,
        "top_keywords": top_keywords,
        "suggestions": suggestions,
        "skills_section": skills_section,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length
    }

# Function to create word cloud
def create_wordcloud(text, title):
    # Preprocess text
    text = clean_text(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words and len(word) > 2]
    processed_text = " ".join(filtered_text)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='viridis',
                          max_words=100).generate(processed_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=22, pad=20)
    plt.axis("off")
    return plt

# Function to wrap text
def wrap_text(text, width=80):
    return "\n".join(textwrap.wrap(text, width=width))

# Main app
def main():
    st.markdown("""
    <div class="header">
        <h1>Resume Analyzer Pro</h1>
        <p>Optimize your resume for Applicant Tracking Systems (ATS) and job requirements</p>
        <div style="margin-top: 15px;">
            <span class="feature-badge">Keyword Analysis</span>
            <span class="feature-badge">ATS Optimization</span>
            <span class="feature-badge">Match Scoring</span>
            <span class="feature-badge">Word Clouds</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
   
    # File upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Upload Your Resume")
        resume_file = st.file_uploader("Supported formats: PDF, DOCX, TXT", 
                                      type=["pdf", "docx", "txt"],
                                      key="resume_upload")
        resume_text = ""
        if resume_file:
            with st.spinner("Extracting text..."):
                resume_text = extract_text(resume_file)
            if resume_text:
                st.success("Resume uploaded successfully!")
                st.info(f"Resume contains {len(resume_text.split())} words")
            else:
                st.warning("Could not extract text from the file")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💼 Enter Job Description")
        job_description = st.text_area("Paste the job description here", height=300)
        if job_description:
            st.success("Job description added!")
            st.info(f"Job description contains {len(job_description.split())} words")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🚀 Analyze Resume Match", use_container_width=True, type="primary"):
        if resume_text and job_description:
            with st.spinner("Analyzing..."):
                analysis = analyze_resume(resume_text, job_description)
                
                if analysis is None:
                    return
                
                # Display match percentage
                st.markdown('<div class="match-card">', unsafe_allow_html=True)
                st.subheader("Resume Match Score")
                st.markdown(f'<div style="font-size: 3.5rem; font-weight: 800; color: var(--primary);">{analysis["match_percentage"]}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-container"><div class="progress-bar" style="width: {analysis["match_percentage"]}%">{analysis["match_percentage"]}% Match</div></div>', unsafe_allow_html=True)
                
                if analysis["match_percentage"] >= 80:
                    st.success("🎉 Excellent match! Your resume aligns well with the job requirements.")
                elif analysis["match_percentage"] >= 60:
                    st.warning("👍 Good match, but could be improved.")
                else:
                    st.error("⚠️ Low match. Your resume needs improvements.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Resume statistics
                st.markdown("""
                <div class="card">
                    <h2>📊 Resume Statistics</h2>
                    <div class="stats-container">
                        <div class="stat-card">
                            <div class="stat-value">{word_count}</div>
                            <div class="stat-label">Total Words</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{sentence_count}</div>
                            <div class="stat-label">Sentences</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{avg_length}</div>
                            <div class="stat-label">Avg. Sentence Length</div>
                        </div>
                    </div>
                </div>
                """.format(
                    word_count=analysis["word_count"],
                    sentence_count=analysis["sentence_count"],
                    avg_length=analysis["avg_sentence_length"]
                ), unsafe_allow_html=True)
                
                # Display keyword analysis
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("✅ Keywords in Your Resume")
                    st.markdown(f"Found {len(analysis['matched_keywords'])}/{len(analysis['top_keywords'])} important keywords")
                    st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                    for keyword in analysis['matched_keywords'][:20]:
                        st.markdown(f'<div class="keyword">{keyword}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("⚠️ Missing Keywords")
                    st.markdown(f"Add these {len(analysis['missing_keywords'])} keywords to improve your resume")
                    st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                    for keyword in analysis['missing_keywords'][:20]:
                        st.markdown(f'<div class="missing-keyword">{keyword}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Suggestions
                    st.subheader("📝 Improvement Suggestions")
                    if analysis['suggestions']:
                        for suggestion in analysis['suggestions']:
                            st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                    else:
                        st.success("Your resume is well optimized for this job!")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Word clouds
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("☁️ Keyword Visualization")
                col5, col6 = st.columns(2)
                with col5:
                    st.pyplot(create_wordcloud(job_description, "Job Description Keywords"))
                with col6:
                    st.pyplot(create_wordcloud(resume_text, "Resume Keywords"))
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Please upload a resume and enter a job description")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by spaCy, NLTK, and Streamlit • Your data is processed locally and not stored</p>
        <p>Resume Analyzer Pro v1.3</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
