import streamlit as st
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

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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
        background: #f0f2f5;
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
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 30px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .progress-container {
        height: 35px;
        background: #e9ecef;
        border-radius: 20px;
        margin: 25px 0;
        overflow: hidden;
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
    }
    
    .keyword-list {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
    }
    
    .keyword {
        background: #e0f7fa;
        color: #006064;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .missing-keyword {
        background: #ffebee;
        color: #b71c1c;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .suggestion-card {
        background: #e8f5e9;
        border-left: 5px solid var(--success);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
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
        padding: 15px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        flex: 1;
        margin: 0 10px;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    
    .feature-badge {
        background: var(--accent);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin: 3px;
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
    return text.strip().lower()

# Improved NLP model loading
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        try:
            # Try to download the model if not found
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
        except Exception as e:
            st.error(f"Failed to load NLP model: {str(e)}")
            return None

# Function to analyze resume
def analyze_resume(resume_text, job_description):
    # Load spaCy model
    nlp = load_nlp_model()
    if nlp is None:
        return None
    
    # Process texts
    resume_doc = nlp(clean_text(resume_text))
    job_doc = nlp(clean_text(job_description))
    
    # Extract important keywords
    job_keywords = [token.lemma_.lower() for token in job_doc 
                   if not token.is_stop and not token.is_punct 
                   and len(token.text) > 2]
    
    # Count keyword frequency
    keyword_counter = Counter(job_keywords)
    top_keywords = [word for word, count in keyword_counter.most_common(15)]
    
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
        suggestions.append(f"Add these keywords: {', '.join(missing_keywords[:5])}")
    if len(resume_text.split()) < 300:
        suggestions.append("Resume seems too short. Add more details.")
    if len(resume_text.split()) > 800:
        suggestions.append("Resume might be too long. Keep it concise.")
    
    # Calculate resume stats
    word_count = len(resume_text.split())
    sentence_count = len(list(resume_doc.sents))
    avg_sentence_length = round(word_count / sentence_count, 1) if sentence_count > 0 else 0
    
    return {
        "match_percentage": match_percentage,
        "matched_keywords": list(matched_keywords),
        "missing_keywords": missing_keywords,
        "suggestions": suggestions,
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
    wordcloud = WordCloud(width=600, height=300, 
                          background_color='white',
                          colormap='viridis',
                          max_words=50).generate(processed_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis("off")
    return plt

# Main app
def main():
    st.markdown("""
    <div class="header">
        <h1>Resume Analyzer Pro</h1>
        <p>Optimize your resume for ATS and job requirements</p>
        <div style="margin-top: 10px;">
            <span class="feature-badge">Keyword Analysis</span>
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
            resume_text = extract_text(resume_file)
            if resume_text:
                st.success("Resume uploaded!")
                st.info(f"{len(resume_text.split())} words")
            else:
                st.warning("Could not extract text")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💼 Job Description")
        job_description = st.text_area("Paste job description here", height=250)
        if job_description:
            st.success("Job description added!")
            st.info(f"{len(job_description.split())} words")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🚀 Analyze Resume Match", use_container_width=True, type="primary"):
        if resume_text and job_description:
            with st.spinner("Analyzing..."):
                analysis = analyze_resume(resume_text, job_description)
                
                if analysis is None:
                    return
                
                # Display match percentage
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Resume Match Score")
                st.markdown(f'<div style="font-size: 2.5rem; font-weight: 800; color: var(--primary); text-align: center;">{analysis["match_percentage"]}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-container"><div class="progress-bar" style="width: {analysis["match_percentage"]}%">{analysis["match_percentage"]}% Match</div></div>', unsafe_allow_html=True)
                
                if analysis["match_percentage"] >= 80:
                    st.success("🎉 Excellent match!")
                elif analysis["match_percentage"] >= 60:
                    st.warning("👍 Good match, could be improved")
                else:
                    st.error("⚠️ Low match, needs improvements")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display keyword analysis
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("✅ Found Keywords")
                    st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                    for keyword in analysis['matched_keywords'][:15]:
                        st.markdown(f'<div class="keyword">{keyword}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("⚠️ Missing Keywords")
                    st.markdown('<div class="keyword-list">', unsafe_allow_html=True)
                    for keyword in analysis['missing_keywords'][:15]:
                        st.markdown(f'<div class="missing-keyword">{keyword}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Resume statistics
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Resume Stats")
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown('<div class="stat-value">{}</div>'.format(analysis["word_count"]), unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Total Words</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col6:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown('<div class="stat-value">{}</div>'.format(analysis["sentence_count"]), unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Sentences</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col7:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown('<div class="stat-value">{}</div>'.format(analysis["avg_sentence_length"]), unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Avg. Sentence Length</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggestions
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📝 Suggestions")
                if analysis['suggestions']:
                    for suggestion in analysis['suggestions']:
                        st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
                else:
                    st.success("Your resume is well optimized!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Word clouds
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("☁️ Keyword Visualization")
                col8, col9 = st.columns(2)
                with col8:
                    st.pyplot(create_wordcloud(job_description, "Job Description"))
                with col9:
                    st.pyplot(create_wordcloud(resume_text, "Your Resume"))
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Please upload a resume and enter a job description")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by spaCy, NLTK, and Streamlit • Data processed locally</p>
        <p>Resume Analyzer Pro v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
