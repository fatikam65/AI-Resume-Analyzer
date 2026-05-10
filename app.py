import streamlit as st
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd # New for timeline data
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.lower()

st.set_page_config(page_title="AI Talent Intelligence Dashboard", layout="wide")
st.title("🚀 AI Talent Intelligence Dashboard")

target_skills = ["python", "sql", "c++", "javascript", "excel", "communication", "finance", "hr", "marketing", "writing", "analysis", "management"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
with col2:
    st.subheader("2. Paste Job Description")
    job_desc = st.text_area("Paste requirements here...", height=200).lower()

if uploaded_file is not None and job_desc != "":
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # --- ANALYSIS LOGIC ---
    documents = [resume_text, job_desc]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    
    blob = TextBlob(resume_text)
    confidence_score = (blob.sentiment.polarity + 1) * 50 

    relevant_job_skills = [s for s in target_skills if s in job_desc]
    matched_skills = [s for s in relevant_job_skills if s in resume_text]
    missing_skills = [s for s in relevant_job_skills if s not in resume_text]

    # --- 4. INTERVIEWER'S INTELLIGENCE LOGIC ---
    import re
    # Scans for metrics like "3.7 CGPA", "500+", "18-hour program", or "30-page paper"
    impact_patterns = [r'\d+%', r'\d+\+', r'CGPA', r'SLA', r'revenue', r'optimized', r'led', r'facilitated']
    impact_matches = []
    for pattern in impact_patterns:
        if re.search(pattern, resume_text, re.IGNORECASE):
            impact_matches.append(pattern)

    # Score based on results-oriented language found in the resume
    impact_score = min(len(impact_matches) * 15, 100)

    st.divider()
    
    
    # ROW 1: GAUGES
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        fig_m = go.Figure(go.Indicator(mode="gauge+number", value=sim_score, title={'text': "Match Score"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#3498db"}}))
        st.plotly_chart(fig_m, use_container_width=True)
    with g_col2:
        fig_t = go.Figure(go.Indicator(mode="gauge+number", value=confidence_score, title={'text': "Tone Confidence"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#9b59b6"}}))
        st.plotly_chart(fig_t, use_container_width=True)

    # NEW SECTION: Interviewer Decision Support
    st.divider()
    st.header("📋 Interviewer's Decision Support")
    d_col1, d_col2 = st.columns([1, 2])
    
    with d_col1:
        st.metric("Interviewer's Impact Score", f"{impact_score}%")
        if impact_score > 60:
            st.success("🎯 **High Potential Candidate**")
        else:
            st.warning("⚖️ **Review Required**")
            
    with d_col2:
        st.subheader("💡 Decision Summary")
        
        import re
        # 1. We search for generic quantifiable achievements (CGPAs, percentages, or 3+ digit counts)
        # This ensures the basis is pulled directly from the NEW text, not your history.
        dynamic_metrics = re.findall(r'(?:CGPA \d+\.\d+)|(?:\d+(?:\.\d+)?%)|(?:\d{3,}\+ [a-z]+)', resume_text, re.IGNORECASE)
        
        # 2. We also look for elite titles like "Top Scorer", "Dean's List", or "Gold Medal"
        elite_patterns = r"(top scorer|dean's honours list|medal of excellence|distinction)"
        elite_matches = re.findall(elite_patterns, resume_text, re.IGNORECASE)

        # 3. Combine everything into one professional string
        # We use set() to ensure that if '3.7 CGPA' appears twice, it only shows once.
        combined_basis = list(set([m.strip().title() for m in dynamic_metrics + elite_matches]))
        executive_string = " | ".join(combined_basis[:4]) # Limits to top 4 metrics for neatness

        if executive_string:
            st.write(f"### **Executive Basis:** {executive_string}")
            st.success("🎯 **Hire Recommendation:** Basis for selection is established via detected quantifiable achievements.")
        else:
            st.info("The resume is descriptive; selection basis relies on qualitative review of skills.")

        if matched_skills:
            top_skill = matched_skills[0].title()
            st.info(f"**Suggested Question:** 'You mentioned experience with **{top_skill}**. Can you walk us through a specific problem you solved using this skill?'")
            


    # --- SKILLS PIE CHART ---
    st.divider()
    res_l, res_r = st.columns(2)
    with res_l:
        chart_data = {"Status": ["Matched", "Missing"], "Count": [len(matched_skills), len(missing_skills)]}
        st.plotly_chart(px.pie(chart_data, values="Count", names="Status", color_discrete_sequence=["#2ecc71", "#e74c3c"], hole=0.4))
    with res_r:
        st.subheader("Analysis Insights")
        st.success(f"✅ Strengths: {', '.join(matched_skills).title()}")
        st.warning(f"❌ Gaps: {', '.join(missing_skills).title()}")

        st.divider()
    st.header("🎭 Tone & Sentiment Deep Dive")
    
    tone_col1, tone_col2 = st.columns([1, 1])
    
    with tone_col1:
        st.subheader("Why this percentage?")
        # Using the confidence_score variable which is calculated from the CURRENT upload
        st.write(f"The score of **{round(confidence_score, 1)}%** is based on the linguistic patterns found in this document.")
        
        if confidence_score > 60:
            st.write("The text uses highly confident and achievement-oriented language.")
        elif confidence_score > 45:
            st.write("The language is objective and professional. This is excellent for technical or academic roles where a neutral, factual tone is preferred.")
        else:
            st.write("The tone is currently very descriptive or passive.")

    with tone_col2:
        st.subheader("How to improve the Tone?")
        st.info("💡 **Pro-Tip:** To increase this score, replace passive descriptions with strong action verbs that show ownership of results.")
        
        # We provide general categories that apply to almost any professional CV
        st.markdown("""
        * **For Technical Roles:** Use words like *Engineered*, *Deployed*, or *Optimized*.
        * **For Management:** Use words like *Spearheaded*, *Coordinated*, or *Orchestrated*.
        * **For Analysis:** Use words like *Synthesized*, *Evaluated*, or *Forecasted*.
        """)
    # --- END OF NEW SECTION ---

    # DOWNLOAD BUTTON
    st.download_button("📥 Download Intelligence Report", data=f"Match Score: {sim_score}%", file_name="AI_Report.txt")