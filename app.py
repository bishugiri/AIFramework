import json
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional

# Load OpenAI API key from Streamlit secrets
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets"""
    return st.secrets.get("OPENAI_API_KEY", "")


import openai
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="IB Analytics AI Readiness Tool", page_icon="🧠", layout="wide")

# Add custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
        padding: 0;
        margin: 0;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem 0;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .header-content {
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 3px solid #e2e8f0;
        box-shadow: 4px 0 20px rgba(0,0,0,0.05);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        margin: -1rem -1rem 0.5rem -1rem;
        border-radius: 0 0 12px 12px;
        text-align: center;
    }
    
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        margin: 0 0.25rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-1px);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 8px 4px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Reset Button */
    .reset-button > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 8px 4px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .reset-button > button:hover {
        background: linear-gradient(135deg, #ff5252 0%, #e53935 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 12px 16px;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.9);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: rgba(255,255,255,0.9);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 12px 16px;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.9);
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    /* Radio Button Styling */
    .stRadio > div {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        border-color: #667eea;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-weight: 600;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
                /* Metric Styling */
                .metric-container {
                    background: rgba(255,255,255,0.95);
                    padding: 0.75rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255,255,255,0.2);
                    margin: 0.25rem 0;
                }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
                /* Card Styling */
                .assessment-card {
                    background: rgba(255,255,255,0.95);
                    border-radius: 12px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                    border: 1px solid rgba(255,255,255,0.2);
                    backdrop-filter: blur(10px);
                    transition: all 0.3s ease;
                }
    
    .assessment-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    /* Progress Indicators */
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Custom Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- Data Persistence ----------------------
def save_to_json():
    """Save all session state data to JSON file"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "onboarding": st.session_state.get("onboarding", {}),
        "data_readiness": st.session_state.get("data_readiness_saved", []),
        "infrastructure": st.session_state.get("infrastructure_saved", []),
        "people": st.session_state.get("people_saved", []),
        "leadership": st.session_state.get("leadership_saved", []),
        "comments": {}
    }
    
    # Calculate and store separate AI Users and AI Builders scores
    people_surveys = st.session_state.get("people_saved", [])
    ai_use_scores = []
    ai_build_scores = []
    
    for survey in people_surveys:
        if survey.get('scores'):
            if survey['type'] == "AI Use (End Users)":
                ai_use_scores.extend(survey['scores'])
            else:  # AI Build
                ai_build_scores.extend(survey['scores'])
    
    data["ai_users_score"] = np.mean(ai_use_scores) if ai_use_scores else 0.0
    data["ai_builders_score"] = np.mean(ai_build_scores) if ai_build_scores else 0.0
    data["ai_users_count"] = len(ai_use_scores)
    data["ai_builders_count"] = len(ai_build_scores)
    
    # Collect all comments
    for key, value in st.session_state.items():
        if isinstance(key, str) and key.endswith("_notes") and isinstance(value, str) and value.strip():
            data["comments"][key] = value.strip()
    
    # Save to file
    filename = f"ai_readiness_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filename

def load_from_json(filename):
    """Load data from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Restore session state
        if "onboarding" in data:
            st.session_state.onboarding = data["onboarding"]
        if "data_readiness" in data:
            st.session_state.data_readiness_saved = data["data_readiness"]
        if "infrastructure" in data:
            st.session_state.infrastructure_saved = data["infrastructure"]
        if "people" in data:
            st.session_state.people_saved = data["people"]
        if "leadership" in data:
            st.session_state.leadership_saved = data["leadership"]
        
        st.success(f"Data loaded successfully from {filename}")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ---------------------- OpenAI Integration ----------------------
def setup_openai():
    """Setup OpenAI client with API key from Streamlit secrets"""
    api_key = get_openai_api_key()
    if not api_key or api_key == "your_openai_api_key_here":
        return None, "API key not found or placeholder detected"
    
    try:
        # Test client creation
        client = openai.OpenAI(api_key=api_key)
        return client, "API key valid"
    except Exception as e:
        return None, f"API key error: {e}"

def analyze_comments_with_ai(comments: str, area: str, score: float, context: str = "") -> Dict:
    """Analyze user comments using OpenAI GPT-4o"""
    client, status = setup_openai()
    if not client:
        return {
            "recommendations": [f"Focus on improving {area} based on current score of {score:.1f}/5.0"],
            "use_cases": [f"Implement {area} best practices"],
            "next_steps": [f"Review {area} processes", f"Train team on {area}", f"Update {area} documentation"],
            "priority": "Medium"
        }
    
    try:
        prompt = f"""
        Analyze the following feedback for an AI readiness assessment in the area of {area}.
        
        Current Score: {score}/5.0
        User Comments: {comments}
        Additional Context: {context}
        
        Please provide:
        1. 3-5 specific, actionable recommendations
        2. 2-3 concrete use cases or examples
        3. 5 prioritized next steps to achieve AI readiness maturity
        4. Priority level (High/Medium/Low) based on the score and comments
        
        Format your response as a JSON object with these exact keys:
        {{
            "recommendations": ["rec1", "rec2", "rec3"],
            "use_cases": ["use case 1", "use case 2"],
            "next_steps": ["step 1", "step 2", "step 3", "step 4", "step 5"],
            "priority": "High/Medium/Low"
        }}
        
        Focus on practical, implementable advice for improving AI readiness. Use knowledge from what market is currently adopting
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse JSON response
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "recommendations": [f"Improve {area} processes and training"],
                "use_cases": [f"Implement {area} best practices"],
                "next_steps": [f"Assess current {area} state", f"Develop {area} roadmap", f"Train team", f"Implement changes", f"Monitor progress"],
                "priority": "Medium"
            }
            
    except Exception as e:
        return {
            "recommendations": [f"Focus on {area} improvement based on score {score:.1f}/5.0"],
            "use_cases": [f"Standardize {area} processes"],
            "next_steps": [f"Review {area} current state", f"Identify gaps", f"Create action plan", f"Implement improvements", f"Measure results"],
            "priority": "Medium"
        }

def get_section_comments(section: str) -> str:
    """Extract all relevant comments from session state for a given section"""
    comments = []
    
    if section == "Data Readiness":
        for i, item in enumerate(st.session_state.get("data_readiness_saved", [])):
            if item.get("notes"):
                comments.append(f"Question {i+1}: {item['notes']}")
    elif section == "Infrastructure":
        for i, item in enumerate(st.session_state.get("infrastructure_saved", [])):
            if item.get("notes"):
                comments.append(f"Question {i+1}: {item['notes']}")
    elif section in ["People - AI Users", "People - AI Builders"]:
        # Check both saved and current people surveys
        for survey in st.session_state.get("people_saved", []):
            if survey.get("notes"):
                comments.append(f"{survey['name']} ({survey['type']}): {survey['notes']}")
        for survey in st.session_state.get("people_surveys", []):
            if survey.get("notes"):
                comments.append(f"{survey['name']} ({survey['type']}): {survey['notes']}")
    elif section == "Leadership & Strategy":
        # Check both saved and current leadership surveys
        for survey in st.session_state.get("leadership_saved", []):
            if survey.get("notes"):
                comments.append(f"{survey['name']} ({survey['role']}): {survey['notes']}")
        for survey in st.session_state.get("leadership_surveys", []):
            if survey.get("notes"):
                comments.append(f"{survey['name']} ({survey['role']}): {survey['notes']}")
    
    return " | ".join(comments) if comments else ""

def generate_basic_recommendations(area: str, score: float) -> List[str]:
    """Generate basic recommendations based on score when AI analysis is not available"""
    if area == "Data Readiness":
        if score < 2:
            return [
                "Create enterprise data inventory, digitize critical records",
                "Centralize storage (DW/lake), and assign data stewards",
                "Establish basic data governance framework",
                "Implement data quality monitoring",
                "Set up backup and recovery procedures"
            ]
        elif score < 3:
            return [
                "Automate ETL/ELT, define DQ rules & monitoring",
                "Stand up catalog + lineage, and enforce access controls",
                "Implement data validation and testing",
                "Establish data retention policies",
                "Create data documentation standards"
            ]
        elif score < 4:
            return [
                "Implement semantic layer, APIs/streaming",
                "Privacy automation, and dataset feedback loops for AI",
                "Advanced data quality and monitoring",
                "Implement data versioning and lineage",
                "Establish data governance committees"
            ]
        else:
            return [
                "Optimize cost/performance, enable vector/RAG ingestion at scale",
                "Formalize observability & lifecycle management",
                "Advanced analytics and AI-ready data pipelines",
                "Implement data mesh architecture",
                "Continuous data governance optimization"
            ]
    elif area == "Infrastructure":
        if score < 2:
            return [
                "Baseline cloud readiness, provision secure environments",
                "Pilot GPU/compute with guardrails",
                "Establish basic security controls",
                "Set up monitoring and alerting",
                "Create disaster recovery plan"
            ]
        elif score < 3:
            return [
                "Standardize MLOps/LLMOps pipelines, registry, CI/CD",
                "Implement observability and monitoring",
                "Establish security best practices",
                "Create infrastructure as code templates",
                "Set up cost monitoring and controls"
            ]
        elif score < 4:
            return [
                "Scale infra with autoscaling, caching, cost tracking",
                "Enforce SRE practices and advanced monitoring",
                "Implement advanced security controls",
                "Optimize performance and reliability",
                "Establish infrastructure governance"
            ]
        else:
            return [
                "Optimize footprints, multi-cloud resilience",
                "Advanced eval/monitoring stacks",
                "Implement AI-specific infrastructure patterns",
                "Continuous optimization and automation",
                "Advanced security and compliance features"
            ]
    elif area == "People — AI Use":
        if score < 2:
            return [
                "Launch AI literacy 101, curated tool list",
                "Safe-use policy; run hands-on clinics",
                "Basic AI training for all employees",
                "Create AI usage guidelines",
                "Establish AI champions program"
            ]
        elif score < 3:
            return [
                "Role-based training, internal champions",
                "Usage playbooks; measure adoption KPIs",
                "Advanced AI tool training",
                "Create AI best practices repository",
                "Implement AI usage tracking"
            ]
        elif score < 4:
            return [
                "Advanced prompts/cookbooks per function",
                "Shared best-practice hub, and office hours",
                "Specialized AI training by department",
                "AI productivity measurement",
                "Cross-functional AI collaboration"
            ]
        else:
            return [
                "Continuous enablement, certification",
                "Cross-team CoPs driving measurable productivity gains",
                "Advanced AI strategy and planning",
                "AI innovation and experimentation",
                "AI leadership development"
            ]
    elif area == "People — AI Build":
        if score < 2:
            return [
                "Upskill builders on Python, APIs, data basics",
                "Pair with mentors; start small POCs",
                "Basic AI/ML training programs",
                "Create learning paths and resources",
                "Establish AI development community"
            ]
        elif score < 3:
            return [
                "Train on RAG, evals, vector DBs, and secure patterns",
                "Create shared templates and frameworks",
                "Intermediate AI development training",
                "Implement development best practices",
                "Create AI development standards"
            ]
        elif score < 4:
            return [
                "Formal SDLC for AI with guardrails",
                "Model registry, and evaluation frameworks",
                "Advanced AI development training",
                "Implement AI development governance",
                "Create AI development platform"
            ]
        else:
            return [
                "Advanced architectures, red-teaming, AB tests",
                "Platform productization and optimization",
                "AI development innovation and research",
                "Advanced AI development methodologies",
                "AI development leadership and strategy"
            ]
    elif area == "Leadership & Strategy":
        if score < 2:
            return [
                "Define AI north star, appoint accountable leader",
                "Allocate seed budget for pilots",
                "Create basic AI strategy document",
                "Establish AI governance framework",
                "Identify initial AI use cases"
            ]
        elif score < 3:
            return [
                "Create 12–18m roadmap tied to business OKRs",
                "Institute steering committee & KPI tracking",
                "Develop AI investment strategy",
                "Create AI risk management framework",
                "Establish AI performance metrics"
            ]
        elif score < 4:
            return [
                "Scale portfolio mgmt, fund highest-ROI cases",
                "Embed AI in BU strategies",
                "Advanced AI governance and oversight",
                "Implement AI value measurement",
                "Create AI innovation programs"
            ]
        else:
            return [
                "Enterprise-wide AI operating model",
                "Continuous value realization and governance audits",
                "AI transformation leadership",
                "Advanced AI strategy and planning",
                "AI ecosystem development"
            ]
    else:
        return [
            f"Prioritize gaps by impact/effort; iterate quarterly",
            f"Focus on high-impact, low-effort improvements",
            f"Establish baseline measurements and tracking",
            f"Create improvement roadmap and timeline",
            f"Implement continuous improvement process"
        ]

# ---------------------- Helper Functions ----------------------
def export_csv(df, name):
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {name} CSV",
        data=csv,
        file_name=f"{name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

def summarize_text(text, max_sentences=5):
    if not text or len(text.strip()) < 50:
        return text
    sentences = text.split('.')
    return '. '.join(sentences[:max_sentences]) + ('.' if len(sentences) > max_sentences else '')

def maturity_band(score):
    if score < 2:
        return "**Foundational** 🌱"
    elif score < 3:
        return "**Developing** 🔄"
    elif score < 4:
        return "**Advanced** 🚀"
    else:
        return "**Optimized** ⭐"

# ---------------------- Initialize Session State ----------------------
if "onboarding" not in st.session_state:
    st.session_state.onboarding = {}

if "data_readiness_saved" not in st.session_state:
    st.session_state.data_readiness_saved = []

if "infrastructure_saved" not in st.session_state:
    st.session_state.infrastructure_saved = []

if "people_surveys" not in st.session_state:
    st.session_state.people_surveys = []

if "leadership_surveys" not in st.session_state:
    st.session_state.leadership_surveys = []

if "people_saved" not in st.session_state:
    st.session_state.people_saved = []

if "leadership_saved" not in st.session_state:
    st.session_state.leadership_saved = []

# ---------------------- Enhanced Sidebar ----------------------
with st.sidebar:
    # Professional sidebar header
    st.markdown("""
    <div class="sidebar-header">
        <h1 class="sidebar-title">🧠 AI Readiness Tool</h1>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Enterprise Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Assessment Type Selection with enhanced styling
    st.markdown("### 📋 Assessment Type")
    assessment_type = st.selectbox(
        "Choose Assessment Type",
        [
            "Basic Assessment",
            "Comprehensive Assessment", 
            "Technical Assessment",
            "ROI Assessment",
            "Security Assessment",
            "Compliance Assessment",
            "Change Management Assessment",
            "Vendor Assessment"
        ],
        index=0,
        help="Select the type of AI readiness assessment to perform"
    )
    
    # Show sub-options for Technical Assessment
    if assessment_type == "Technical Assessment":
        technical_subtype = st.selectbox(
            "Technical Assessment Type",
            [
                "Infrastructure Assessment",
                "Data Readiness Assessment",
                "API & Integration Assessment",
                "Performance Assessment"
            ],
            index=0
        )
    
    st.divider()
    
    # Show current assessment info
    if assessment_type == "Basic Assessment":
        st.success("✅ **Current Assessment: Basic Assessment**")
    elif assessment_type == "Comprehensive Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info("Advanced assessment with detailed scoring, benchmarking, and industry comparisons")
    elif assessment_type == "Technical Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info(f"Deep technical evaluation: {technical_subtype}")
    elif assessment_type == "ROI Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info("ROI analysis, cost-benefit evaluation, and investment planning")
    elif assessment_type == "Security Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info("AI security, privacy, and risk assessment")
    elif assessment_type == "Compliance Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info("Regulatory compliance and governance assessment")
    elif assessment_type == "Change Management Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info("Organizational change readiness and adoption assessment")
    elif assessment_type == "Vendor Assessment":
        st.warning("⚠️ **Coming Soon**")
        st.info("AI vendor evaluation and selection criteria")
    
    # Note: Data Management and AI Integration features are available in the main app tabs

# ---------------------- Enhanced Main Header ----------------------
st.markdown("""
<div class="main-header">
    <div class="header-content">
        <h1 class="header-title">🧠 IB Analytics</h1>
        <h2 class="header-subtitle">AI Readiness Assessment Platform</h2>
    </div>
</div>
""", unsafe_allow_html=True)

# Professional description with enhanced styling
st.markdown("""
<div class="assessment-card fade-in-up">
    <h3 style="color: #1e3c72; margin-bottom: 0.75rem; font-weight: 600;">🎯 Comprehensive AI Readiness Evaluation</h3>
    <p style="font-size: 1rem; color: #4a5568; margin-bottom: 0.5rem;">
        Assess your organization's readiness for AI implementation across key dimensions with our enterprise-grade assessment platform.
    </p>
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 3px; border-radius: 2px; margin: 0.75rem 0;"></div>
    <p style="color: #718096; font-size: 0.9rem; margin: 0;">
        <strong>Five Assessment Modules:</strong> Onboarding → Data Readiness → Infrastructure → People → Leadership & Strategy → Results
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------- Onboarding Tab ----------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Onboarding", 
    "📊 Data Readiness", 
    "⚙️ Infrastructure", 
    "👥 People", 
    "🎯 Leadership & Strategy", 
    "📈 Results"
])

with tab1:
    st.markdown("""
    <div class="assessment-card fade-in-up">
        <h2 style="color: #1e3c72; margin-bottom: 1rem; font-weight: 600; display: flex; align-items: center;">
            🏠 Onboarding Information
        </h2>
        <p style="color: #4a5568; margin-bottom: 1rem; font-size: 1rem;">
            Provide your organization's basic information to begin the AI readiness assessment process.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced form with better styling
    st.markdown('<div class="assessment-card fade-in-up">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Basic Information")
        st.session_state.onboarding["Company Name"] = st.text_input(
            "Company Name", 
            value=st.session_state.onboarding.get("Company Name", ""),
            placeholder="Enter your company name",
            help="The official name of your organization"
        )
        st.session_state.onboarding["Sector"] = st.selectbox(
            "Industry Sector", 
            [
            "Banking/Finance",
            "Insurance", 
            "Retail",
            "Manufacturing",
            "Healthcare",
            "Education",
            "Public Sector",
            "Technology",
            "Other"
            ], 
            index=0 if "Sector" not in st.session_state.onboarding else ["Banking/Finance", "Insurance", "Retail", "Manufacturing", "Healthcare", "Education", "Public Sector", "Technology", "Other"].index(st.session_state.onboarding["Sector"]),
            help="Select your primary industry sector"
        )
        st.session_state.onboarding["Phone"] = st.text_input(
            "Phone Number", 
            value=st.session_state.onboarding.get("Phone", ""),
            placeholder="+1 (555) 123-4567",
            help="Primary contact phone number"
        )
        st.session_state.onboarding["Address"] = st.text_area(
            "Business Address", 
            value=st.session_state.onboarding.get("Address", ""), 
            height=80,
            placeholder="Enter your business address",
            help="Primary business location address"
        )
    
    with col2:
        st.markdown("### 📧 Contact Information")
        st.session_state.onboarding["Email"] = st.text_input(
            "Email Address", 
            value=st.session_state.onboarding.get("Email", ""),
            placeholder="contact@company.com",
            help="Primary contact email address"
        )
        st.session_state.onboarding["Website"] = st.text_input(
            "Website (Optional)", 
            value=st.session_state.onboarding.get("Website", ""),
            placeholder="https://www.company.com",
            help="Company website URL"
        )
    
    st.markdown("### 📋 Additional Context")
    st.text_area(
        "General Notes (Optional)", 
        value=st.session_state.onboarding.get("onboarding_notes", ""), 
        key="onboarding_notes", 
        placeholder="Provide any additional context, goals, constraints, or specific areas of focus for this AI readiness assessment...",
        help="Share any relevant background information that might influence the assessment"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Save and Reset buttons
    st.markdown("""
    <div style="margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.95); border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <h4 style="color: #1e3c72; margin-bottom: 0.5rem; font-weight: 600;">💾 Save Your Progress</h4>
        <p style="color: #4a5568; margin-bottom: 1rem; font-size: 0.9rem;">Save your onboarding information to continue with the assessment later.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Onboarding Info", key="save_onboarding", use_container_width=True):
            # Update onboarding notes in session state
            if "onboarding_notes" in st.session_state:
                st.session_state.onboarding["onboarding_notes"] = st.session_state["onboarding_notes"]
            st.success("✅ Onboarding information saved successfully!")
    
    with col3:
        if st.button("🔄 Reset Form", key="reset_onboarding", use_container_width=True):
            st.session_state.onboarding = {}
            if "onboarding_notes" in st.session_state:
                del st.session_state["onboarding_notes"]
            st.rerun()

# ---------------------- Data Readiness Tab ----------------------
with tab2:
    st.markdown("""
    <div class="assessment-card fade-in-up">
        <h2 style="color: #1e3c72; margin-bottom: 1rem; font-weight: 600; display: flex; align-items: center;">
            📊 Data Readiness Assessment
        </h2>
        <p style="color: #4a5568; margin-bottom: 0.75rem; font-size: 1rem;">
            Evaluate your organization's data infrastructure, quality, governance, and AI readiness across 34 key dimensions.
        </p>
        <div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); height: 3px; border-radius: 2px; margin: 0.75rem 0;"></div>
        <p style="color: #718096; font-size: 0.9rem; margin: 0;">
            <strong>Scoring Scale:</strong> 1 = Absent/Not Implemented → 5 = Optimized/Excellent
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    data_readiness_questions = [
        "An enterprise data catalog exists listing key datasets and owners",
        ">90% of critical data is digitized (vs. paper/PDF only)",
        "Data is centralized or virtually unified (DW/lake, federated catalog)",
        "Key datasets have primary keys/relationships (customer_id, account_id)",
        "Data is timestamped with clear refresh frequencies",
        "Unstructured data (PDFs/emails/audio) is organized/OCR/transcribed",
        "Completeness targets & monitoring exist (missing value thresholds)",
        "Accuracy checks & de-duplication routines run regularly",
        "Consistency rules enforced across systems (types/codes/master data)",
        "Update SLAs are met for intended uses (dashboards/AI)",
        "Validation tests are automated; failures trigger alerts",
        "Modern data platform (warehouse/lakehouse) with scalable storage",
        "Reliable ETL/ELT pipelines (batch/stream) connect source systems",
        "Governed data APIs provide access for apps/AI services",
        "Backups, DR, and versioning are implemented & tested",
        "Cost monitoring for storage/egress/compute guides lifecycle",
        "Business & technical metadata maintained in searchable catalog",
        "Automated lineage shows flow from source to consumption",
        "Semantic layer / data contracts define metrics & schemas",
        "Docs/playbooks exist for dataset usage, refresh, ownership",
        "Data owners/stewards assigned with responsibilities",
        "RBAC/ABAC and audit logs enforced for sensitive data",
        "Sensitive data classified & protected (encryption/key mgmt)",
        "3rd-party & shadow AI data flows inventoried & governed",
        "Incident response & rollback plans exist and are tested",
        "Compliance map (GDPR/CCPA/HIPAA/KYC/AML etc.) exists",
        "Consent mgmt & subject rights implemented",
        "Retention & deletion policies defined & automated where possible",
        "Privacy impact assessments (DPIA) done for new data uses",
        "Regulatory reporting data is accurate, timely, auditable",
        "Vector DB / embeddings layer available for semi/unstructured content",
        "Pipelines for OCR/ASR ingestion, chunking, and enrichment exist",
        "Human feedback loops & labeling used for improvement",
        "Evaluation & observability for drift and model performance exist"
    ]
    
    for i, q in enumerate(data_readiness_questions):
        st.markdown(f"**{i + 1}.** {q}")
        
        # Initialize session state if not exists
        if f"data_readiness_{i}" not in st.session_state:
            st.session_state[f"data_readiness_{i}"] = 3
        if f"data_readiness_notes_{i}" not in st.session_state:
            st.session_state[f"data_readiness_notes_{i}"] = ""
        
        # Create radio button with current value
        score = st.radio(
            "Rating", 
            [1, 2, 3, 4, 5], 
            index=st.session_state[f"data_readiness_{i}"] - 1,
            key=f"data_readiness_{i}",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Create text area with current value
        notes = st.text_area(
            "Notes", 
            value=st.session_state[f"data_readiness_notes_{i}"],
            key=f"data_readiness_notes_{i}",
            placeholder="Add your observations here..."
        )
        
        st.divider()
    
    # Enhanced Save and Reset buttons
    st.markdown("""
    <div style="margin: 1rem 0; padding: 1rem; background: rgba(255,255,255,0.95); border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <h4 style="color: #1e3c72; margin-bottom: 0.5rem; font-weight: 600;">💾 Save Data Readiness Assessment</h4>
        <p style="color: #4a5568; margin-bottom: 1rem; font-size: 0.9rem;">Save your Data Readiness assessment to continue with other modules or generate reports.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("💾 Save Data Readiness", key="save_data_readiness", use_container_width=True):
            st.session_state.data_readiness_saved = []
            for i, q in enumerate(data_readiness_questions):
                score = st.session_state.get(f"data_readiness_{i}", 3)
                notes = st.session_state.get(f"data_readiness_notes_{i}", "")
                st.session_state.data_readiness_saved.append({
                    "question": q,
                    "score": score,
                    "notes": notes
                })
            st.success("✅ Data Readiness assessment saved successfully!")
    
    with col3:
        if st.button("🔄 Reset Assessment", key="reset_data_readiness", use_container_width=True):
            for i in range(len(data_readiness_questions)):
                if f"data_readiness_{i}" in st.session_state:
                    del st.session_state[f"data_readiness_{i}"]
                if f"data_readiness_notes_{i}" in st.session_state:
                    del st.session_state[f"data_readiness_notes_{i}"]
            st.rerun()

# ---------------------- Infrastructure Tab ----------------------
with tab3:
    st.header("⚙️ Infrastructure")
    
    infrastructure_questions = [
        "Sufficient compute (CPU/GPU) capacity available on-demand",
        "Cloud/hybrid foundations in place with secure networking",
        "Environment isolation & guardrails for AI experimentation",
        "Standardized pipelines for training/eval/deployment (CI/CD)",
        "Model/Prompt registry & versioning",
        "Feature store / vector store operationalized",
        "Secure APIs & connectors to embed AI in workflows",
        "Streaming/batch data pipelines for production use",
        "Eventing/queues for robust orchestration",
        "Monitoring (performance, cost, latency) and alerting exist",
        "SLOs/SLIs for AI services defined and tracked",
        "Incident response runbooks and chaos testing",
        "Secrets mgmt, KMS, egress controls for LLM/RAG patterns",
        "Access controls (RBAC/ABAC) and audit logs enforced",
        "Regular pen-tests and red-teaming for AI systems"
    ]
    
    for i, q in enumerate(infrastructure_questions):
        st.markdown(f"**{i + 1}.** {q}")
        
        # Initialize session state if not exists
        if f"infrastructure_{i}" not in st.session_state:
            st.session_state[f"infrastructure_{i}"] = 3
        if f"infrastructure_notes_{i}" not in st.session_state:
            st.session_state[f"infrastructure_notes_{i}"] = ""
        
        # Create radio button with current value
        score = st.radio(
            "Rating", 
            [1, 2, 3, 4, 5], 
            index=st.session_state[f"infrastructure_{i}"] - 1,
            key=f"infrastructure_{i}",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Create text area with current value
        notes = st.text_area(
            "Notes", 
            value=st.session_state[f"infrastructure_notes_{i}"],
            key=f"infrastructure_notes_{i}",
            placeholder="Add your observations here..."
        )
        
        st.divider()
    
    # Save and Reset buttons at the bottom
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Infrastructure", key="save_infrastructure"):
            st.session_state.infrastructure_saved = []
            for i, q in enumerate(infrastructure_questions):
                score = st.session_state.get(f"infrastructure_{i}", 3)
                notes = st.session_state.get(f"infrastructure_notes_{i}", "")
                st.session_state.infrastructure_saved.append({
                    "question": q,
                    "score": score,
                    "notes": notes
                })
            st.success("Infrastructure saved!")
    
    with col2:
        if st.button("🔄 Reset Infrastructure", key="reset_infrastructure"):
            for i in range(len(infrastructure_questions)):
                if f"infrastructure_{i}" in st.session_state:
                    del st.session_state[f"infrastructure_{i}"]
                if f"infrastructure_notes_{i}" in st.session_state:
                    del st.session_state[f"infrastructure_notes_{i}"]
            st.rerun()

# ---------------------- People Tab ----------------------
with tab4:
    st.header("👥 People")
    
    # Add new respondent survey
    with st.expander("➕ Add New Respondent Survey", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_name = st.text_input("Name", key="new_people_name")
        with col2:
            new_role = st.text_input("Role", key="new_people_role")
        with col3:
            new_type = st.selectbox("Type", ["AI Use (End Users)", "AI Build (Builders)"], key="new_people_type")
        
        if st.button("Add Respondent Survey", key="add_people_survey"):
            if new_name and new_role:
                new_survey = {
                    "name": new_name,
                    "role": new_role,
                    "type": new_type,
                    "scores": [],
                    "notes": "",
                    "saved": False
                }
                st.session_state.people_surveys.append(new_survey)
                st.success(f"Added survey for {new_name}")
                st.rerun()
            else:
                st.error("Please provide both name and role")
    
    # Current survey status
    if st.session_state.people_surveys:
        st.subheader("📊 Current Survey Status")
        total_surveys = len(st.session_state.people_surveys)
        saved_surveys = sum(1 for s in st.session_state.people_surveys if s.get("saved", False))
        st.info(f"Total Surveys: {total_surveys} | Saved: {saved_surveys} | Pending: {total_surveys - saved_surveys}")
    
    # Display existing surveys
    for i, survey in enumerate(st.session_state.people_surveys):
        with st.expander(f"📝 {survey['name']} - {survey['role']} ({survey['type']})", expanded=False):
            st.markdown(f"**Type:** {survey['type']}")
            
            if survey['type'] == "AI Use (End Users)":
                people_questions = [
                    "General AI literacy (understanding of capabilities/limits)",
                    "Have you used any AI tools at work? (1:No → 5:Daily heavy use)",
                    "How many distinct AI tools have you tried? (1:0 → 5:5+)",
                    "Average time per week using AI tools (1:<30m → 5:5h+)",
                    "Did tools measurably help productivity/quality?",
                    "Willingness to learn and adopt more AI at work"
                ]
            else:  # AI Build (Builders)
                people_questions = [
                    "Understands key terms (AI, ML, LLMs, embeddings, RAG, evals)",
                    "Can build basic prototypes (APIs, Python, notebooks)",
                    "Understands what is needed to build/ship an AI feature (data, evals, guardrails)",
                    "Familiar with vector DBs, prompt versioning, and evaluation basics",
                    "Can integrate AI into an app via secure patterns (auth, logging)",
                    "Willingness to upskill with targeted AI training"
                ]
            
            # Collect scores
            scores = []
            for j, q in enumerate(people_questions):
                st.markdown(f"**{j + 1}.** {q}")
                
                # Initialize session state if not exists
                if f"people_{i}_{j}" not in st.session_state:
                    st.session_state[f"people_{i}_{j}"] = 3
                
                # Create radio button with current value
                score = st.radio(
                    "Rating", 
                    [1, 2, 3, 4, 5], 
                    index=st.session_state[f"people_{i}_{j}"] - 1,
                    key=f"people_{i}_{j}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                scores.append(score)
            
            # Notes
            survey["notes"] = st.text_area(
                "Overall Notes", 
                value=survey.get("notes", ""),
                key=f"people_notes_{i}",
                placeholder="Add your observations here..."
            )
            
            # Save individual survey
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(f"💾 Save {survey['name']}'s Survey", key=f"save_people_{i}"):
                    survey["scores"] = scores
                    survey["saved"] = True
                    st.success(f"{survey['name']}'s survey saved!")
                    st.rerun()
            
            with col2:
                if survey.get("saved", False):
                    st.success("✅ Survey Saved")
                else:
                    st.warning("⏳ Survey Not Saved")
            
            # Remove survey
            if st.button(f"🗑️ Remove Survey", key=f"remove_people_{i}"):
                st.session_state.people_surveys.pop(i)
                st.rerun()
    
    # Save and Reset buttons at the bottom
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.people_surveys:
            if st.button("💾 Save All People Surveys", key="save_all_people"):
                st.session_state.people_saved = st.session_state.people_surveys.copy()
                st.success("All people surveys saved!")
        else:
            st.info("No surveys to save")
    
    with col2:
        if st.session_state.people_surveys:
            if st.button("🔄 Reset All People Surveys", key="reset_all_people"):
                st.session_state.people_surveys = []
                st.rerun()
        else:
            st.info("No surveys to reset")

# ---------------------- Leadership & Strategy Tab ----------------------
with tab5:
    st.header("🎯 Leadership & Strategy")
    
    # Add new leadership survey
    with st.expander("➕ Add New Leadership Survey", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_leadership_name = st.text_input("Name", key="new_leadership_name")
        with col2:
            new_leadership_role = st.text_input("Role", key="new_leadership_role")
        
        if st.button("Add Leadership Survey", key="add_leadership_survey"):
            if new_leadership_name and new_leadership_role:
                new_survey = {
                    "name": new_leadership_name,
                    "role": new_leadership_role,
                    "scores": [],
                    "notes": "",
                    "saved": False
                }
                st.session_state.leadership_surveys.append(new_survey)
                st.success(f"Added survey for {new_leadership_name}")
                st.rerun()
            else:
                st.error("Please provide both name and role")
    
    # Display existing surveys
    for i, survey in enumerate(st.session_state.leadership_surveys):
        with st.expander(f"📝 {survey['name']} - {survey['role']}", expanded=False):
            leadership_questions = [
                "AI is explicitly included in the 3–5 year corporate strategy",
                "A dedicated senior owner for AI exists (with decision rights)",
                "Annual budget is allocated for AI initiatives",
                "Leadership communicates AI vision and expected outcomes",
                "Use cases are prioritized by ROI/feasibility; KPIs tracked",
                "Long-term plan to improve operations with AI is defined"
            ]
            
            # Collect scores
            scores = []
            for j, q in enumerate(leadership_questions):
                st.markdown(f"**{j + 1}.** {q}")
                
                # Initialize session state if not exists
                if f"leadership_{i}_{j}" not in st.session_state:
                    st.session_state[f"leadership_{i}_{j}"] = 3
                
                # Create radio button
                score = st.radio(
                    "Rating", 
                    [1, 2, 3, 4, 5], 
                    index=st.session_state[f"leadership_{i}_{j}"] - 1,
                    key=f"leadership_{i}_{j}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                

                
                scores.append(score)
            
            # Notes
            survey["notes"] = st.text_area(
                "Overall Notes", 
                value=survey.get("notes", ""),
                key=f"leadership_notes_{i}",
                placeholder="Add your observations here..."
            )
            
            # Save individual survey
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(f"💾 Save {survey['name']}'s Survey", key=f"save_leadership_{i}"):
                    survey["scores"] = scores
                    survey["saved"] = True
                    st.success(f"{survey['name']}'s survey saved!")
                    st.rerun()
            
            with col2:
                if survey.get("saved", False):
                    st.success("✅ Survey Saved")
                else:
                    st.warning("⏳ Survey Not Saved")
            
            # Remove survey
            if st.button(f"🗑️ Remove Survey", key=f"remove_leadership_{i}"):
                st.session_state.leadership_surveys.pop(i)
                st.rerun()
    
    # Save and Reset buttons at the bottom
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.leadership_surveys:
            if st.button("💾 Save All Leadership Surveys", key="save_all_leadership"):
                st.session_state.leadership_saved = st.session_state.leadership_surveys.copy()
                st.success("All leadership surveys saved!")
        else:
            st.info("No surveys to save")
    
    with col2:
        if st.session_state.leadership_surveys:
            if st.button("🔄 Reset All Leadership Surveys", key="reset_all_leadership"):
                st.session_state.leadership_surveys = []
                st.rerun()
        else:
            st.info("No surveys to reset")

# ---------------------- Results Tab ----------------------
with tab6:
    st.markdown("""
    <div class="assessment-card fade-in-up">
        <h2 style="color: #1e3c72; margin-bottom: 1rem; font-weight: 600; display: flex; align-items: center;">
            📈 AI Readiness Results & Analysis
        </h2>
        <p style="color: #4a5568; margin-bottom: 0.75rem; font-size: 1rem;">
            Comprehensive assessment results with AI-powered recommendations and actionable insights.
        </p>
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 3px; border-radius: 2px; margin: 0.75rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate scores
    scores_data = []
    
    # Data Readiness
    if st.session_state.data_readiness_saved:
        data_scores = [item["score"] for item in st.session_state.data_readiness_saved]
        scores_data.append({
            "Area": "Data Readiness",
            "Avg Score": np.mean(data_scores),
            "Count": len(data_scores)
        })
    
    # Infrastructure
    if st.session_state.infrastructure_saved:
        infra_scores = [item["score"] for item in st.session_state.infrastructure_saved]
        scores_data.append({
            "Area": "Infrastructure",
            "Avg Score": np.mean(infra_scores),
            "Count": len(infra_scores)
        })
    
    # People - AI Users (check both saved and current surveys)
    ai_use_scores = []
    # Check saved surveys
    for survey in st.session_state.people_saved:
        if survey.get("scores") and survey["type"] == "AI Use (End Users)":
            ai_use_scores.extend(survey["scores"])
    # Check current surveys (not yet saved)
    for survey in st.session_state.people_surveys:
        if survey.get("scores") and survey["type"] == "AI Use (End Users)":
            ai_use_scores.extend(survey["scores"])
    
    if ai_use_scores:
        scores_data.append({
            "Area": "People - AI Users",
            "Avg Score": np.mean(ai_use_scores),
            "Count": len(ai_use_scores)
        })
    
    # People - AI Builders (check both saved and current surveys)
    ai_build_scores = []
    # Check saved surveys
    for survey in st.session_state.people_saved:
        if survey.get("scores") and survey["type"] == "AI Build (Builders)":
            ai_build_scores.extend(survey["scores"])
    # Check current surveys (not yet saved)
    for survey in st.session_state.people_surveys:
        if survey.get("scores") and survey["type"] == "AI Build (Builders)":
            ai_build_scores.extend(survey["scores"])
    
    if ai_build_scores:
        scores_data.append({
            "Area": "People - AI Builders",
            "Avg Score": np.mean(ai_build_scores),
            "Count": len(ai_build_scores)
        })
    
    # Leadership (check both saved and current surveys)
    leadership_scores = []
    # Check saved surveys
    for survey in st.session_state.leadership_saved:
        if survey.get("scores"):
            leadership_scores.extend(survey["scores"])
    # Check current surveys (not yet saved)
    for survey in st.session_state.leadership_surveys:
        if survey.get("scores"):
            leadership_scores.extend(survey["scores"])
    
    if leadership_scores:
        scores_data.append({
            "Area": "Leadership & Strategy",
            "Avg Score": np.mean(leadership_scores),
            "Count": len(leadership_scores)
        })
    
    # Initialize variables
    scores_df = None
    overall = 0.0
    
    if scores_data:
        scores_df = pd.DataFrame(scores_data)
        scores_df = scores_df.sort_values("Avg Score", ascending=False)
        
        # Overall score
        overall = scores_df["Avg Score"].mean()
        
        # Ultra-compact results display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); padding: 0.75rem; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                <h3 style="color: #1e3c72; margin: 0; font-size: 1.5rem; font-weight: 700;">{overall:.2f}</h3>
                <p style="color: #4a5568; margin: 0; font-size: 0.8rem;">Overall Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); padding: 0.75rem; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                <h3 style="color: #1e3c72; margin: 0; font-size: 1.5rem; font-weight: 700;">{len(scores_df)}</h3>
                <p style="color: #4a5568; margin: 0; font-size: 0.8rem;">Areas Assessed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); padding: 0.75rem; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                <h3 style="color: #1e3c72; margin: 0; font-size: 1.5rem; font-weight: 700;">{maturity_band(overall).split('**')[1].split('**')[0]}</h3>
                <p style="color: #4a5568; margin: 0; font-size: 0.8rem;">Maturity Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); padding: 0.75rem; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                <h3 style="color: #1e3c72; margin: 0; font-size: 1.5rem; font-weight: 700;">{overall/5*100:.0f}%</h3>
                <p style="color: #4a5568; margin: 0; font-size: 0.8rem;">Readiness</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Radar Chart - All 4 Dimensions Visible
        if len(scores_df) > 0:
            fig = go.Figure()
            
            # Add the main radar trace
            fig.add_trace(go.Scatterpolar(
                r=scores_df["Avg Score"].tolist(),
                theta=scores_df["Area"].tolist(),
                fill='toself',
                name='AI Readiness Score',
                fillcolor='rgba(102, 126, 234, 0.25)',
                line_color='rgba(102, 126, 234, 1)',
                line_width=3,
                marker=dict(
                    size=8,
                    color='rgba(102, 126, 234, 1)',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{theta}</b><br>Score: %{r:.2f}/5.0<extra></extra>'
            ))
            
            # Enhanced layout for better visibility
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        ticktext=['0', '1', '2', '3', '4', '5'],
                        tickvals=[0, 1, 2, 3, 4, 5],
                        tickfont=dict(size=12, color='#4a5568'),
                        gridcolor='rgba(102, 126, 234, 0.2)',
                        linecolor='rgba(102, 126, 234, 0.3)',
                        linewidth=1,
                        showline=True
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11, color='#2d3748'),
                        gridcolor='rgba(102, 126, 234, 0.15)',
                        linecolor='rgba(102, 126, 234, 0.3)',
                        linewidth=1,
                        showline=True,
                        rotation=0
                    ),
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                showlegend=False,
                height=450,
                width=500,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif")
            )
            
            # Add title and center the chart
            fig.update_layout(
                title=dict(
                    text="",
                    x=0.5,
                    y=0.98,
                    font=dict(size=14, color='#1e3c72'),
                    xanchor='center'
                )
            )
            
            # Center the chart with better proportions
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.plotly_chart(fig, use_container_width=True)
        
        # Ultra-compact AI Analysis
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Check OpenAI API status
            client, status = setup_openai()
            if client:
                st.success("✅ OpenAI API connected")
            else:
                st.warning(f"⚠️ {status}")
        
        with col2:
            if st.button("🚀 Generate AI Analysis", use_container_width=True):
                for _, row in scores_df.iterrows():
                    area = row['Area']
                    score = row['Avg Score']
                    comments = get_section_comments(area)
                    
                    # Generate AI analysis even if no comments, using score context
                    context = f"Current score: {score}/5.0. Area: {area}. No specific user comments provided."
                    ai_analysis = analyze_comments_with_ai(comments or "No specific feedback provided", area, score, context)
                    st.session_state[f"ai_analysis_{area}"] = ai_analysis
                
                st.success("✅ AI analysis generated!")
                st.rerun()
        
        # Compact AI analysis display
        for _, row in scores_df.iterrows():
            area = row['Area']
            score = row['Avg Score']
            
            with st.expander(f"🤖 {area} ({score:.2f}/5.00)", expanded=False):
                # Display AI analysis if available
                if f"ai_analysis_{area}" in st.session_state:
                    ai_analysis = st.session_state[f"ai_analysis_{area}"]
                    
                    # Priority indicator
                    priority_color = {
                        "High": "🔴",
                        "Medium": "🟡", 
                        "Low": "🟢"
                    }.get(ai_analysis.get("priority", "Medium"), "🟡")
                    
                    st.markdown(f"**{priority_color} Priority:** {ai_analysis.get('priority', 'Medium')}")
                    
                    # Recommendations (compact)
                    st.markdown("**💡 Key Recommendations:**")
                    for rec in ai_analysis.get("recommendations", []):
                        st.markdown(f"• {rec}")
                    
                    # Next Steps (compact)
                    st.markdown("**🚀 Next Steps:**")
                    for i, step in enumerate(ai_analysis.get("next_steps", [])[:3], 1):  # Show only first 3
                        st.markdown(f"{i}. {step}")
                    
                    # Regenerate button (compact)
                    if st.button(f"🔄 Regenerate", key=f"ai_regenerate_{area}"):
                        del st.session_state[f"ai_analysis_{area}"]
                        st.rerun()
                        
                else:
                    # Generate AI analysis if not already available
                    if st.button(f"🧠 Generate Analysis", key=f"ai_generate_{area}"):
                        context = f"Current score: {score}/5.0. Area: {area}. No specific user comments provided."
                        ai_analysis = analyze_comments_with_ai("No specific feedback provided", area, score, context)
                        st.session_state[f"ai_analysis_{area}"] = ai_analysis
                        st.rerun()
                    
                    # Generate basic recommendations based on score
                    basic_recs = generate_basic_recommendations(area, score)
                    st.markdown("**📊 Basic Recommendations:**")
                    for rec in basic_recs[:2]:  # Show only first 2
                        st.markdown(f"• {rec}")
    else:
        st.info("📊 No assessment data available yet. Complete surveys in other tabs to see results here.")
        st.markdown("**Start with:**")
        st.markdown("• 🏠 **Onboarding** - Enter company details")
        st.markdown("• 📊 **Data Readiness** - Assess data capabilities")
        st.markdown("• ⚙️ **Infrastructure** - Evaluate technical setup")
        st.markdown("• 👥 **People** - Survey team members")
        st.markdown("• 🎯 **Leadership & Strategy** - Assess leadership readiness")

    # Ultra-compact Exports section
    if scores_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            export_csv(scores_df, "Module Scores")
        
        with col2:
            # Markdown report
            ts = datetime.now().isoformat()
            ob = st.session_state.onboarding
            report_lines = []
            report_lines.append(f"# IB Analytics — AI Readiness Report\n")
            report_lines.append(f"**Company:** {ob.get('Company Name','(unspecified)')}  ")
            report_lines.append(f"**Sector:** {ob.get('Sector','(unspecified)')}  ")
            report_lines.append(f"**Email:** {ob.get('Email','')}  ")
            report_lines.append(f"**Phone:** {ob.get('Phone','')}  ")
            report_lines.append(f"**Date:** {ts}\n")
            report_lines.append("## Scores")
            report_lines.append(scores_df.to_markdown(index=False))
            report_lines.append(f"\n**Overall:** {overall:.2f}/5.00  \n**Maturity:** {maturity_band(overall)}\n")
            
            # Collect all comments
            comment_blob = ""
            for key, value in st.session_state.items():
                if isinstance(key, str) and key.endswith("_notes") and isinstance(value, str) and value.strip():
                    comment_blob += value.strip() + " "
            
            report_lines.append("## AI Summary of Comments")
            report_lines.append(summarize_text(comment_blob, max_sentences=8) or "_No comments provided._")
            report_lines.append("\n## AI-Powered Recommendations")
            
            for _, row in scores_df.iterrows():
                area = row['Area']
                score = row['Avg Score']
                
                # Check if AI analysis exists
                if f"ai_analysis_{area}" in st.session_state:
                    ai_analysis = st.session_state[f"ai_analysis_{area}"]
                    report_lines.append(f"\n### {area} (Score: {score:.2f}/5.00)")
                    report_lines.append(f"**Priority:** {ai_analysis.get('priority', 'Medium')}")
                    
                    report_lines.append("\n**Key Recommendations:**")
                    for rec in ai_analysis.get("recommendations", []):
                        report_lines.append(f"- {rec}")
                    
                    report_lines.append("\n**Concrete Use Cases:**")
                    for use_case in ai_analysis.get("use_cases", []):
                        report_lines.append(f"- {use_case}")
                    
                    report_lines.append("\n**Next Steps:**")
                    for i, step in enumerate(ai_analysis.get("next_steps", []), 1):
                        report_lines.append(f"{i}. {step}")
                else:
                    # Fallback to basic recommendations
                    basic_recs = generate_basic_recommendations(area, score)
                    report_lines.append(f"\n### {area} (Score: {score:.2f}/5.00)")
                    report_lines.append("**Basic Recommendations:**")
                    for rec in basic_recs:
                        report_lines.append(f"- {rec}")
            
            st.download_button(
                "📊 Download Report",
                "\n".join(report_lines).encode("utf-8"),
                file_name="ai_readiness_report.md",
                mime="text/markdown",
                use_container_width=True
            )
    else:
        st.info("📄 Complete assessments to generate downloadable reports")

