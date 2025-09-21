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

# Add custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        margin: 5px;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .reset-button > button {
        background-color: #dc3545;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        margin: 5px;
    }
    .reset-button > button:hover {
        background-color: #c82333;
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
    elif section == "People":
        for survey in st.session_state.get("people_saved", []):
            if survey.get("notes"):
                comments.append(f"{survey['name']} ({survey['type']}): {survey['notes']}")
    elif section == "Leadership & Strategy":
        for survey in st.session_state.get("leadership_saved", []):
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

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.title("🧠 AI Readiness Tool")
    
    # Assessment Type Selection
    st.subheader("📋 Assessment Type")
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
        index=0
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

# ---------------------- Main App ----------------------
st.title("🧠 IB Analytics — AI Readiness Tool")

st.markdown("Assess your organization's readiness for AI implementation across key dimensions.")
st.caption("Five modules: Onboarding → Data Readiness → Infrastructure → People → Leadership & Strategy → Results")

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
    st.header("🏠 Onboarding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.onboarding["Company Name"] = st.text_input("Company Name", value=st.session_state.onboarding.get("Company Name", ""))
        st.session_state.onboarding["Sector"] = st.selectbox("Sector", [
            "Banking/Finance",
            "Insurance", 
            "Retail",
            "Manufacturing",
            "Healthcare",
            "Education",
            "Public Sector",
            "Technology",
            "Other"
        ], index=0 if "Sector" not in st.session_state.onboarding else ["Banking/Finance", "Insurance", "Retail", "Manufacturing", "Healthcare", "Education", "Public Sector", "Technology", "Other"].index(st.session_state.onboarding["Sector"]))
        st.session_state.onboarding["Phone"] = st.text_input("Phone Number", value=st.session_state.onboarding.get("Phone", ""))
    
    with col2:
        st.session_state.onboarding["Email"] = st.text_input("Email", value=st.session_state.onboarding.get("Email", ""))
        st.session_state.onboarding["Address"] = st.text_area("Address", value=st.session_state.onboarding.get("Address", ""), height=80)
        st.session_state.onboarding["Website"] = st.text_input("Website (optional)", value=st.session_state.onboarding.get("Website", ""))
    
    st.text_area("General Notes (optional)", value=st.session_state.onboarding.get("onboarding_notes", ""), key="onboarding_notes", placeholder="Context, goals, constraints…")
    
    # Save and Reset buttons
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Onboarding Info", key="save_onboarding"):
            # Update onboarding notes in session state
            if "onboarding_notes" in st.session_state:
                st.session_state.onboarding["onboarding_notes"] = st.session_state["onboarding_notes"]
            st.success("Onboarding information saved!")
    
    with col2:
        if st.button("🔄 Reset Onboarding", key="reset_onboarding"):
            st.session_state.onboarding = {}
            if "onboarding_notes" in st.session_state:
                del st.session_state["onboarding_notes"]
            st.rerun()

# ---------------------- Data Readiness Tab ----------------------
with tab2:
    st.header("📊 Data Readiness")
    
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
    
    # Save and Reset buttons at the bottom
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Data Readiness", key="save_data_readiness"):
            st.session_state.data_readiness_saved = []
            for i, q in enumerate(data_readiness_questions):
                score = st.session_state.get(f"data_readiness_{i}", 3)
                notes = st.session_state.get(f"data_readiness_notes_{i}", "")
                st.session_state.data_readiness_saved.append({
                    "question": q,
                    "score": score,
                    "notes": notes
                })
            st.success("Data Readiness saved!")
    
    with col2:
        if st.button("🔄 Reset Data Readiness", key="reset_data_readiness"):
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
    st.header("📈 Results")
    
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
    
    # People - AI Users
    ai_use_scores = []
    for survey in st.session_state.people_saved:
        if survey.get("scores") and survey["type"] == "AI Use (End Users)":
            ai_use_scores.extend(survey["scores"])
    
    if ai_use_scores:
        scores_data.append({
            "Area": "People - AI Users",
            "Avg Score": np.mean(ai_use_scores),
            "Count": len(ai_use_scores)
        })
    
    # People - AI Builders
    ai_build_scores = []
    for survey in st.session_state.people_saved:
        if survey.get("scores") and survey["type"] == "AI Build (Builders)":
            ai_build_scores.extend(survey["scores"])
    
    if ai_build_scores:
        scores_data.append({
            "Area": "People - AI Builders",
            "Avg Score": np.mean(ai_build_scores),
            "Count": len(ai_build_scores)
        })
    
    # Leadership
    leadership_scores = []
    for survey in st.session_state.leadership_saved:
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
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Overall AI Readiness Score: {overall:.2f}/5.00**")
            st.markdown(f"**Maturity Level:** {maturity_band(overall)}")
        
        with col2:
            st.metric("Total Areas Assessed", len(scores_df))
            st.metric("Average Score", f"{overall:.2f}/5.0")
        
        # Radar chart
        if len(scores_df) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=scores_df["Avg Score"].tolist(),
                theta=scores_df["Area"].tolist(),
                fill='toself',
                name='AI Readiness Score',
                fillcolor='rgba(0, 123, 255, 0.3)',
                line_color='rgba(0, 123, 255, 0.8)',
                line_width=2
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        ticktext=['0', '1', '2', '3', '4', '5'],
                        tickvals=[0, 1, 2, 3, 4, 5],
                        tickfont=dict(size=12),
                        gridcolor='lightgray',
                        linecolor='gray'
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11),
                        gridcolor='lightgray'
                    )
                ),
                showlegend=False,
                height=600,
                width=700,
                margin=dict(l=50, r=50, t=80, b=50),
                title=dict(
                    text="AI Readiness Assessment - Radar Chart",
                    x=0.5,
                    font=dict(size=16, color='#333')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Center the chart
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.plotly_chart(fig, use_container_width=True)
        
        # AI-Powered Recommendations
        st.subheader("🤖 AI-Powered Recommendations & Use Cases")
        
        # Check OpenAI API status
        client, status = setup_openai()
        if client:
            st.success("✅ OpenAI API connected - AI analysis available")
        else:
            st.warning(f"⚠️ {status} - Using basic recommendations")
        
        # Generate AI analysis for all areas
        if st.button("🚀 Generate AI Analysis for All Areas"):
            for _, row in scores_df.iterrows():
                area = row['Area']
                score = row['Avg Score']
                comments = get_section_comments(area)
                
                if comments:
                    ai_analysis = analyze_comments_with_ai(comments, area, score)
                    st.session_state[f"ai_analysis_{area}"] = ai_analysis
            
            st.success("AI analysis generated for all areas!")
            st.rerun()
        
        # Display AI analysis for each area
        for _, row in scores_df.iterrows():
            area = row['Area']
            score = row['Avg Score']
            
            with st.expander(f"🤖 {area} (Score: {score:.2f}/5.00)", expanded=False):
                comments = get_section_comments(area)
                
                if comments:
                    st.markdown("**💬 User Comments:**")
                    st.info(comments)
                else:
                    st.info("No specific comments provided for this area.")
                    
                # Generate AI analysis if not already available
                if f"ai_analysis_{area}" not in st.session_state:
                    if st.button(f"🧠 Generate AI Analysis for {area}", key=f"ai_generate_{area}"):
                        context = f"Current score: {score}/5.0. Area: {area}. {'User comments: ' + comments if comments else 'No specific user comments provided.'}"
                        ai_analysis = analyze_comments_with_ai(comments or "No specific feedback provided", area, score, context)
                        st.session_state[f"ai_analysis_{area}"] = ai_analysis
                        st.rerun()
                
                # Display AI analysis if available
                if f"ai_analysis_{area}" in st.session_state:
                    ai_analysis = st.session_state[f"ai_analysis_{area}"]
                    
                    # Priority indicator
                    priority_color = {
                        "High": "🔴",
                        "Medium": "🟡", 
                        "Low": "🟢"
                    }.get(ai_analysis.get("priority", "Medium"), "🟡")
                    
                    st.markdown(f"**{priority_color} Priority Level:** {ai_analysis.get('priority', 'Medium')}")
                    
                    # Recommendations
                    st.markdown("**💡 Key Recommendations:**")
                    for rec in ai_analysis.get("recommendations", []):
                        st.markdown(f"• {rec}")
                    
                    # Use Cases
                    st.markdown("**🎯 Concrete Use Cases:**")
                    for use_case in ai_analysis.get("use_cases", []):
                        st.markdown(f"• {use_case}")
                    
                    # Next Steps
                    st.markdown("**🚀 Prioritized Next Steps:**")
                    for i, step in enumerate(ai_analysis.get("next_steps", []), 1):
                        st.markdown(f"{i}. {step}")
                    
                    # Regenerate button
                    if st.button(f"🔄 Regenerate Analysis for {area}", key=f"ai_regenerate_{area}"):
                        del st.session_state[f"ai_analysis_{area}"]
                        st.rerun()
                        
                else:
                    st.info(f"No AI analysis generated yet for {area}. Click 'Generate AI Analysis' above to get AI-powered recommendations!")
                    
                    # Generate basic recommendations based on score
                    basic_recs = generate_basic_recommendations(area, score)
                    st.markdown("**📊 Basic Recommendations (Score-based):**")
                    for rec in basic_recs:
                        st.markdown(f"• {rec}")
    else:
        st.info("📊 No assessment data available yet. Complete surveys in other tabs to see results here.")
        st.markdown("**Start with:**")
        st.markdown("• 🏠 **Onboarding** - Enter company details")
        st.markdown("• 📊 **Data Readiness** - Assess data capabilities")
        st.markdown("• ⚙️ **Infrastructure** - Evaluate technical setup")
        st.markdown("• 👥 **People** - Survey team members")
        st.markdown("• 🎯 **Leadership & Strategy** - Assess leadership readiness")

    st.divider()
    st.markdown("### Exports")
    
    if scores_df is not None:
        export_csv(scores_df, "Module Scores")
        
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
            "Download Report (Markdown)",
            "\n".join(report_lines).encode("utf-8"),
            file_name="ib_analytics_ai_readiness_report.md",
            mime="text/markdown",
        )
    else:
        st.info("📄 Complete assessments to generate downloadable reports")

