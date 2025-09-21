# AI Readiness Tool

A comprehensive Streamlit application for assessing organizational AI readiness across multiple dimensions including Data Readiness, Infrastructure, People (AI Users & Builders), and Leadership & Strategy.

## Features

- **Multi-dimensional Assessment**: Evaluate AI readiness across 5 key areas
- **Multi-respondent Surveys**: Support for multiple people from the same organization
- **AI-Powered Recommendations**: OpenAI integration for intelligent analysis and actionable insights
- **Data Persistence**: Save and load assessment data in JSON format
- **Interactive Visualizations**: Radar charts and metrics for easy interpretation
- **Export Capabilities**: Generate comprehensive reports in Markdown and CSV formats

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key** (for AI-powered recommendations):
   
   **Using Streamlit secrets (Recommended)**
   
   Create a `.streamlit/secrets.toml` file in the project root directory:
   ```
   [secrets]
   OPENAI_API_KEY = "your_actual_openai_api_key_here"
   
   # Optional: Other configuration variables
   OPENAI_MODEL = "gpt-4o"
   OPENAI_MAX_TOKENS = 1000
   OPENAI_TEMPERATURE = 0.7
   ```
   
   **Note**: The app will work without the API key but won't provide AI-powered recommendations.

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Onboarding
- Enter company information and context
- Add general notes about goals and constraints
- Save onboarding information using the save button

### 2. Data Readiness Assessment
- Evaluate data inventory, digitization, and governance
- Score each pillar from 1-5 (1=Absent, 5=Optimized)
- Add notes and evidence for each assessment

### 3. Infrastructure Assessment
- Assess compute, cloud, and security capabilities
- Evaluate environment isolation and AI experimentation readiness

### 4. People Assessment
- **AI Users (End Users)**: Assess AI literacy and tool adoption
- **AI Builders**: Evaluate development capabilities and technical skills
- Support for multiple respondents from the same organization
- Individual survey saving and management

### 5. Leadership & Strategy
- Evaluate AI strategy integration and budget allocation
- Assess leadership commitment and governance
- Support for multiple leadership perspectives

### 6. Results & AI Recommendations
- Comprehensive scoring and maturity assessment
- AI-powered analysis of user feedback
- Concrete use cases and actionable recommendations
- Export capabilities for reports and presentations

## File Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.streamlit/secrets.toml` - Streamlit secrets configuration (create this file with your API key)
- `env_example.txt` - Template for environment variables (legacy)

## Security Notes

- **Never commit your secrets.toml file** to version control
- The secrets.toml file is automatically ignored by git
- API keys are stored securely and not logged
- App works without API key but won't provide AI recommendations

## Troubleshooting

### OpenAI API Issues
- Ensure your API key is valid and has sufficient credits
- Check that the `.streamlit/secrets.toml` file is in the correct location
- Verify the API key format (starts with "sk-")
- Ensure the secrets.toml file has proper TOML format

### Dependencies Issues
- Update pip: `python -m pip install --upgrade pip`
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Streamlit Issues
- Clear cache: `streamlit cache clear`
- Check port conflicts: Use `streamlit run app.py --server.port 8502`

## Support

For issues or questions, please check:
1. Dependencies are correctly installed
2. `.streamlit/secrets.toml` file is properly configured
3. OpenAI API key is valid and active
4. No firewall or network restrictions blocking API calls
