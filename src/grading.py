import google.generativeai as genai
import os
import re
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found. Set GEMINI_API_KEY in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# Criteria with their weightage
criteria_weights = {
    "Coherence & Organization": 2.0,
    "Grammar & Syntax": 2.5,
    "Relevance to the Topic": 1.5,
    "Use of Evidence & Examples": 1.5,
    "Vocabulary & Language Variety": 1.0,
    "Critical Thinking & Argument Strength": 1.5
}

# Function to evaluate a criterion using Gemini API
def evaluate_criterion(criterion, topic, essay):
    prompt = f"""
    You are an expert essay evaluator. Assess the given essay based on "{criterion}".
    
    **Instructions:**  
    1. Assign a score out of 100.  
    2. Provide a detailed justification for the score.  
    3. Give specific feedback on what was good and what needs improvement.  

    **Essay Topic:** {topic}  
    **User Essay:** {essay}  

    **Response Format:**  
    - **Score:** [number between 0-100]  
    - **Justification:** [Why this score was given]  
    - **Strengths:** [What the essay did well]  
    - **Areas for Improvement:** [What can be improved]
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
        output = response.text.strip()

        # Debugging log to check raw response
        # print(f"\n🔍 Debug: Raw response for {criterion}:\n{output}\n{'='*50}")

        # Regex patterns for extracting data
        score_match = re.search(r"\*\*Score:\*\*\s*(\d+)", output)
        justification_match = re.search(r"\*\*Justification:\*\*\s*(.+)", output, re.DOTALL)
        strengths_match = re.search(r"\*\*Strengths:\*\*\s*(.+)", output, re.DOTALL)
        improvement_match = re.search(r"\*\*Areas for Improvement:\*\*\s*(.+)", output, re.DOTALL)

        print(f"📝 Criterion: {criterion}")
        print(f"🎯 Score: {score_match.group(1) if score_match else 'Not provided.'}")
        print(f"📚 Justification: {justification_match.group(1).strip() if justification_match else 'Not provided.'}")
        print(f"🌟 Strengths: {strengths_match.group(1).strip() if strengths_match else 'Not provided.'}")
        print(f"🚀 Areas for Improvement: {improvement_match.group(1).strip() if improvement_match else 'Not provided.'}")


        # Extract data safely
        score = int(score_match.group(1)) if score_match else 50  # Default score if missing
        justification = justification_match.group(1).strip() if justification_match else "Not provided."
        strengths = strengths_match.group(1).strip() if strengths_match else "Not provided."
        improvement = improvement_match.group(1).strip() if improvement_match else "Not provided."

    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        score, justification, strengths, improvement = 50, "Could not extract grading.", "Not available", "Not available"

    return score, justification, strengths, improvement

# Function to calculate final score with feedback
def grade_essay(topic, essay):
    total_score = 0
    feedback = {}

    for criterion, weight in criteria_weights.items():
        score, justification, strengths, improvement = evaluate_criterion(criterion, topic, essay)
        weighted_score = (score / 100) * weight
        total_score += weighted_score
        feedback[criterion] = {
            "Score": score,
            "Justification": justification,
            "Strengths": strengths,
            "Areas for Improvement": improvement
        }

    return round(total_score, 2), feedback
