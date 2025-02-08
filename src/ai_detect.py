import google.generativeai as genai
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if not GEMINI_API_KEY:
    raise ValueError("Gemini API Key not found. Set GEMINI_API_KEY in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

def generate_ai_essay(topic, num_variations=10):
    """
    Generates multiple AI-generated essays on a given topic using the Gemini API.
    """
    ai_essays = []
    model = genai.GenerativeModel("gemini-1.5-flash")  # Using Gemini Flash for faster response
    
    for _ in range(num_variations):
        response = model.generate_content(f"Write a detailed essay on the topic: {topic}")
        
        if response and hasattr(response, "text"):
            essay_text = response.text.strip()
            if essay_text:
                ai_essays.append(essay_text)
    
    return ai_essays

def check_ai_similarity(user_essay, ai_essays):
    """
    Checks if the user's essay is similar to AI-generated essays using TF-IDF cosine similarity.
    """
    if not ai_essays:
        return 0.0  # No AI-generated essays to compare

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([user_essay] + ai_essays)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    return avg_similarity

if __name__ == "__main__":
    topic = "The Impact of Artificial Intelligence on Society"
    ai_essays = generate_ai_essay(topic, num_variations=10)

    print("\nGenerated AI Essays:")
    for i, essay in enumerate(ai_essays, 1):
        print(f"\nAI Essay {i}: {essay[:500]}...")  # Show only first 500 chars
