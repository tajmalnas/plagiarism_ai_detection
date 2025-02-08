import requests
import json
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pandas as pd
import textstat  # For readability scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")


def crawl_web(query, num_results=10):
    """
    Crawl the web using the Serper API for articles matching the query.
    """
    if not API_KEY:
        raise ValueError("API Key not found. Set SERPAPI_KEY in your .env file.")

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        results = [result['link'] for result in data.get('organic', [])][:num_results]
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error fetching results: {e}")
        return []


def fetch_content(url):
    """
    Fetch the main content of the webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = " ".join([para.get_text() for para in paragraphs])
        return content.strip()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ""


def calculate_features(content, all_contents):
    """
    Calculate additional features for a document.
    """
    word_count = len(content.split())
    sentence_count = len(content.split('.'))
    avg_word_length = sum(len(word) for word in content.split()) / word_count if word_count else 0
    readability = textstat.flesch_reading_ease(content)
    
    # Calculate cosine similarity with other documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([content] + all_contents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if len(similarity_scores) > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "readability": readability,
        "avg_similarity": avg_similarity,
    }


def get_docs(query, num_results=10):
    """
    Generate a list of documents with additional features.
    """
    urls = crawl_web(query, num_results)
    docs = []
    all_contents = []

    for url in urls:
        print(f"Processing {url}...")
        content = fetch_content(url)
        if content:
            all_contents.append(content)
            docs.append({"source": url, "content": content})

    # Calculate features for each document
    for doc in docs:
        features = calculate_features(doc["content"], all_contents)
        doc.update(features)

    return docs


def create_dataset(docs, filename="plagiarism_dataset.csv"):
    """
    Save the fetched documents as a CSV dataset.
    """
    df = pd.DataFrame(docs)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Dataset saved as {filename}")


if __name__ == "__main__":
    topic = "Excessive work environment in Indian startup ecosystem"
    docs = get_docs(topic, num_results=10)

    print("\nFetched Documents with Features:")
    for i, doc in enumerate(docs, start=1):
        print(f"\nDocument {i}:")
        print(f"Source: {doc['source']}")
        print(f"Content (First 500 chars): {doc['content'][:500]}...")
        print(f"Word Count: {doc['word_count']}, Readability: {doc['readability']}, Avg Similarity: {doc['avg_similarity']}")

    create_dataset(docs, filename="enhanced_plagiarism_dataset.csv")