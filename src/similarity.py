from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(essay, dataset):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([essay] + dataset)
    similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])
    return similarity_scores[0]
