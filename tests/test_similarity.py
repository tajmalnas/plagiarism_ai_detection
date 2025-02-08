from src.similarity import calculate_similarity

def test_calculate_similarity():
    essay = "The Indian startup ecosystem faces challenges."
    dataset = [
        "Challenges in Indian startups include excessive workloads.",
        "Startups in India are booming with work culture issues."
    ]
    scores = calculate_similarity(essay, dataset)
    assert max(scores) > 0.5
