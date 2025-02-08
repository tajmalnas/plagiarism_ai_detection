from src.crawl import crawl_web

if __name__ == "__main__":
    topic = "Excessive work environment in Indian startup ecosystem"
    urls = crawl_web(topic, num_results=5)
    print("Fetched URLs:")
    for url in urls:
        print(url)
