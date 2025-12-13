from openai import OpenAI
from dotenv import load_dotenv
from scipy.spatial import distance


load_dotenv()
client = OpenAI()


def get_articles():
    articles = [
        {"headline": "Economic Growth Continues Amid Global Uncertainty",
            "topic": "Business"},
        {"headline": "Interest rates fall to historic lows", "topic": "Business"},
        {"headline": "Scientists Make Breakthrough Discovery in Renewable Energy",
            "topic": "Science"},
        {"headline": "India Successfully Lands Near Moon's South Pole", "topic": "Science"},
        {"headline": "New Particle Discovered at CERN", "topic": "Science"},
        {"headline": "Tech Company Launches Innovative Product to Improve Online Accessibility", "topic": "Tech"},
        {"headline": "Tech Giant Buys 49% Stake In AI Startup", "topic": "Tech"},
        {"headline": "New Social Media Platform Has Everyone Talking!", "topic": "Tech"},
        {"headline": "The Blues get promoted on the final day of the season!",
            "topic": "Sport"},
        {"headline": "1.5 Billion Tune-in to the World Cup Final", "topic": "Sport"}
    ]
    return articles


def get_articles_with_keywords():
    articles = [
        {
            "headline": "Economic Growth Continues Amid Global Uncertainty",
            "topic": "Business",
            "keywords": ["economic growth", "global markets", "uncertainty"]
        },
        {
            "headline": "Interest rates fall to historic lows",
            "topic": "Business",
            "keywords": ["interest rates", "central banks", "monetary policy"]
        },
        {
            "headline": "Scientists Make Breakthrough Discovery in Renewable Energy",
            "topic": "Science",
            "keywords": ["renewable energy", "scientific breakthrough", "sustainability"]
        },
        {
            "headline": "India Successfully Lands Near Moon's South Pole",
            "topic": "Science",
            "keywords": ["space exploration", "moon landing", "ISRO"]
        },
        {
            "headline": "New Particle Discovered at CERN",
            "topic": "Science",
            "keywords": ["particle physics", "CERN", "fundamental research"]
        },
        {
            "headline": "Tech Company Launches Innovative Product to Improve Online Accessibility",
            "topic": "Tech",
            "keywords": ["accessibility", "technology innovation", "inclusive design"]
        },
        {
            "headline": "Tech Giant Buys 49% Stake In AI Startup",
            "topic": "Tech",
            "keywords": ["business", "ai"]
        },
        {
            "headline": "New Social Media Platform Has Everyone Talking!",
            "topic": "Tech",
            "keywords": ["social media", "online trends", "digital platforms"]
        },
        {
            "headline": "The Blues get promoted on the final day of the season!",
            "topic": "Sport",
            "keywords": ["football", "promotion", "league season"]
        },
        {
            "headline": "1.5 Billion Tune-in to the World Cup Final",
            "topic": "Sport",
            "keywords": ["soccer", "world cup", "tv audience"]
        }
    ]
    return articles


def embed_articles(articles):
    article_headlines = [article['headline'] for article in articles]
    articles_embeddings = create_embeddings(article_headlines)
    # articles_embeddings_dict = articles_embeddings.model_dump()

    for i, article in enumerate(articles):
        article["embedding"] = articles_embeddings[i]
    return articles


def create_embeddings(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return [data['embedding'] for data in response.model_dump()['data']]


def create_article_text(article):
    return f"""
    Headline: {article['headline']}
    Topic: {article['topic']}
    keywords: {', '.join(article['keywords'])}
    """


def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    return distances_sorted[0:n]
