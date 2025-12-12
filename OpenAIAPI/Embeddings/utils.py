from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def get_articles():    
    articles = [
    {"headline": "Economic Growth Continues Amid Global Uncertainty", "topic": "Business"},
    {"headline": "Interest rates fall to historic lows", "topic": "Business"},
    {"headline": "Scientists Make Breakthrough Discovery in Renewable Energy", "topic": "Science"},
    {"headline": "India Successfully Lands Near Moon's South Pole", "topic": "Science"},
    {"headline": "New Particle Discovered at CERN", "topic": "Science"},
    {"headline": "Tech Company Launches Innovative Product to Improve Online Accessibility", "topic": "Tech"},
    {"headline": "Tech Giant Buys 49% Stake In AI Startup", "topic": "Tech"},
    {"headline": "New Social Media Platform Has Everyone Talking!", "topic": "Tech"},
    {"headline": "The Blues get promoted on the final day of the season!", "topic": "Sport"},
    {"headline": "1.5 Billion Tune-in to the World Cup Final", "topic": "Sport"}
    ]
    return articles


def embed_articles(articles):
    article_headlines = [article['headline'] for article in articles]
    articles_embeddings = create_embeddings(article_headlines)
    #articles_embeddings_dict = articles_embeddings.model_dump()

    for i, article in enumerate(articles):
        article["embedding"] = articles_embeddings[i] 
    return articles
    

def create_embeddings(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input= text
    )
    return [data['embedding'] for data in response.model_dump()['data']]

