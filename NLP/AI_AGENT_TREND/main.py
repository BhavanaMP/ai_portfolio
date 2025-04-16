from src.data_handler.scraping import WebScraper
from src.data_handler.preprocessing import TextCleaner
from src.model.sentiment_model import SentimentAnalyzer
import yaml


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    scraper = WebScraper()
    cleaner = TextCleaner()
    sentiment_analyzer = SentimentAnalyzer()

    raw_text = scraper.scrape_web(config['scrape_url'])
    if raw_text:
        cleaned_text = cleaner.clean_text(raw_text)
        processed_text = cleaner.preprocess_text(cleaned_text)
        sentiment_score = sentiment_analyzer.analyze_sentiment(processed_text)
        print("Processed Text Sample:", processed_text[:500])
        print("Sentiment Score:", sentiment_score)
