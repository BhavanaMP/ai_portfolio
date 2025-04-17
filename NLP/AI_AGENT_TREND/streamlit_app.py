# streamlit_app.py (Streamlit UI for Deployment)
import streamlit as st
from src.data_handler.scraping import WebScraper
from src.data_handler.preprocessing import TextCleaner
from src.model.sentiment_model import SentimentAnalyzer


def main():
    st.title("AI Agent Trend Sentiment Analyzer")
    url = st.text_input("Enter the URL to analyze:")

    if st.button("Analyze"):
        if url:
            scraper = WebScraper()
            cleaner = TextCleaner()
            sentiment_analyzer = SentimentAnalyzer()

            raw_text = scraper.scrape_web(url)
            if raw_text:
                cleaned_text = cleaner.clean_text(raw_text)
                processed_text = cleaner.preprocess_text(cleaned_text)
                sentiment_score = sentiment_analyzer.analyze_sentiment(processed_text)

                st.subheader("Processed Text Sample:")
                st.write(processed_text[:500])
                st.subheader("Sentiment Score:")
                st.write(sentiment_score)
            else:
                st.error("Failed to fetch content. Please check the URL.")
        else:
            st.warning("Please enter a URL.")


if __name__ == "__main__":
    # streamlit run streamlit_app.py
    main()
