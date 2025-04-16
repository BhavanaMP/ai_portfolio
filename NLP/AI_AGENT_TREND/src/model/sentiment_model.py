from textblob import TextBlob


class SentimentAnalyzer:
    """
    AI Trend Analyzer - Sentiment Analysis
    """
    def analyze_sentiment(self, text):
        """Analyzes sentiment of the given text."""
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # Returns a value between -1 and 1