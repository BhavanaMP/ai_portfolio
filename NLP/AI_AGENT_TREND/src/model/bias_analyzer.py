class BiasAnalyzer:
    """
    AI Ethics & Bias Analyzer
    """
    def check_bias(self, text):
        """Checks AI-generated responses for biases."""
        flagged_words = ['biased', 'misleading', 'stereotype']
        return any(word in text.lower() for word in flagged_words)