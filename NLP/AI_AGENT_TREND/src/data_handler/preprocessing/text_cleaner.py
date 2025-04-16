import re
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class TextCleaner:
    """
    Handles Text Preprocessing
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Cleans and preprocesses the text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def preprocess_text(self, text):
        """Tokenizes and normalizes text using spaCy."""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop])
