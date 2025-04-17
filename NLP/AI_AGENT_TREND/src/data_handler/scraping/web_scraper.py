import requests
from bs4 import BeautifulSoup
import tweepy
import json


class WebScraper:
    """
    Handles Web Scraping
    """
    def __init__(self, twitter_keys):
        self.twitter_keys = twitter_keys
        self.api = self.authenticate_twitter()

    def authenticate_twitter(self):
        auth = tweepy.OAuthHandler(self.twitter_keys["api_key"], self.twitter_keys["api_secret"])
        auth.set_access_token(self.twitter_keys["access_token"], self.twitter_keys["access_secret"])
        return tweepy.API(auth, wait_on_rate_limit=True)

    def collect_twitter_data(self, queries):
        tweets = []
        for query in queries:
            for tweet in tweepy.Cursor(self.api.search_tweets, q=query, lang="en", tweet_mode="extended").items(100):
                tweets.append(tweet.full_text)
        return tweets

    def collect_web_data(self, urls):
        texts = []
        for url in urls:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            texts.extend([p.get_text() for p in paragraphs])
        return texts

    def collect_review_data(self, review_api):
        response = requests.get(review_api)
        if response.status_code == 200:
            return json.loads(response.text)["reviews"]
        return []

    def collect_data(self, sources):
        twitter_data = self.collect_twitter_data(sources["twitter_queries"])
        web_data = self.collect_web_data(sources["websites"])
        review_data = self.collect_review_data(sources["review_api"])
        return twitter_data + web_data + review_data

    def __init__(self, twitter_api_keys):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.twitter_auth = tweepy.OAuthHandler(twitter_api_keys["api_key"], twitter_api_keys["api_secret"])
        self.twitter_auth.set_access_token(twitter_api_keys["access_token"], twitter_api_keys["access_secret"])
        self.twitter_api = tweepy.API(self.twitter_auth)

    def scrape_web(self, url):
        """Scrapes text content from a given webpage."""
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join([para.get_text() for para in paragraphs])
            return text
        return None

    def fetch_tweets(self, query, count=10):
        """Fetches latest tweets related to a query."""
        tweets = tweepy.Cursor(self.twitter_api.search_tweets, q=query, lang='en', tweet_mode='extended').items(count)
        return [tweet.full_text for tweet in tweets]

    def fetch_reviews(self, api_url):
        """Fetches customer reviews from an API endpoint."""
        response = requests.get(api_url)
        if response.status_code == 200:
            return json.loads(response.text)
        return None

    def collect_data(self, sources):
        """Collects data from multiple sources and aggregates it."""
        collected_text = []

        if "websites" in sources:
            for url in sources["websites"]:
                text = self.scrape_web(url)
                if text:
                    collected_text.append(text)

        if "twitter_queries" in sources:
            for query in sources["twitter_queries"]:
                tweets = self.fetch_tweets(query)
                collected_text.extend(tweets)

        if "review_api" in sources:
            reviews = self.fetch_reviews(sources["review_api"])
            if reviews:
                collected_text.extend(reviews)

        return collected_text
