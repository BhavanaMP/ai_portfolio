class AIRecommender:
    """
    AI Agent Recommender System
    """
    def recommend_agent(self, query):
        """Recommends AI agents based on user needs."""
        agents = {
            'coding': 'ChatGPT',
            'content creation': 'Claude',
            'customer support': 'Gemini'
        }
        return agents.get(query.lower(), 'Unknown AI Agent')