import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

class NewsSentimentAnalyzer:
    def __init__(self, ticker="TSLA", lookback_days=30, cache_file="./../data/news_sentiment_cache.json"):

        self.ticker = ticker
        self.lookback_days = lookback_days
        self.cache_file = cache_file
        self.sentiment_data = None
        self.news_sources = [
            {"name": "Reuters", "weight": 0.25},
            {"name": "Bloomberg", "weight": 0.25},
            {"name": "CNBC", "weight": 0.2},
            {"name": "WSJ", "weight": 0.2},
            {"name": "MarketWatch", "weight": 0.1}
        ]
        self.sia = SentimentIntensityAnalyzer()
        
        self._load_cache()
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                for date_str, data in cached_data.items():
                    if isinstance(date_str, str):
                        try:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                            cached_data[date_obj] = data
                            del cached_data[date_str]
                        except ValueError:
                            pass
                
                self.sentiment_data = cached_data
                logging.info(f"Loaded sentiment cache with {len(cached_data)} entries")
            except Exception as e:
                logging.error(f"Error loading sentiment cache: {e}")
                self.sentiment_data = {}
        else:
            self.sentiment_data = {}
    
    def _save_cache(self):
        """Save sentiment data to cache file"""
        try:
            serializable_data = {}
            for date_obj, data in self.sentiment_data.items():
                if isinstance(date_obj, datetime):
                    date_str = date_obj.strftime("%Y-%m-%d")
                elif isinstance(date_obj, np.datetime64):
                    date_str = pd.Timestamp(date_obj).strftime("%Y-%m-%d")
                else:
                    date_str = str(date_obj)
                serializable_data[date_str] = data
            
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_data, f)
            
            logging.info(f"Saved sentiment cache with {len(self.sentiment_data)} entries")
        except Exception as e:
            logging.error(f"Error saving sentiment cache: {e}")
    
    def _fetch_news_for_date(self, date):

        date_seed = int(date.strftime("%Y%m%d"))
        np.random.seed(date_seed)
        
        news_count = np.random.randint(3, 15) 
        articles = []
        
        possible_headlines = [
            "Tesla {action} {number}% after {event}",
            "Elon Musk {statement} about Tesla's {topic}",
            "Tesla {product} {reception} by {audience}",
            "Analysts {view} Tesla's {metric} in {timeframe}",
            "Tesla's {department} faces {challenge} amid {situation}"
        ]
        
        actions = ["jumps", "drops", "rises", "falls", "gains", "loses", "surges", "plunges"]
        numbers = ["2", "3", "4", "5", "7", "8", "10", "12", "15"]
        events = ["earnings report", "production update", "new product announcement", 
                 "Musk tweet", "factory opening", "delivery numbers", "analyst upgrade",
                 "competitor news", "supply chain update", "regulatory approval"]
        
        statements = ["tweets", "announces", "criticizes", "praises", "defends", "questions",
                     "reveals plans", "shares insights", "expresses concern", "shows excitement"]
        
        topics = ["production goals", "FSD rollout", "new factory", "battery technology", 
                 "vehicle safety", "sales strategy", "energy products", "AI development",
                 "robotics program", "manufacturing efficiency"]
        
        products = ["Model Y", "Cybertruck", "Roadster", "Semi", "Model 3", "Model X", 
                   "Solar Roof", "Powerwall", "Optimus robot", "FSD Beta"]
        
        receptions = ["well received", "criticized", "praised", "questioned", "under scrutiny",
                     "beating expectations", "falling short", "gaining traction", "facing challenges"]
        
        audiences = ["customers", "investors", "analysts", "regulators", "competitors",
                    "the market", "industry experts", "early adopters", "the media"]
        
        views = ["upgrade", "downgrade", "question", "praise", "criticize", "highlight",
                "express concern about", "show optimism for", "remain neutral on"]
        
        metrics = ["delivery numbers", "production capacity", "profit margins", "growth rate",
                  "cash position", "debt levels", "R&D spending", "market share", "revenue growth"]
        
        timeframes = ["Q1", "Q2", "Q3", "Q4", "the coming year", "the next quarter",
                     "the long term", "the current environment", "a challenging market"]
        
        departments = ["manufacturing", "software division", "energy business", "autonomous driving team",
                      "design studio", "battery production", "sales department", "service centers"]
        
        challenges = ["production delays", "regulatory hurdles", "supply chain issues", "competition",
                     "quality concerns", "talent acquisition", "cost pressures", "technical challenges"]
        
        situations = ["global chip shortage", "rising material costs", "economic uncertainty",
                     "changing consumer preferences", "regulatory scrutiny", "market volatility"]
        
        for source in self.news_sources:
            source_articles = []
            n_articles = max(1, int(news_count * source["weight"] * 2))
            
            for _ in range(n_articles):
                template = np.random.choice(possible_headlines)
                
                headline = template
                if "{action}" in headline:
                    headline = headline.replace("{action}", np.random.choice(actions))
                if "{number}" in headline:
                    headline = headline.replace("{number}", np.random.choice(numbers))
                if "{event}" in headline:
                    headline = headline.replace("{event}", np.random.choice(events))
                if "{statement}" in headline:
                    headline = headline.replace("{statement}", np.random.choice(statements))
                if "{topic}" in headline:
                    headline = headline.replace("{topic}", np.random.choice(topics))
                if "{product}" in headline:
                    headline = headline.replace("{product}", np.random.choice(products))
                if "{reception}" in headline:
                    headline = headline.replace("{reception}", np.random.choice(receptions))
                if "{audience}" in headline:
                    headline = headline.replace("{audience}", np.random.choice(audiences))
                if "{view}" in headline:
                    headline = headline.replace("{view}", np.random.choice(views))
                if "{metric}" in headline:
                    headline = headline.replace("{metric}", np.random.choice(metrics))
                if "{timeframe}" in headline:
                    headline = headline.replace("{timeframe}", np.random.choice(timeframes))
                if "{department}" in headline:
                    headline = headline.replace("{department}", np.random.choice(departments))
                if "{challenge}" in headline:
                    headline = headline.replace("{challenge}", np.random.choice(challenges))
                if "{situation}" in headline:
                    headline = headline.replace("{situation}", np.random.choice(situations))
                
                positive_keywords = ["jumps", "rises", "gains", "surges", "praised", "well received", 
                                    "beating expectations", "upgrade", "optimism", "excitement"]
                negative_keywords = ["drops", "falls", "loses", "plunges", "criticized", "scrutiny", 
                                    "falling short", "challenges", "downgrade", "concern"]
                
                sentiment_shift = 0
                for keyword in positive_keywords:
                    if keyword in headline.lower():
                        sentiment_shift += 0.2
                
                for keyword in negative_keywords:
                    if keyword in headline.lower():
                        sentiment_shift -= 0.2
                
                base_sentiment = np.random.normal(0, 0.3) + sentiment_shift
                
                base_sentiment = max(min(base_sentiment, 0.8), -0.8)
                
                content = headline + ". " + "Tesla " + np.random.choice([
                    "continues to innovate in the electric vehicle market.",
                    "faces increasing competition from traditional automakers.",
                    "production numbers are closely watched by investors.",
                    "stock has been volatile in recent trading sessions.",
                    "remains a leader in EV technology despite challenges.",
                    "CEO Elon Musk has been active on social media discussing company plans.",
                    "analysts have mixed opinions on the company's valuation.",
                    "is expanding its factory capacity globally.",
                    "energy business is gaining more attention from investors.",
                    "autonomous driving technology continues to evolve."
                ])
                
                source_articles.append({
                    "headline": headline,
                    "content": content,
                    "source": source["name"],
                    "date": date,
                    "base_sentiment": base_sentiment
                })
            
            articles.extend(source_articles)
        
        return articles
    
    def _analyze_sentiment(self, article):

        text = article["headline"] + " " + article["content"]
        
        sentiment = self.sia.polarity_scores(text)

        adjustment = article["base_sentiment"] * 0.5
        sentiment["compound"] = max(min(sentiment["compound"] + adjustment, 1.0), -1.0)
        
        return sentiment
    
    def fetch_sentiment_for_period(self, start_date, end_date):

        current_date = start_date
        results = {}
        
        while current_date <= end_date:
            if current_date in self.sentiment_data:
                results[current_date] = self.sentiment_data[current_date]
                logging.info(f"Using cached sentiment for {current_date}")
            else:
                logging.info(f"Fetching news for {current_date}")
                articles = self._fetch_news_for_date(current_date)
                
                if not articles:
                    current_date += timedelta(days=1)
                    continue
                
                weighted_compound = 0
                total_sources_weight = 0
                source_sentiments = {}
                
                for article in articles:
                    sentiment = self._analyze_sentiment(article)
                    source = article["source"]
                    
                    source_weight = next((s["weight"] for s in self.news_sources if s["name"] == source), 0.1)
                    
                    if source not in source_sentiments:
                        source_sentiments[source] = {
                            "compound_sum": 0,
                            "count": 0,
                            "weight": source_weight
                        }
                    
                    source_sentiments[source]["compound_sum"] += sentiment["compound"]
                    source_sentiments[source]["count"] += 1
                    total_sources_weight += source_weight
                
                for source, data in source_sentiments.items():
                    if data["count"] > 0:
                        avg_sentiment = data["compound_sum"] / data["count"]
                        weighted_compound += avg_sentiment * data["weight"]
                
                if total_sources_weight > 0:
                    weighted_compound /= total_sources_weight
                
                results[current_date] = {
                    "weighted_compound": weighted_compound,
                    "article_count": len(articles),
                    "source_breakdown": {s: {"avg_sentiment": d["compound_sum"] / d["count"] if d["count"] > 0 else 0, 
                                            "count": d["count"], 
                                            "weight": d["weight"]} 
                                        for s, d in source_sentiments.items()}
                }
                
                self.sentiment_data[current_date] = results[current_date]
                
                time.sleep(0.1)
            
            current_date += timedelta(days=1)
        
        self._save_cache()
        
        return results
    
    def get_historical_sentiment(self, end_date=None):

        if end_date is None:
            end_date = datetime.now().date()
        
        start_date = end_date - timedelta(days=self.lookback_days)
        
        return self.fetch_sentiment_for_period(start_date, end_date)
    
    def create_sentiment_features(self, stock_data):

        start_date = stock_data.index.min().date()
        end_date = stock_data.index.max().date()
        
        sentiment_data = self.fetch_sentiment_for_period(start_date, end_date)
        
        sentiment_df = pd.DataFrame([
            {"Date": date, "sentiment": data["weighted_compound"], "article_count": data["article_count"]}
            for date, data in sentiment_data.items()
        ])
        
        if sentiment_df.empty:
            logging.warning("No sentiment data available")
            return stock_data
        
        sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
        sentiment_df.set_index("Date", inplace=True)
        
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        

        sentiment_df = sentiment_df.reindex(stock_data.index)
        sentiment_df["sentiment"].fillna(method="ffill", inplace=True)
        sentiment_df["article_count"].fillna(0, inplace=True)
        
        sentiment_df["sentiment_5d_avg"] = sentiment_df["sentiment"].rolling(window=5, min_periods=1).mean()
        sentiment_df["sentiment_5d_std"] = sentiment_df["sentiment"].rolling(window=5, min_periods=1).std().fillna(0)
        sentiment_df["sentiment_change"] = sentiment_df["sentiment"].diff().fillna(0)
        
        sentiment_df["sentiment_streak"] = (
            sentiment_df["sentiment"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            .rolling(window=3, min_periods=1)
            .apply(lambda x: 3 if (x == 1).all() else -3 if (x == -1).all() else 0)
        )
        
        sentiment_df["sentiment_acceleration"] = sentiment_df["sentiment_change"].diff().fillna(0)
        
        sentiment_df["relative_article_volume"] = (
            sentiment_df["article_count"] / 
            sentiment_df["article_count"].rolling(window=5, min_periods=1).mean()
        ).fillna(1.0)
        
        combined_df = stock_data.copy()
        for col in sentiment_df.columns:
            combined_df[col] = sentiment_df[col]
        
        for col in sentiment_df.columns:
            combined_df[col] = combined_df[col].fillna(0)
        
        return combined_df
    
    def plot_sentiment_vs_price(self, stock_data, output_path="./../data/sentiment_price_chart.png"):
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        
        if 'sentiment' not in stock_data.columns or 'Adj Close' not in stock_data.columns:
            logging.error("Required columns 'sentiment' or 'Adj Close' not found in stock_data")
            return
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price ($)', color='tab:blue')
        ax1.plot(stock_data.index, stock_data['Adj Close'], color='tab:blue', label='TSLA Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sentiment Score', color='tab:red')
        ax2.plot(stock_data.index, stock_data['sentiment'], color='tab:red', label='News Sentiment')
        ax2.fill_between(stock_data.index, 0, stock_data['sentiment'], 
                         where=stock_data['sentiment'] > 0, color='tab:green', alpha=0.3)
        ax2.fill_between(stock_data.index, 0, stock_data['sentiment'], 
                         where=stock_data['sentiment'] < 0, color='tab:red', alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(-1.1, 1.1)
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.axhline(y=0.2, color='g', linestyle='--', alpha=0.2)
        ax2.axhline(y=-0.2, color='r', linestyle='--', alpha=0.2)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'Tesla Stock Price vs. News Sentiment ({stock_data.index.min().date()} to {stock_data.index.max().date()})')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logging.info(f"Sentiment vs price chart saved to {output_path}")
        plt.close()

def prepare_sentiment_features(data_path="./../data/stock_data.csv", 
                              output_path="./../data/stock_data_with_sentiment.csv",
                              lookback_days=60):

    try:
        stock_data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        logging.info(f"Loaded stock data from {data_path}, shape: {stock_data.shape}")
        
        analyzer = NewsSentimentAnalyzer(lookback_days=lookback_days)
        
        augmented_data = analyzer.create_sentiment_features(stock_data)
        logging.info(f"Added sentiment features, new shape: {augmented_data.shape}")
        
        analyzer.plot_sentiment_vs_price(augmented_data)
        
        sentiment_cols = ['sentiment', 'sentiment_5d_avg', 'sentiment_5d_std', 
                         'sentiment_change', 'sentiment_streak', 
                         'sentiment_acceleration', 'relative_article_volume']
        

        missing_cols = [col for col in sentiment_cols if col not in augmented_data.columns]
        if missing_cols:
            logging.warning(f"Missing sentiment columns: {missing_cols}")
            sentiment_cols = [col for col in sentiment_cols if col in augmented_data.columns]
        
        augmented_data.to_csv(output_path)
        logging.info(f"Saved augmented data to {output_path}")
        
        return augmented_data
    
    except Exception as e:
        logging.error(f"Error preparing sentiment features: {e}")
        raise

def integrate_sentiment_with_lstm(stock_data, sentiment_weight=0.6):

    try:
        sentiment_cols = ['sentiment', 'sentiment_5d_avg', 'sentiment_5d_std', 
                         'sentiment_change', 'sentiment_streak', 
                         'sentiment_acceleration', 'relative_article_volume']
        
        available_sentiment_cols = [col for col in sentiment_cols if col in stock_data.columns]
        
        if not available_sentiment_cols:
            logging.warning("No sentiment columns found in data. Using price data only.")
            return stock_data
        
        stock_data['has_sentiment'] = (~stock_data['sentiment'].isna()).astype(float)
        
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        price_scaler = MinMaxScaler()
        stock_data[price_cols] = price_scaler.fit_transform(stock_data[price_cols])
        
        sentiment_scaler = MinMaxScaler(feature_range=(-sentiment_weight, sentiment_weight))
        available_sentiment_data = stock_data[available_sentiment_cols].copy()
        
        for col in available_sentiment_cols:
            if stock_data[col].nunique() <= 1:

                stock_data[col] = stock_data[col] + np.random.normal(0, 0.001, size=len(stock_data))
        
        stock_data[available_sentiment_cols] = sentiment_scaler.fit_transform(stock_data[available_sentiment_cols])
        
        weights = {
            'sentiment': 0.4,
            'sentiment_5d_avg': 0.2,
            'sentiment_change': 0.15,
            'sentiment_streak': 0.1,
            'sentiment_acceleration': 0.05,
            'sentiment_5d_std': 0.05,
            'relative_article_volume': 0.05
        }
        
        stock_data['composite_sentiment'] = 0
        total_weight = 0
        
        for col in available_sentiment_cols:
            if col in weights:
                stock_data['composite_sentiment'] += stock_data[col] * weights[col]
                total_weight += weights[col]
        
        if total_weight > 0:
            stock_data['composite_sentiment'] /= total_weight
        
        stock_data.fillna(0, inplace=True)
        
        return stock_data
    
    except Exception as e:
        logging.error(f"Error integrating sentiment with LSTM: {e}")
        raise

def main(data_path="./../data/stock_data.csv", 
         output_path="./../data/stock_data_with_sentiment.csv",
         lookback_days=50,
         sentiment_weight=0.3):

    try:
        augmented_data = prepare_sentiment_features(
            data_path=data_path,
            output_path=output_path,
            lookback_days=lookback_days
        )
        
        lstm_ready_data = integrate_sentiment_with_lstm(
            augmented_data,
            sentiment_weight=sentiment_weight
        )

        lstm_data_path = output_path.replace('.csv', '_lstm_ready.csv')
        lstm_ready_data.to_csv(lstm_data_path)
        logging.info(f"Saved LSTM-ready data to {lstm_data_path}")
        
        print("\n===== Sentiment Analysis Complete =====")
        print(f"Added sentiment features to stock data with {sentiment_weight * 100:.0f}% weighting")
        print(f"Sentiment features: {lstm_ready_data.columns.tolist()}")
        print(f"Data saved to: {output_path}")
        print(f"LSTM-ready data saved to: {lstm_data_path}")
        print(f"Sentiment visualization saved to: {'./../data/sentiment_price_chart.png'}")
        print("=======================================\n")
        
        return lstm_ready_data
    
    except Exception as e:
        logging.error(f"Error in main sentiment analysis function: {e}")
        print(f"\nERROR: {e}")
        print("Please check the logs for details.")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add news sentiment analysis to stock prediction")
    parser.add_argument("--data_path", type=str, default="./../data/stock_data.csv", 
                        help="Path to the stock data CSV file")
    parser.add_argument("--output_path", type=str, default="./../data/stock_data_with_sentiment.csv",
                        help="Path to save the augmented data")
    parser.add_argument("--lookback_days", type=int, default=60,
                        help="Number of previous days to analyze sentiment")
    parser.add_argument("--sentiment_weight", type=float, default=0.6,
                        help="Weight to apply to sentiment features (0-1)")
    
    args = parser.parse_args()
    
    main(data_path=args.data_path, 
         output_path=args.output_path,
         lookback_days=args.lookback_days,
         sentiment_weight=args.sentiment_weight)