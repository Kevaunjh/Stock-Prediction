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
    def __init__(self, ticker="TSLA", lookback_days=30, cache_file="./../data/news_sentiment_cache.json", 
                 newsapi_key="06317fc92c2c4336ac58576138f827f8"):
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.cache_file = cache_file
        self.sentiment_data = None
        self.newsapi_key = newsapi_key 
        
        self.news_sources = [
            {"name": "reuters", "weight": 0.25},
            {"name": "bloomberg", "weight": 0.25},
            {"name": "cnbc", "weight": 0.2},
            {"name": "wsj", "weight": 0.2},
            {"name": "marketwatch", "weight": 0.1}
        ]
        
        self.sia = SentimentIntensityAnalyzer()
        
        self._load_cache()
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                processed_cache = {}
                for date_str, data in cached_data.items():
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                        processed_cache[date_obj] = data
                    except ValueError:
                        processed_cache[date_str] = data
                
                self.sentiment_data = processed_cache
                logging.info(f"Loaded sentiment cache with {len(processed_cache)} entries")
            except Exception as e:
                logging.error(f"Error loading sentiment cache: {e}")
                self.sentiment_data = {}
        else:
            self.sentiment_data = {}
    
    def _save_cache(self):
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
        articles = []
        try:
            date_str = date.strftime("%Y-%m-%d")
            next_date = (date + timedelta(days=1)).strftime("%Y-%m-%d")
            
            query = f"{self.ticker} OR Tesla"
            
            url = "https://newsapi.org/v2/everything"
            
            params = {
                'q': query,
                'from': date_str,
                'to': next_date,
                'language': 'en',
                'sortBy': 'relevancy',
                'apiKey': self.newsapi_key,
                'pageSize': 100  
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                news_data = response.json()
                
                if news_data['status'] == 'ok':
                    for article in news_data['articles']:
                        source_name = article.get('source', {}).get('name', '').lower()
                        
                        source_weight = 0.1 
                        for source in self.news_sources:
                            if source['name'].lower() in source_name:
                                source_name = source['name']  
                                source_weight = source['weight']
                                break
                        
                        article_data = {
                            'headline': article.get('title', ''),
                            'content': article.get('description', '') or article.get('content', ''),
                            'source': source_name,
                            'date': date,
                            'url': article.get('url', ''),
                            'source_weight': source_weight
                        }
                        
                        if article_data['headline'] and article_data['content']:
                            articles.append(article_data)
                
                logging.info(f"Retrieved {len(articles)} articles for {date_str}")
            else:
                logging.error(f"Failed to retrieve news. Status code: {response.status_code}")
                logging.error(f"Response: {response.text}")
        
        except Exception as e:
            logging.error(f"Error fetching news for {date}: {e}")
        

        if not articles:
            logging.warning(f"No articles found for {date}. Using fallback method.")
            articles = self._fetch_news_fallback(date)
        
        return articles
    
    def _fetch_news_fallback(self, date):
        try:

            ticker_symbol = self.ticker
            company_name = "Tesla" if self.ticker == "TSLA" else self.ticker
            
            date_str = date.strftime("%Y-%m-%d")
            fallback_articles = []
            
            search_terms = [ticker_symbol, company_name]
            
            for term in search_terms:
                search_url = f"https://www.google.com/search?q={term}&tbm=nws&tbs=cdr:1,cd_min:{date_str},cd_max:{date_str}"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(search_url, headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    news_items = soup.find_all('div', {'class': 'SoaBEf'})
                    
                    for item in news_items:
                        try:
                            headline_elem = item.find('div', {'role': 'heading'})
                            headline = headline_elem.text if headline_elem else ""
                            
                            source_elem = item.find('div', {'class': 'CEMjEf'})
                            source_name = source_elem.text.split('·')[0].strip() if source_elem else "Unknown"
                            
                            content_elem = item.find('div', {'class': 'GI74Re'})
                            content = content_elem.text if content_elem else ""
                
                            source_weight = 0.1  
                            normalized_source = "other"
                            for source in self.news_sources:
                                if source['name'].lower() in source_name.lower():
                                    normalized_source = source['name']
                                    source_weight = source['weight']
                                    break
                            
                            if not headline or not content:
                                continue
                                
                            fallback_articles.append({
                                'headline': headline,
                                'content': content,
                                'source': normalized_source,
                                'date': date,
                                'source_weight': source_weight
                            })
                        except Exception as inner_e:
                            logging.error(f"Error parsing news item: {inner_e}")
                
            logging.info(f"Fallback retrieved {len(fallback_articles)} articles for {date_str}")
            return fallback_articles
            
        except Exception as e:
            logging.error(f"Fallback news retrieval failed for {date}: {e}")
            
            logging.warning(f"Using placeholder news data for {date}")
            
            placeholder_articles = [
                {
                    'headline': f"Tesla stock update for {date.strftime('%Y-%m-%d')}",
                    'content': "No real news available for this date. This is placeholder content.",
                    'source': "placeholder",
                    'date': date,
                    'source_weight': 0.1
                }
            ]
            
            return placeholder_articles
    
    def _analyze_sentiment(self, article):
        """Analyze the sentiment of an article using VADER sentiment analyzer"""
        text = article["headline"] + " " + article["content"]
        
        sentiment = self.sia.polarity_scores(text)
        
        tesla_positive_terms = ['beat', 'exceeded', 'record', 'growth', 'profit', 'innovation',
                               'delivered', 'ahead', 'positive', 'bullish', 'upgrade']
        
        tesla_negative_terms = ['recall', 'crash', 'investigation', 'delay', 'missed',
                               'lawsuit', 'fire', 'bearish', 'downgrade', 'competition']
        
        for term in tesla_positive_terms:
            if term.lower() in text.lower():
                sentiment['compound'] = min(sentiment['compound'] + 0.05, 1.0)
                
        for term in tesla_negative_terms:
            if term.lower() in text.lower():
                sentiment['compound'] = max(sentiment['compound'] - 0.05, -1.0)
        
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
                article_sentiments = []
                
                for article in articles:
                    sentiment = self._analyze_sentiment(article)
                    source = article["source"]
                    
                    article_sentiments.append({
                        'headline': article['headline'][:50] + '...',
                        'source': source,
                        'sentiment': sentiment['compound']
                    })
                    
                    source_weight = article.get('source_weight', 0.1)
                    
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
                                        for s, d in source_sentiments.items()},
                    "articles": article_sentiments[:5]  
                }
                
                self.sentiment_data[current_date] = results[current_date]
                
                time.sleep(1)
            
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
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        ax1.set_ylabel('Stock Price ($)', color='tab:blue')
        ax1.plot(stock_data.index, stock_data['Adj Close'], color='tab:blue', label='TSLA Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')
        ax1.set_title(f'Tesla Stock Price ({stock_data.index.min().date()} to {stock_data.index.max().date()})')
        
        ax2.set_ylabel('Sentiment Score', color='tab:red')
        ax2.plot(stock_data.index, stock_data['sentiment'], color='tab:red', label='News Sentiment')
        ax2.fill_between(stock_data.index, 0, stock_data['sentiment'], 
                         where=stock_data['sentiment'] > 0, color='tab:green', alpha=0.3)
        ax2.fill_between(stock_data.index, 0, stock_data['sentiment'], 
                         where=stock_data['sentiment'] < 0, color='tab:red', alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(-1.1, 1.1)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.legend(loc='upper left')
        ax2.set_title('News Sentiment')
        
        ax3.set_ylabel('Article Count', color='tab:purple')
        ax3.bar(stock_data.index, stock_data['article_count'], color='tab:purple', label='Article Count', alpha=0.7)
        ax3.tick_params(axis='y', labelcolor='tab:purple')
        ax3.legend(loc='upper left')
        ax3.set_title('Daily Article Count')
        
        plt.tight_layout()
        debug_path = output_path.replace('.png', '_debug.png')
        plt.savefig(debug_path)
        logging.info(f"Debug chart saved to {debug_path}")
        plt.close()

def prepare_sentiment_features(data_path="./../data/stock_data.csv", 
                              output_path="./../data/stock_data_with_sentiment.csv",
                              lookback_days=60,
                              newsapi_key="06317fc92c2c4336ac58576138f827f8"):
    try:
        stock_data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        logging.info(f"Loaded stock data from {data_path}, shape: {stock_data.shape}")
        
        analyzer = NewsSentimentAnalyzer(lookback_days=lookback_days, newsapi_key=newsapi_key)
        
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
         sentiment_weight=0.3,
         newsapi_key="06317fc92c2c4336ac58576138f827f8"):
    try:
        augmented_data = prepare_sentiment_features(
            data_path=data_path,
            output_path=output_path,
            lookback_days=lookback_days,
            newsapi_key=newsapi_key
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
        print(f"Debug visualization saved to: {'./../data/sentiment_price_chart_debug.png'}")
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
    parser.add_argument("--newsapi_key", type=str, required=True,
                        help="06317fc92c2c4336ac58576138f827f8")
    
    args = parser.parse_args()
    
    main(data_path=args.data_path, 
         output_path=args.output_path,
         lookback_days=args.lookback_days,
         sentiment_weight=args.sentiment_weight,
         newsapi_key=args.newsapi_key)