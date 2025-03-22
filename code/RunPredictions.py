

import DownloadData
import LSTMTraining
import Prediction
import NewsSentiment

def run_all():
    
    DownloadData.main()
    
    NewsSentiment.main(data_path="./../data/stock_data.csv",
        output_path="./../data/stock_data_with_sentiment.csv",
    lookback_days=30, newsapi_key
    ="06317fc92c2c4336ac58576138f827f8")
    
    LSTMTraining.main(data_path="./../data/stock_data.csv",
        model_path="./../model/lstm_model.h5",
        lookback=30)
    
    Prediction.main()

if __name__ == "__main__":
    run_all()
