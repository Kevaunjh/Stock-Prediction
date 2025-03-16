# RunPredictions.py

import DownloadData
import LSTMTraining
import Prediction

def run_all():
    
    DownloadData.main()
    
    LSTMTraining.main()
    
    Prediction.main()

if __name__ == "__main__":
    run_all()
