from basicts import BasicTSLauncher

from src.model.lstm import LSTM, LSTMConfig
from src.train.config import make_config

if __name__ == '__main__':
    # 10 epochs: ~2 min/epoch on CPU, time limit for the first results
    BasicTSLauncher.launch_training(make_config(LSTM, LSTMConfig(), num_epochs=10))
