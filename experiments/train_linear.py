from basicts import BasicTSLauncher

from src.model.linear import Linear, LinearConfig
from src.train.config import make_config

if __name__ == '__main__':
    BasicTSLauncher.launch_training(make_config(Linear, LinearConfig()))
