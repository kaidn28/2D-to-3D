import sys, argparse
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from regression import Regression

def parse_args():
    parser =argparse.ArgumentParser(description="Run chessboard corners detector")
    parser.add_argument('--image_dir', type=str, help='path to image directory', default= './dataset_29072021/')
    parser.add_argument('--data', type =str, default='./regression/data.csv')
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    reg = Regression(args)
    reg.train()
if __name__ == "__main__":
    main()

