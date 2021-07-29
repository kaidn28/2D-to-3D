import sys, argparse
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from calibration import Trainer

def parse_args():
    parser =argparse.ArgumentParser(description="Run chessboard corners detector")
    parser.add_argument('--image', type=str, help='path to image', default= './calibration/chessboard.jpg')
    parser.add_argument('--out_dir', type = str, help='path to save chessboard detector demo image', default = './calib_demo/')
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
    
if __name__ == "__main__":
    main()

