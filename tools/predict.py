import pickle 
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from calibration.Functions import distance, roundToPoint5
from calibration import Predictor
import sys
import argparse
def parse_args():
    parser =argparse.ArgumentParser(description="Run calibration")
    parser.add_argument('--image', type=str, help='path to image', default= './calibration/chessboard.jpg')
    parser.add_argument('--calib_data', type=str, help='calibration data')
    parser.add_argument('--origin', type=set, help='coordinates of origin point by(x,y)', default=(581,34))
    parser.add_argument('--out_dir', type=str, help='output images path', default='./out_dir/images/')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    predictor = Predictor(args)
    origin = (581, 34)
    real_edge = 3
    corners, appr_pixel = pickle.load(open(args.calib_data, 'rb'))
    while True:
        x,y = [float(i) for i in input().split()]
        #print(a,b)
        real_x, real_y = predictor.predict(x, y)

if __name__ == "__main__":
    main()