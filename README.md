chessboard based robot calibration using hough lines and bi-linear interpolation  
  
usage:   
train chessboard corners detector:  

python tools/train.py --image <image-path> --out_dir <output-dir>  

predict:  
python tools/predict.py --image <image-path> --out_dir <output-dir> --origin <origin coordinates>
