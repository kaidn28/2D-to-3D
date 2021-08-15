chessboard based robot calibration using hough lines and bi-linear interpolation  
  
usage:   
train chessboard corners detector:  

python tools/train.py --image <image-path> --out_dir <output-dir>  

train 2d to 3d regression: 
python tools/train_regression.py --data <<train_data_path>>  

predict:  
python tools/predict.py --image <<image-path>> --out_dir <<output-dir>> --origin <origin coordinates>
