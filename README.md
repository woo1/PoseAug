# PoseAug
## PoseAug additional codes

[Original code](https://github.com/jfzhang95/PoseAug)

## Inference
### 1. Prepare data
- prepare npz data using openpose json output
`
python prepare_data.py --kp_json_path OPENPOSE_JSON_PATH --img_path IMG_PATH
`

### 2. Run
- Run inference code and save result image in ./samples
`
sh run.sh NPZ_FILE_PATH
`

