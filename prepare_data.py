import json
import numpy as np
import cv2
import os.path as osp
import argparse

# https://github.com/jfzhang95/PoseAug/issues/10
"""
'this file check and convert the pose data'
'-----------------'
0'Right Ankle', 3
1'Right Knee', 2
2'Right Hip', 1
3'Left Hip', 4
4'Left Knee', 5
5'Left Ankle', 6
6'Right Wrist', 15
7'Right Elbow', 14
8'Right Shoulder', 13
9'Left Shoulder', 10
10'Left Elbow', 11
11'Left Wrist', 12
12'Neck', 8 
13'Top of Head', 9
14'Pelvis)', 0
15'Thorax', 7 
16'Spine', mpi3d
17'Jaw', mpi3d
18'Head', mpi3d

mpi3dval: reorder = [14,2,1,0,3,4,5, 16,12,18,9,10,11,8,7,6]
"""

"""
openpose to 16 data

reorder = [8, 9, 10, 11, 12, 13, 14, 1-8, 1, 0, 5, 6, 7, 2, 3, 4]
"""

def normalize_screen_coordinates(X,mask, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return (X / w * 2 - [1, h / w] ) * mask

def get_2d_pose_reorderednormed(source):
    reorder = [14,2,1,0,3,4,5,16,12,18,9,10,11,8,7,6]
    tmp_array = source['kpts2d'][reorder][:, :2]
    mask = source['kpts2d'][reorder][:, 2:]
    tmp_array1 = normalize_screen_coordinates(tmp_array, mask, source['width'], source['height'])
    return tmp_array1, mask

def get_3d_pose_reordered(source):
    reorder = [14,2,1,0,3,4,5,16,12,18,9,10,11,8,7,6]
    tmp_array = source['kpts3d'][reorder][:, :3]
    mask = source['kpts3d'][reorder][:, 3:]
    return tmp_array, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--kp_json_path', default='', type=str, help='openpose keypoints json file path')
    parser.add_argument('--img_path', default='', type=str, help='image file path')

    args = parser.parse_args()

    # load the data
    openpose_path = args.kp_json_path
    img_path = args.img_path

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    with open(openpose_path) as f:
        data = json.load(f)
        data = data['people'][0]['pose_keypoints_2d']
        data_list = data
        reorder = [8, 9, 10, 11, 12, 13, 14, 25, 1, 0, 5, 6, 7, 2, 3, 4]
        data = np.array(data).reshape((-1, 3))
        point1 = data[1]
        point2 = data[8]
        spine_x = (point2[0] + point1[0]) / 2
        spine_y = (point2[1] + point1[1]) / 2
        data_list.append(spine_x)
        data_list.append(spine_y)
        data_list.append((point1[2]+point2[2])/2)
        data = np.array(data_list).reshape((-1, 3))

        print(point1, point2, (spine_x, spine_y))
        print('data', data.shape)

        joint2D = data[reorder, :2]
        mask = data[reorder, 2:]
        img_width = w
        img_height = h

        tmp_array1 = normalize_screen_coordinates(joint2D, mask, img_width, img_height)
        # print(tmp_array1)
        total_data = []
        total_data.append(tmp_array1)
        print('joint2D', tmp_array1.shape)
        print(tmp_array1)

        np.savez('./samples/'+osp.splitext(osp.basename(openpose_path))[0]+'.npz', pose2d=np.array(total_data))
        print('file saved in ' + './samples/'+osp.splitext(osp.basename(openpose_path))[0]+'.npz')