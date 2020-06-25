import tensorflow as tf
import numpy as np
import random
import math
import h5py
import cv2
import os


def time_calculate(sec):
    s = sec % 60
    m = sec // 60
    h = m // 60
    m = m % 60
    return h, m, s

def str2int_list(str):
    num_list = str.split(',')
    for idx, elem in enumerate(num_list):
        num_list[idx] = int(elem)
    return num_list

def scale2str(scale):
    scale_str = ''
    for elem in scale:
        scale_str = scale_str + str(elem) + 'x'
    return scale_str

def show_all_variables():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

''' For Load Datasets '''
def load_291(scale):
    label = []
    data = []

    for elem in scale:
        filename = '291_' + str(elem) + 'x.h5'
        file_dir = os.path.join(os.getcwd(), 'data', 'Train', filename)
    
        with h5py.File(file_dir, 'r') as f:
            label.append(list(f['label']))
            data.append(list(f['data']))
    
    data = np.concatenate([d for d in data] ,axis=0)
    label = np.concatenate([l for l in label] ,axis=0)

    # label, data = _shuffle((label, data))
    return label, data

def load_set5(scale=2, color_space='y'):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set5', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set5', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            label.append(read_img[:,:,2:3])
        else:
            label.append(read_img)
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            data.append(read_img[:,:,2:3])
        else:
            data.append(read_img)

    return _normalize(label), _normalize(data)

def load_set14(scale=2, color_space='y'):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set14', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Set14', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            label.append(read_img[:,:,2:3])
        else:
            label.append(read_img)
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            data.append(read_img[:,:,2:3])
        else:
            data.append(read_img)

    return _normalize(label), _normalize(data)

def load_b100(scale=2, color_space='y'):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'B100', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'B100', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            label.append(read_img[:,:,2:3])
        else:
            label.append(read_img)
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            data.append(read_img[:,:,2:3])
        else:
            data.append(read_img)

    return _normalize(label), _normalize(data)

def load_urban100(scale=2, color_space='y'):
    GT_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Urban100', 'gt')
    ILR_DIR = os.path.join(os.getcwd(), 'data', 'Test', 'Urban100', 'bicubic_' + str(scale) + 'x')

    label = []
    data = []

    for img in os.listdir(GT_DIR):
        IMG_PATH = os.path.join(GT_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            label.append(read_img[:,:,2:3])
        else:
            label.append(read_img)
    
    for img in os.listdir(ILR_DIR):
        IMG_PATH = os.path.join(ILR_DIR, img)
        read_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        if color_space == 'y' or color_space == 'Y':
            data.append(read_img[:,:,2:3])
        else:
            data.append(read_img)

    return _normalize(label), _normalize(data)

''' For Dataset Pre-processing'''
def create_sub_patches(x, patch_size=41, stride=41):
    (HR, LR) = x
    HR_patches = []
    LR_patches = []

    for idx in range(len(HR)):
        HR_img = HR[idx]
        LR_img = LR[idx]
        row = HR_img.shape[0]
        col = HR_img.shape[1]
        row_cnt = row//patch_size
        col_cnt = col//patch_size

        for i in range(row_cnt):
            for j in range(col_cnt):
                HR_img_crop = HR_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                LR_img_crop = LR_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                HR_patches.append(HR_img_crop)
                LR_patches.append(LR_img_crop)
            
            if col - col_cnt*patch_size > col/3:
                HR_img_crop = HR_img[i*patch_size:(i+1)*patch_size, -patch_size:]
                LR_img_crop = LR_img[i*patch_size:(i+1)*patch_size, -patch_size:]

                HR_patches.append(HR_img_crop)
                LR_patches.append(LR_img_crop)

        if row - row_cnt * patch_size > patch_size/3:
            for j in range(col_cnt):
                HR_img_crop = HR_img[-patch_size:, j*patch_size:(j+1)*patch_size]
                LR_img_crop = LR_img[-patch_size:, j*patch_size:(j+1)*patch_size]

                HR_patches.append(HR_img_crop)
                LR_patches.append(LR_img_crop)            

            if col - col_cnt*patch_size > patch_size/3:
                HR_img_crop = HR_img[-patch_size:, -patch_size:]
                LR_img_crop = LR_img[-patch_size:, -patch_size:]

                HR_patches.append(HR_img_crop)
                LR_patches.append(LR_img_crop)

    return HR_patches, LR_patches

def _shuffle(x):
    (HR, LR) = x
    
    seed = 777
    np.random.seed(seed)
    np.random.shuffle(HR)

    np.random.seed(seed)
    np.random.shuffle(LR)

    return HR, LR

def _normalize(x):
    return np.array(x) / 255.

def denormalize(x):
    x *= 255.
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return np.array(x)

''' Invert Color Channel '''
def bgr2ycrcb(img):
    cvt_img = cv2.cvtColor(x[idx], cv2.COLOR_BGR2YCrCb) 
    return cvt_img

def ycrcb2bgr(img):
    cvt_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return cvt_img

if __name__ == '__main__' :
    label, data = load_291([2])
    print(np.array(label).shape)
    print(np.array(data).shape)

    print(scale2str([2,3,4]))