import cv2 
import os
import numpy as np
from PIL import Image
from ipdb import set_trace

path = "/home/liux/文档/项目/seg_lar/train_new/new2/"
src_save_path = "/home/liux/文档/项目/seg_lar/train_new/train1/"
mask_save_path = "/home/liux/文档/项目/seg_lar/train_new/train2/"

t_src_save_path = "/home/liux/文档/项目/seg_lar/test/test1/"
t_mask_save_path = "/home/liux/文档/项目/seg_lar/test/test2/"

img_dir = [path+i for i in os.listdir(path)]
img_dir = np.asarray(img_dir)
index = [i for i in range(len(img_dir))]
np.random.shuffle(index)
# set_trace()
train = img_dir[index[0:30]]
test = img_dir[index[31:39]]


for i in train:
    img = cv2.imread(i)
    h,w,ch = img.shape
    tem = np.zeros((h,w,ch))
    cv2.imwrite(src_save_path+i.split("/")[-1],img)
    cv2.imwrite(mask_save_path+i.split("/")[-1],tem)

for i in test:
    img = cv2.imread(i)
    h,w,ch = img.shape
    tem = np.zeros((h,w,ch))
    cv2.imwrite(t_src_save_path+i.split("/")[-1],img)
    cv2.imwrite(t_mask_save_path+i.split("/")[-1],tem)

# src_path = "/home/liux/文档/项目/seg_lar/train_new/new1/"
# mask_path = "/home/liux/文档/项目/seg_lar/train_new/new2/"

# src_img = [src_path+i for i in os.listdir(src_path)]
# mask_img = [mask_path+i for i in os.listdir(mask_path)]


# index = [i for i in range(len(src_img))]
# np.random.shuffle(index)
# set_trace()

# for i ,j in zip(src_img,mask_img):
    