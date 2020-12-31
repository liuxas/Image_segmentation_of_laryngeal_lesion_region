import os

import cv2
import numpy as np
from ipdb import set_trace
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import keras

def get_dir(origin_path,mask_path):
    origin_image_paths = [origin_path+i for i in os.listdir(origin_path)]
    mask_image_paths = [mask_path+i for i in os.listdir(mask_path)]
    origin_image_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[0])+int(x.split("/")[-1].split("_")[1][:-4]))
    mask_image_paths.sort(key=lambda x: int(x.split("/")[-1].split("_")[0])+int(x.split("/")[-1].split("_")[1][:-4]))
    return origin_image_paths,mask_image_paths

def generator_train_data(images_path,masks_path):
    data_gen_args = dict(rotation_range=90,zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    image_generator = image_datagen.flow_from_directory(directory=images_path,class_mode=None,seed=seed,
    target_size=(480,480),batch_size=2)
    mask_generator = mask_datagen.flow_from_directory(directory=masks_path,class_mode=None,seed=seed,
    target_size=(480,480),batch_size=2,color_mode="grayscale")
    train_generator = zip(image_generator, mask_generator)
    return train_generator

# def generator_train_data(get_dir):
#     origin_image_paths,mask_image_paths = get_dir("/home/liux/文档/项目/seg_lar/train_new/images/","/home/liux/文档/项目/seg_lar/train_new/masks/")
#     while True:
#         for img_path,label_path in zip(origin_image_paths,mask_image_paths):
#             img = Image.open(img_path) 
#             label = Image.open(label_path)
#             img = img.resize((480,480))
#             label = label.resize((480,480))
#             img = np.asarray(img)
#             img = img.astype("float32")
#             label = np.asarray(label)
#             label = label.reshape(480*480,1)
#             label = keras.utils.to_categorical(label)
#             img = img/255.0
#             img = img[np.newaxis,:,:,:]
#             label = label[np.newaxis,:,:]
#             yield(img,label)

# def generator_val_data(get_dir):
#     origin_image_paths,mask_image_paths = get_dir("/home/liux/文档/项目/seg_lar/test/images/","/home/liux/文档/项目/seg_lar/test/masks/")
#     while True:
#         for img_path,label_path in zip(origin_image_paths,mask_image_paths):
#             img = Image.open(img_path) 
#             label = Image.open(label_path)
#             img = img.resize((480,480))
#             label = label.resize((480,480))
#             img = np.asarray(img)
#             img = img.astype("float32")
#             label = np.asarray(label)
#             label = label.reshape(480*480,1)
#             label = keras.utils.to_categorical(label)
#             img = img/255.0
#             img = img[np.newaxis,:,:,:]
#             label = label[np.newaxis,:,:]
#             yield(img,label)
