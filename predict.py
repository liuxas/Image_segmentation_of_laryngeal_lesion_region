import cv2
import os
import keras
import numpy as np
import tensorflow as tf
from ipdb import set_trace
from keras import Input, Model, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, UpSampling2D, concatenate, core)
from PIL import Image

import util

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

my_model = keras.models.load_model("./best_model_sum.h5")

img_path = "/home/liux/文档/项目/seg_lar/test/sum1/images/"
test_img = [img_path+i for i in os.listdir(img_path)]
save_path = "/home/liux/文档/项目/seg_lar/test/result_sum/"
for  i in test_img:
    img = Image.open(i)
    img = img.resize((480,480))
    img = np.asarray(img)
    img = img.astype("float32")
    img = img/255.0+0.000000001
    img = img[np.newaxis,:,:,:]
    result = my_model.predict(img)

    result = np.argmax(result,axis=2)
    result = result.reshape(480,480,1)
    result = result.astype("float32")
    tem = cv2.threshold(result,0,255,cv2.THRESH_BINARY)
    cv2.imwrite(save_path+i.split("/")[-1],tem[1])