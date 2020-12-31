import cv2
import keras
import numpy as np
import tensorflow as tf
from ipdb import set_trace
from keras import Input, Model, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D, UpSampling2D, concatenate, core)
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops, math_ops

import util

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


def custom_loss(y_true, y_pred,axis=-1):
    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    print(y_pred)
    y_pred = y_pred / math_ops.reduce_sum(y_pred, axis, True)
    epsilon = tf.convert_to_tensor(1e-07, y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon, 1. - epsilon)
    temp = y_true * math_ops.log(y_pred)
    temp = temp*[0.2,8]
    return -math_ops.reduce_sum(temp, axis)

def get_unet(patch_height,patch_width,n_ch):
    inputs = Input(shape=(patch_height,patch_width,n_ch))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    # conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    conv4 = Conv2D(256,(3,3),activation="relu",padding="same")(pool3)
    # conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256,(3,3),activation="relu",padding="same")(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3,up1],axis=3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    # conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    #
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2,up2], axis=3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    # conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
    #
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1,up3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    # conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(2, (1, 1), activation='relu',padding='same')(conv7)
    conv8 = core.Reshape((patch_height*patch_width,2))(conv8)
    ############
    conv9 = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    # adam = optimizers.Adam(lr=0.0001)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def pre_vgg16(input_shape):
    model_vgg16_conv = VGG16(weights="imagenet",include_top=False)
    # input = Input(input_shape,name="image_input")
    x = model_vgg16_conv.output
    up1 = UpSampling2D(size=(2, 2))(x)
    up1 = concatenate([model_vgg16_conv.get_layer("block5_conv3").output,up1], axis=3)
    conv1 = Conv2D(256,(3,3),activation="relu",padding="same",name="block6_conv1")(up1)
    up2 = UpSampling2D(size=(2, 2))(conv1)
    up2 = concatenate([model_vgg16_conv.get_layer("block4_conv3").output,up2], axis=3)
    conv2 = Conv2D(128,(3,3),activation="relu",padding="same",name="block6_conv2")(up2)
    up3 = UpSampling2D(size=(2, 2))(conv2)
    up3 = concatenate([model_vgg16_conv.get_layer("block3_conv3").output,up3], axis=3)
    conv3 = Conv2D(64,(3,3),activation="relu",padding="same",name="block6_conv3")(up3)
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = concatenate([model_vgg16_conv.get_layer("block2_conv2").output,up4], axis=3)
    conv4 = Conv2D(32,(3,3),activation="relu",padding="same",name="block6_conv4")(up4)
    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = concatenate([model_vgg16_conv.get_layer("block1_conv2").output,up5], axis=3)
    conv5 = Conv2D(32,(3,3),activation="relu",padding="same",name="block6_conv5")(up5)
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same',name="block_conv6")(conv5)
    conv7 = core.Reshape((480*480,2))(conv6)
    conv9 = core.Activation('softmax')(conv7)
    my_model = Model(inputs=model_vgg16_conv.input, outputs=conv9)
    # for layer in model_vgg16_conv.layers:
    #     layer.trainable = False
    print(my_model.summary())
    sgd = optimizers.SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True)
    my_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
    return my_model

my_model = pre_vgg16((480,480,3))

def generator_modify(batch_size,images_path,masks_path):
    train_generator = util.generator_train_data(images_path,masks_path)
    while True:
        for img,label in train_generator:
            label_new = np.empty((2,480,480,1))
            for i in range(batch_size):
                tem = cv2.threshold(label[i],0,1,cv2.THRESH_BINARY)
                tem = tem[1].reshape(480,480,1)
                label_new[i,:,:,:] = tem
            label_new = keras.utils.to_categorical(label_new)
            label_new = label_new.reshape(2,480*480,2)
            img = img/255.0+0.0000001
            yield (img,label_new)


my_model.fit_generator(generator=generator_modify(2,images_path="/home/liux/文档/项目/seg_lar/train_new/hh1/",masks_path="/home/liux/文档/项目/seg_lar/train_new/hh2/")
,steps_per_epoch=500,epochs=20,validation_data=generator_modify(2,images_path="/home/liux/文档/项目/seg_lar/test/hh1/",masks_path="/home/liux/文档/项目/seg_lar/test/hh2/"),
validation_steps=38,shuffle=True)

# my_model.fit_generator(generator=generator_modify(2,images_path="/home/liux/文档/项目/seg_lar/train_new/hh3/",masks_path="/home/liux/文档/项目/seg_lar/train_new/hh4/")
# ,steps_per_epoch=200,epochs=10,shuffle=True)




img = Image.open("/home/liux/文档/项目/seg_lar/nan_/7460210312972_13.jpg")
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
cv2.imwrite("./nan_/7460210312972_13.png",tem[1])


# set_trace()
