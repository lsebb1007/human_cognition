import tensorflow as tf
import numpy as np
import os
import cv2

tf.set_random_seed(777)
batch_size = 100
epochs = 200
learning_rate = 0.0001

train_images = []
tlabels = []

neg_path = os.getcwd() + "/neg/neg_train"
counter = 0
for filename in os.listdir(neg_path): #이미지 사이즈 조정 
    image = cv2.imread(neg_path+"/"+filename,0)
    image = cv2.resize(image,(70, 134))
    train_images.append(image)
    tlabels.append(0)
    counter += 1

pos_path = os.getcwd() + "/pos/pos_train"

for filename in os.listdir(pos_path): #이미지 사이즈 조정
    image = cv2.imread(pos_path+"/"+filename,0)
    train_images.append(image)
    tlabels.append(1)
    counter += 1

train_images = np.array(train_images) #
train_images = train_images.reshape(counter, 9380, ) #

tlabels = np.array(tlabels)
tlabels = tlabels.reshape(counter,1)

train_labels  = np.array(np.zeros(counter*2).reshape(counter,2))