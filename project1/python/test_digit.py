import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import os

# Load the model architecture
layers = get_lenet(1) # change to 1 (default = 100)
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
images = []
images.append(cv2.imread('../test_data/e1.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('../test_data/e2.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('../test_data/e3.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('../test_data/e4.jpg', cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('../test_data/e5.jpg', cv2.IMREAD_GRAYSCALE))
# print('actrul: \n1, 5, 8, 9, 3')
# print('predict: ')
# images[0] = cv2.resize(images[0], (28,28))
# plt.imshow(images[0])
# plt.show()
counts = 0
for i in [1,5,8,9,3]:
    # images[i] = images[i].reshape(-1,1)
    # print(params)
    images[counts] = cv2.resize(images[counts], (28,28))
    #images[counts] = 255-images[counts]
    # images[counts] = images[counts].reshape(-1,1).reshape(1,-1)
    # if np.linalg.det(images[counts]) == 0:
    #     print('error')
    # print(images[counts].shape)
    # images[counts] = np.linalg.inv(images[counts])
    # plt.imshow(images[i],cmap='gray')
    # plt.show()

    _, P = convnet_forward(params, layers, images[counts], test=True)
    #print(P)
    predict = np.argmax(P, axis=0)
    #predicts = predicts.extend(predict)
    predict = ', '.join(map(str, predict))
    print(f'actual: {i} predict {predict}')
    #print(', '.join(map(str, predict)))
    counts += 1

