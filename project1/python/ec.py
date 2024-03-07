import cv2
import numpy as np
from utils import get_lenet
from init_convnet import init_convnet
from scipy.io import loadmat
from load_mnist import load_mnist
from conv_net import convnet_forward
import matplotlib.pyplot as plt


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


def Manipulate_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []


    for idx, cnt in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx), (x, y - 10), font, 0.8, (0, 255, 0), 2)
        pad = 1
        digit = thresh_img[y-pad:y+h+pad, x-pad:x+w+pad]
        digit = cv2.resize(digit, (28, 28))
        # cv2.imshow('digit',digit)
        # #print(len(contours))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        digits.append(digit)

    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return digits

def recognize(digits):
    results = []
    for digit in digits:
        # digit = digit.reshape((1, 28, 28, 1))
        # digit = digit.astype('float32') / 255
        # plt.imshow(digit)
        # plt.show()
        _, P = convnet_forward(params, layers, digit, test=True)
        #print(P)
        predict = np.argmax(P, axis=0)
        results.append(predict)
        # cv2.imshow('digit', digit)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    return results

def main():
    images_path = ['../images/image1.JPG', '../images/image2.JPG', '../images/image3.png', '../images/image4.jpg']
    # result = []
    for path in images_path:
        digits = Manipulate_image(path)
        result = recognize(digits)
        result = np.reshape(result,(-1))
        result = ', '.join(map(str, result))
        print(f"Image: {path}")
        print(f"Recognized Digits: {result}")
        
if __name__ == "__main__":
    main()