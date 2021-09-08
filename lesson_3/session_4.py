from scipy.cluster.vq import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


pic = './face_images/stones.jpg'


def draw(draw_img, location, title):
    plt.axis('off')
    plt.subplot(location)
    plt.title(title)
    plt.imshow(draw_img)


def Kms(infile, K=1):
    im = np.array(Image.open(infile))
    dx = im.shape[0] / 100
    dy = im.shape[1] / 100
    features = []

    for x in range(100):
        for y in range(100):
            R = np.mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 0])
            G = np.mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 1])
            B = np.mean(im[int(x * dx):int((x + 1) * dx), int(y * dy):int((y + 1) * dy), 2])
            features.append([R, G, B])
    features = np.array(features, 'f')
    centroids, variance = kmeans(features, K)
    code, distance = vq(features, centroids)
    code_im = code.reshape(100, 100)
    return np.array(Image.fromarray(code_im).resize((im.shape[1], im.shape[0])))


if __name__ == '__main__':
    plt.figure()
    draw(np.array(Image.open(pic)), 231, 'original')
    for k in range(2, 7):
        draw(Kms(pic, k), 230 + k, f'K={k}')
    plt.show()
