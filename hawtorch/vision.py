import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
import numpy as np

def imread(filename):
    """
    Read the image as it is. (Grayscale, RGB, RGBA)
    cv2.imread() needs to set FLAGS to negative.
    """
    return plt.imread(filename)
    # return cv2.imread(filename, -1)

def imsave(filename, img):
    """
    Save the image as it is.
    plt.imsave() can't save grayscale images.
    """
    cv2.imwrite(filename, img)

def _plot_point_cloud(ax, pc, axes=[0,1,2], keep_ratio=1.0, pointsize=0.05, color='k'):
    N = pc.shape[0]
    selected = np.random.choice(N, int(N*keep_ratio))
    if not isinstance(color, str):
        color = color[selected]
    ax.scatter(*(pc[selected[:,None], axes].T), s=pointsize, c=color, alpha=0.5)
    if len(axes)==3: 
        ax.view_init(50, 135)

def plot_point_cloud(pc, axes, keep_ratio=1.0, pointsize=0.05):
    fig = plt.figure(figsize=(20,10))
    if len(axes) == 3:
        ax = fig.add_subplot(111, projection='3d')
    elif len(axes) == 2:
        ax = fig.add_subplot(111)
    else:
        print("Axes should be either 2 or 3")
        exit(1)
    _plot_point_cloud(ax, pc, axes, keep_ratio=keep_ratio, pointsize=pointsize)
    plt.show()