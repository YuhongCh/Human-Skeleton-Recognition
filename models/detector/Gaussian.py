import scipy.ndimage
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

def gaussian_filter(image, sigma):
    """
    Given an image, apply a Gaussian filter with the input kernel size
    and standard deviation

    Input
      image: image of size HxW
      sigma: scalar standard deviation of Gaussian Kernel

    Output
      Gaussian filtered image of size HxW
    """
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # Similar to Corner detection, use scipy's convolution function.
    # Again, be consistent with the settings (mode = 'reflect').
    coef = 1 / (np.sqrt(2 * np.pi * sigma**2))
    denominator = -1 / (2 * sigma**2)
    v1 = np.zeros((1,kernel_size))
    v2 = np.zeros((kernel_size,1))
    start = -int(kernel_size/2)
    for i in range (kernel_size):
        v1[0, i] = coef * np.exp(start**2 * denominator)
        v2[i, 0] = coef * np.exp(start**2 * denominator)
        start += 1
    kernel_gaussian = np.matmul(v2, v1)
    kernel_gaussian /= np.sum(kernel_gaussian)

    output = scipy.ndimage.convolve(image, kernel_gaussian, mode="reflect")
    return output

def find_maxima(scale_space, k_xy=5, k_s=1):
    """
    Extract the peak x,y locations from scale space

    Input
      scale_space: Scale space of size HxWxS
      k: neighborhood in x and y
      ks: neighborhood in scale

    Output
      list of (x,y) tuples; x<W and y<H
    """
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None]

    H, W, S = scale_space.shape
    maxima = []
    for i in range(H):
        for j in range(W):
            for s in range(S):
                # extracts a local neighborhood of max size
                # (2k_xy+1, 2k_xy+1, 2k_s+1)
                neighbors = scale_space[max(0, i - k_xy):min(i + k_xy + 1, H),
                                        max(0, j - k_xy):min(j + k_xy + 1, W),
                                        max(0, s - k_s):min(s + k_s + 1, S)]
                mid_pixel = scale_space[i, j, s]
                num_neighbors = np.prod(neighbors.shape) - 1
                # if mid_pixel > all the neighbors; append maxima
                if np.sum(mid_pixel < neighbors) == num_neighbors:
                    maxima.append((i, j, s))
    return maxima


def visualize_scale_space(scale_space, min_sigma, k, file_path=None):
    """
    Visualizes the scale space

    Input
      scale_space: scale space of size HxWxS
      min_sigma: the minimum sigma used
      k: the sigma multiplier
    """
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None]
    H, W, S = scale_space.shape

    # number of subplots
    p_h = int(np.floor(np.sqrt(S)))
    p_w = int(np.ceil(S / p_h))
    for i in range(S):
        plt.subplot(p_h, p_w, i + 1)
        plt.axis('off')
        plt.title('{:.1f}:{:.1f}'.format(min_sigma * k**i,
                                         min_sigma * k**(i + 1)))
        plt.imshow(scale_space[:, :, i])

    # plot or save to fig
    if file_path:
        plt.savefig(file_path, dpi=300)
    else:
        plt.show()


def visualize_maxima(image, maxima, min_sigma, k, file_path=None):
    """
    Visualizes the maxima on a given image

    Input
      image: image of size HxW
      maxima: list of (x,y) tuples; x<W, y<H
      file_path: path to save image. if None, display to screen
    Output-   None
    """
    H, W = image.shape
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    print(f"there are {len(maxima)} maximum")
    for maximum in maxima:
        y, x, s = maximum
        assert x < W and y < H and x >= 0 and y >= 0
        radius = np.sqrt(2) * min_sigma * (k**s)
        # radius = 1
        circ = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_patch(circ)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()


def main():
    image = cv2.imread("../dataset/Images/273271,1a02900084ed5ae8.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sigma_1, sigma_2 = 20, 22
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_large = gauss_2 - gauss_1  # to implement
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          'polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     'polka_large.png')

if __name__ == "__main__":
    main()