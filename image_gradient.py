import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

out_dir = 'output_images/'


def sobel_x(img, threshold=(15, 255)):
    """
    Apply Sobel thresold on Horizontal Line
    :param img:
    :param threshold:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    return binary_output


def sobel_y(img, threshold=(35, 255)):
    """
    Apply Sobel thresold on Vertical Line
    :param img:
    :param threshold:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    return binary_output


def gradient_direction(img, sobel_kernel=9, thresh=(0.7, 1.1)):
    """

    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(abs_grad_dir)
    binary_output[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1
    return binary_output


def color_thresholds(img, s_thresh=(60, 255), v_thresh=(120, 255)):
    """
    
    :param img:
    :param s_thresh:
    :param v_thresh:
    :return:
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1
    return binary_output


def gradient_magnitude(img, sobel_kernel=9, mag_threshold=(60, 255)):
    """

    :param img:
    :param sobel_kernel:
    :param mag_threshold: Magnitude Threshold
    :return: Binary image with filter applied
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_threshold[0]) & (gradmag <= mag_threshold[1])] = 1
    return binary_output


def combined_thresholds(img):
    binary_x = sobel_x(img)
    binary_y = sobel_y(img)
    mag = gradient_magnitude(img)
    direct = gradient_direction(img)
    color = color_thresholds(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(img)
    binary_output[(binary_x == 1) & (binary_y == 1) & (mag == 1) | (color == 1) | (mag == 1) & (direct == 1)] = 1
    return binary_output


def plot_image(image):
    combined_binary = combined_thresholds(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)

    ax2.imshow(combined_binary, cmap='gray')
    ax2.set_title('After Thresholds', fontsize=15)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def save_images(image, image_path, type):
    image_name = image_path.split("\\")[1].split('.')[0]
    name = "{}_{}.jpg".format(image_name, type)
    plt.imsave(out_dir + name, image, cmap='gray')


if __name__ == '__main__':

    images = glob.glob('test_images/*.jpg')

    for img in images:
        image = plt.imread(img)
        mod_image = combined_thresholds(image)
        save_images(mod_image, img, 'combined_thresholds')
        # save_images(image, img)
        # plot_image(image)
