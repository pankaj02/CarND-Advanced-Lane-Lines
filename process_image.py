import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def pipeline(img, s_thresh=(120, 255), sx_thresh=(20, 255)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    # h_channel = hls[:,:,0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold saturation channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = 255 * np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)).astype('uint8')
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) | (sxbinary == 1))] = 1
    combined_binary = 255 * np.dstack((combined_binary, combined_binary, combined_binary)).astype('uint8')
    return combined_binary, color_binary


if __name__ == '__main__':
    image = mpimg.imread('test_images/test1.jpg')

    result, color_binary = pipeline(image, s_thresh=(120, 255), sx_thresh=(20, 255))

    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=30)

    ax2.imshow(result, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    ax3.imshow(color_binary, )
    ax3.set_title('Stacked thresholds', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()

    print("Done")
