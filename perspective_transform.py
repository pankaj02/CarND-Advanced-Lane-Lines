import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

out_dir = 'output_images/'

# load pickled distortion matrix
with open('camera_cal/wide_dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]


def warp(img, corners = np.float32([[260, 680], [580, 460], [702, 460], [1040, 680]])):
    #corners = np.float32([[190, 720], [589, 457], [698, 457], [1145, 720]])
    new_top_left = np.array([corners[0, 0], 0])
    new_top_right = np.array([corners[3, 0], 0])
    offset = [150, 0]

    img_size = (img.shape[1], img.shape[0])
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M


def undistort(img):
    result = cv2.undistort(img, mtx, dist, None, mtx)
    return result


if __name__ == '__main__':

    image = plt.imread('test_images/straight_lines1.jpg')
    # corners = np.float32([[250, 720], [589, 457], [698, 457], [1145, 720]])
    # top right, bottom right, bottom left, top left
    corners = np.float32([[260, 680], [580, 460], [702, 460], [1040, 680]])

    undist_img = cv2.undistort(image, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    f.tight_layout()

    ax1.imshow(image)
    ax1.plot(corners[0][0], corners[0][1], 'r.')  # Bottom Left
    ax1.plot(corners[1][0], corners[1][1], 'r.')  # TOP Left
    ax1.plot(corners[2][0], corners[2][1], 'r.')  # TOP Right
    ax1.plot(corners[3][0], corners[3][1], 'r.')  # Bottom Right

    ax1.set_title('Original Image', fontsize=30)

    ax2.imshow(undist_img)
    ax2.set_title('Undistorted Image', fontsize=30)
    ax2.plot(corners[0][0], corners[0][1], 'r.')  # Bottom Left
    ax2.plot(corners[1][0], corners[1][1], 'r.')  # TOP Left
    ax2.plot(corners[2][0], corners[2][1], 'r.')  # TOP Right
    ax2.plot(corners[3][0], corners[3][1], 'r.')  # Bottom Right

    plt.show()

    imshape = undist_img.shape

    corner_tuples = []
    for ind, c in enumerate(corners):
        corner_tuples.append(tuple(corners[ind]))

    cv2.line(undist_img, corner_tuples[0], corner_tuples[1], color=[255, 0, 0], thickness=1)
    cv2.line(undist_img, corner_tuples[1], corner_tuples[2], color=[255, 0, 0], thickness=1)
    cv2.line(undist_img, corner_tuples[2], corner_tuples[3], color=[255, 0, 0], thickness=1)
    cv2.line(undist_img, corner_tuples[3], corner_tuples[0], color=[255, 0, 0], thickness=1)

    warped, _ = warp(undist_img,corners)
    plt.imsave(out_dir + 'test5_undistorted.jpg', undist_img)
    plt.imsave(out_dir + 'test5_lines1_warped.jpg', warped)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original', fontsize=30)

    ax2.imshow(warped)
    ax2.set_title('Warped', fontsize=30)

    plt.show()

    print("Waiting")
