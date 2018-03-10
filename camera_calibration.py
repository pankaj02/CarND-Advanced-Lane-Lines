# Camera Calibration with OpenCV


import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

CHESSBOARD_SHAPE = (9, 6)

OUT_DIR = 'output_images/undistorted/'

CAM_CAL_P = "camera_cal/wide_dist_pickle.p"


def corners_matrix(draw_images=False):
    """
    This method takes an image and returns objpoints (3d points in real world space) and it's mapping on 2d points
    :param draw_images:
    :return:
    """

    # global objpoints, imgpoints, img, ret
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHESSBOARD_SHAPE[0] * CHESSBOARD_SHAPE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SHAPE, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            if draw_images == True:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, CHESSBOARD_SHAPE, corners, ret)
                # write_name = 'corners_found'+str(idx)+'.jpg'
                # cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    if draw_images:
        cv2.destroyAllWindows()
    return objpoints, imgpoints


# Define a function that takes an image, number of x and y points,
# camera matrix and distortion coefficients
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them! (just for fun)
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100  # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                          [img_size[0] - offset, img_size[1] - offset],
                          [offset, img_size[1] - offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M


def plot_undistortion(image, objpoints, imgpoints):
    """
    Test undistortion on an image
    :param image: Image Path example - 'camera_cal/calibration1.jpg'
    :param objpoints: 3d points in real world space
    :param imgpoints: 2d points in image plane.
    :return: None
    """
    img = cv2.imread(image)
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig("camera_caliber.jpg")


def save(image, objpoints, imgpoints):
    """
    Save undistortion on an image
    :param image: Image Path example - 'camera_cal/calibration1.jpg'
    :param objpoints: 3d points in real world space
    :param imgpoints: 2d points in image plane.
    :return: None
    """
    img = cv2.imread(image)
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    cv2.imwrite('undist_test5.jpg', undst)


def undistort(image, objpoints, imgpoints):
    """
    Takes an image and returns undistorted image
    :param image: image matrix
    :param objpoints: 3D object points
    :param imgpoints: 2D pixel points
    :return: undistorted image
    """
    img_size = (image.shape[1], image.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undst = cv2.undistort(image, mtx, dist, None, mtx)
    return undst


def save_figure(image, undistorted,image_path):
    """
    Takes an image and undistorted image and saves that
    :param image: Original Image
    :param undistorted: Undistorted Image
    :return: None
    """
    image_name = image_path.split("\\")[1].split('.')[0]
    name = "{}_{}.jpg".format(image_name, 'undistort')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=15)

    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=15)

    fig.savefig(OUT_DIR + name,bbox_inches='tight')


def dump_matrix(image, objpoints, imgpoints):
    # Save the camera calibration result for later use
    img = cv2.imread(image)
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(CAM_CAL_P, "wb"))


if __name__ == '__main__':
    # img = cv2.imread('test_images/test5.jpg')
    # for CV2 to Plt compatibility use cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    objpoints, imgpoints = corners_matrix()

    images = glob.glob('test_images/*.jpg')

    for img in images:
        image = plt.imread(img)
        undist = undistort(image, objpoints, imgpoints)
        save_figure(image, undist,img)


    # dump_matrix('camera_cal/calibration1.jpg',objpoints,imgpoints)
