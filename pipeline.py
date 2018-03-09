import numpy as np
import cv2
import matplotlib.pyplot as plt
from perspective_transform import *
from image_gradient import *
import glob
from collections import deque
from statistics import mean

first_image = True
gleft_fit = None
gright_fit = None


class AverageCoefficient:
    def __init__(self):
        self.a = deque(maxlen=20)
        self.b = deque(maxlen=20)
        self.c = deque(maxlen=20)

    def add_coefficient(self, coefficient):
        self.a.append(coefficient[0])
        self.b.append(coefficient[1])
        self.c.append(coefficient[2])

    def average(self):
        avg_coeff = [mean(self.a), mean(self.b), mean(self.c)]
        return np.array(avg_coeff)


avg_left_fit = AverageCoefficient()
avg_right_fit = AverageCoefficient()


def find_peak(binary_warped, plot=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    if plot:
        plt.plot(histogram)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    # leftx_base = np.argmax(histogram[:midpoint], axis=0)[0]
    # rightx_base = np.argmax(histogram[midpoint:], axis=0)[0] + midpoint

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return leftx_base, rightx_base


##out_img, leftx_base, rightx_base = find_peak()


def find_sliding_window(binary_warped):
    leftx_base, rightx_base = find_peak(binary_warped)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows - vertical height / nwindows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        #        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        #        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return leftx, lefty, rightx, righty, left_fit, right_fit


def warp_binary_pipeline(img):
    undist = undistort(img)
    binary = combined_thresholds(undist)
    result, M, Minv = warp(binary)
    return result, Minv


def sliding_window_pipeline(img, visualize=False):
    global first_image, gleft_fit, gright_fit

    binary_warped, Minv = warp_binary_pipeline(img)

    leftx = lefty = rightx = righty = left_fit = right_fit = None
    if first_image == True:
        leftx, lefty, rightx, righty, left_fit, right_fit = find_sliding_window(binary_warped)
        if visualize:
            visualize_image(binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit)
        gleft_fit = left_fit
        gright_fit = right_fit
        first_image = False
    else:
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (gleft_fit[0] * (nonzeroy ** 2) + gleft_fit[1] * nonzeroy +
                                       gleft_fit[2] - margin)) & (nonzerox < (gleft_fit[0] * (nonzeroy ** 2) +
                                                                              gleft_fit[1] * nonzeroy + gleft_fit[
                                                                                  2] + margin)))

        right_lane_inds = ((nonzerox > (gright_fit[0] * (nonzeroy ** 2) + gright_fit[1] * nonzeroy +
                                        gright_fit[2] - margin)) & (nonzerox < (gright_fit[0] * (nonzeroy ** 2) +
                                                                                gright_fit[1] * nonzeroy + gright_fit[
                                                                                    2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        avg_left_fit.add_coefficient(left_fit)
        avg_right_fit.add_coefficient(right_fit)

        left_fit = avg_left_fit.average()
        right_fit = avg_right_fit.average()

        ##################Visualize
        # Create an image to draw on and an image to show the selection window
        if visualize:
            visualize_image(binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit)

    return left_fit, right_fit, binary_warped, Minv,leftx,rightx


def visualize_image(binary_warped, leftx, lefty, rightx, righty, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def find_curvature(ploty, left_fit, right_fit, leftx, rightx):
    """
    Curvature of line in meter

    """

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    y_eval = np.max(ploty)
    leftx = np.array(leftx, dtype=np.float32)
    rightx = np.array(rightx, dtype=np.float32)

    left_fit_cr = np.polyfit(leftx * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(rightx * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def plot_image(undist, binary_warped, left_fit, right_fit, Minv,leftx,rightx):

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_curverad,right_curverad = find_curvature(ploty, left_fit, right_fit, leftx, rightx)

    cv2.putText(result, 'Left Radius of Curvature: %.2fm' % left_curverad, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    cv2.putText(result, 'Right Radius of Curvature: %.2fm' % right_curverad, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    return result


def image_pipeline(image):
    left_fit, right_fit, binary_warped, Minv,leftx,rightx = sliding_window_pipeline(image)
    return plot_image(image, binary_warped, left_fit, right_fit, Minv,leftx,rightx)


if __name__ == '__main__':
    # image = plt.imread('test_images/test1.jpg')
    # result = warp_binary_pipeline(image)
    #
    # out_img, leftx, lefty, rightx, righty, left_fit, right_fit = find_sliding_window(result)
    #
    # visualize(out_img, leftx, lefty, rightx, righty)

    images = glob.glob('test_images/test6*.jpg')

    for image in images:
        print(image)
        image = plt.imread(image)
        # left_fit, right_fit, binary_warped, Minv = sliding_window_pipeline(image)
        # find_curvature(left_fit)
        # plot_image(image, binary_warped, left_fit, right_fit, Minv)
        result = image_pipeline(image)
        plt.imsave('output_images/test6_final.jpg', result)
        plt.imshow(result)

    print("Done")

    # from moviepy.editor import VideoFileClip
    # from IPython.display import HTML
    #
    # white_output = 'project_video_output.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # white_clip = clip1.fl_image(image_pipeline)
