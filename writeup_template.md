
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original"
[image2]: ./output_images/undist_calibration1.jpg "Undistorted"
[image3]: ./test_images/test5.jpg "Road Original"
[image4]: ./output_images/undist_test5.jpg "Road Transformed"
[image5]: ./output_images/test5_sobel_x.jpg "Sobel X"
[image6]: ./output_images/test5_sobel_y.jpg "Sobel Y"
[image7]: ./output_images/test5_color_thresholds.jpg "Color Threshold"
[image8]: ./output_images/test5_gradient_magnitude.jpg "Gradient Magnitude"
[image9]: ./output_images/test5_combined_thresholds.jpg "Color Threshold"
[image10]: ./output_images/straight_lines1_undistort.jpg "Undistorted"  
[image11]: ./output_images/straight_lines1_warped.jpg "Warped"
[image12]: ./output_images/test5_final.jpg "Lane Lines"

[video11]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the .py file located in "./camera_calibration.py".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Then I saved the Camera Matrix and Distortion Coefficient to be used later using pickel

####Original
![alt text][image1]


####Undistorted
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
####Original
![alt text][image3]

####Undistorted
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 99 through 108 in `image_gradient.py`).  

Here's an example of my output for individual steps 

#### Sobel X
![alt text][image5]

#### Sobel Y
![alt text][image6]


#### Color Threshold
![alt text][image7]

#### Gradient Magnitude
![alt text][image8]

#### Combined
![alt text][image9]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 17 through 30 in the file `perspective_transform.py`.  
The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination points are calculated using offset.  
I chose the hardcode the source and destination points in the following manner:

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

### Undistorted
![alt text][image10]
### Warped
![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used Sliding Window approach to fit Lane line with 2nd order Polynomial. Code can be found at `pipeline.py` line 136 to 189.
To smoothen out edges I implemented `AverageCoefficient` logic to take average of Polynomiyal Coefficients. It's based on deque and keeps average of 20 A,B and C coefficients of
both left and right lines.

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 232 through 254 in my code in `pipeline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `pipeline.py` in the function `plot_image`.  Here is an example of my result on a test image:

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced issues with Line jumping at top corners, I fixed this by taking average of polynomial constants for both left and right lines. 
Another issue which I had faced was right lanes were not clearly recognized when brightness of roads were changing, to fix this I used multiple gradient filters  