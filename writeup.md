## Writeup for Advanced Lane Finding Project

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test4_undistorted.jpg "Road Transformed"
[image3]: ./examples/test4_binary.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in section "1. Calibrate camera" of IPython notebook "CarND-Advanced-Lane-Lines.ipynb".

* Although the imaged chessboard has 9x6 inside corners, function "cv2.findChessboardCorners" does not find a solution with that many corners for all calibration images. Hence, I define function "findChessboardCornersIter" that first calls "cv2.findChessboardCorners" with 9x6 inside corners and, if no solution is found, starts to decrease the number of corners until a solution is found.

* I first test function "findChessboardCornersIter" with "camera_cal/calibration1.jpg".

* For the calibration proper, I define two empty lists imgPoints and objPoints. I then loop over all calibration images and call "findChessboardCornersIter" for each image. If a solution is found, I append the image corners found to imgPoints and the corresponding object points to objPoints. To simplify, I express the object points on a reference frame with XY on the chessboard plane, origin on the top-left corner and where each unit corresponds to the distance between two consecutive chessboard corners. Crucially, note that the number of points in objPoints shall be equal to the number of points in imgPoints for each calibration image, hence most objPoints elements contain 9x6 points, but some contain fewer points.

* I then compute the camera calibration and distortion coefficients by calling "cv2.calibrateCamera()" on the object and image point lists. I apply this distortion correction to all calibration images and the first test road image. The result of the distortion correction on the first chessboard image is shown below.

![alt text][image1]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

Below, I display the result of applying distortion correction to road test image "test4.jpg".
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is in sections 2.1 and 2.2 of notebook "CarND-Advanced-Lane-Lines.ipynb".

* In Section 2.1 I explore a variety of color spaces and channels on all the test images. From the grid of images displayed on the notebook, it is apparent that the S channel of the HLS space is the best to discriminate between lane lines and asphalt. The S channel of the HVS color space does a much poorer job. Low intensities in the H channel of the HLS space also correlate well with lane lines.

* In Section 2.2 I implement a variety of color and gradient thresholding functions. After testing the functions on the test images, I settled for function "combined_mask", which is the one I finally used to process the project and challenge videos. "combined_mask" uses a combination of i) color thresholding in H and S channels followed by gradient magnitude thresolding on the resulting mask, and ii) gradient direction and magnitude thresholding on the L channel of HLS. i) was sufficient to deal with the project video, but ii) was added to satisfactory handle the challenge video as well.

Below I show the binary image that results from applying"combined_mask" to "test4.jpg".

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

