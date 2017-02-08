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
[image3]: ./output_images/test4_binary.jpg "Binary Example"
[image4]: ./output_images/straight_lines1_warped.jpg "Warp Example"
[image5]: ./output_images/test3_fitLaneLines.jpg "Fit Visual"
[image6]: ./output_images/test3_output.jpg "Output"
[video1]: ./project_video_processed.mp4 "Video"
[video2]: ./challenge_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in section "1. Calibrate camera" of IPython notebook "CarND-Advanced-Lane-Lines.ipynb".

* Although the imaged chessboard has 9x6 inside corners, function `cv2.findChessboardCorners` does not find a solution with that many corners for all calibration images. Hence, I define function `findChessboardCornersIter` that first calls `cv2.findChessboardCorners` with 9x6 inside corners and, if no solution is found, starts to decrease the number of corners until a solution is found.

* I first test function `findChessboardCornersIter` with "camera_cal/calibration1.jpg".

* For the calibration proper, I define two empty lists imgPoints and objPoints. I then loop over all calibration images and call `findChessboardCornersIter` for each image. If a solution is found, I append the image corners found to imgPoints and the corresponding object points to objPoints. To simplify, I express the object points on a reference frame with XY on the chessboard plane, origin on the top-left corner and where each unit corresponds to the distance between two consecutive chessboard corners. Crucially, note that the number of points in objPoints shall be equal to the number of points in imgPoints for each calibration image, hence most objPoints elements contain 9x6 points, but some contain fewer points.

* I then compute the camera calibration and distortion coefficients by calling `cv2.calibrateCamera()` on the object and image point lists. I apply this distortion correction to all calibration images and the first test road image. The result of the distortion correction on the first chessboard image is shown below.

![alt text][image1]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

Below, I display the result of applying distortion correction to road test image "test4.jpg".
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is in sections 2.1 and 2.2 of notebook "CarND-Advanced-Lane-Lines.ipynb".

* In Section 2.1 I explore a variety of color spaces and channels on all the test images. From the grid of images displayed on the notebook, it is apparent that the S channel of the HLS space is the best to discriminate between lane lines and asphalt. The S channel of the HVS color space does a much poorer job. Low intensities in the H channel of the HLS space also correlate well with lane lines.

* In Section 2.2 I implement a variety of color and gradient thresholding functions. After testing the functions on the test images, I settled for function `combined_mask`, which is the one I finally used to process the project and challenge videos. `combined_mask` uses a combination of i) color thresholding in H and S channels followed by gradient magnitude thresolding on the resulting mask, and ii) gradient direction and magnitude thresholding on the L channel of HLS. i) was sufficient to deal with the project video, but ii) was added to satisfactory handle the challenge video as well.

Below I show the binary image that results from applying `combined_mask` to "test4.jpg".

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is given in section 2.3 of notebook "CarND-Advanced-Lane-Lines.ipynb".

* I manually select two points on each line lane of "straight_lines1.jpg" (source points `src`), and map them to a rectangle of base image_width/2 and height image_height (destination points `dst`), resulting in the following coordinates for the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 577,  460      | 320, 0       | 
| 196,  720      | 320, 720     |
| 1127, 720      | 960, 720     |
| 705,  460      | 960, 0       |

* I then use `cv2.getPerspectiveTransform(src, dst)` to calculate the transformation matrix from camera to "birds-eye" perspective, and `cv2.getPerspectiveTransform(dst, src)` to calculate the inverse transformation.

* To validate the perspective transform, I warp image "straight_lines1.jpg" with `cv2.warpPerspective` and the transformation matrix calcualted before, and draw the `src`and `dst` points onto the original undistored image and its warped counterpact to verify that the lines appear parallel in the warpred image. I repeat the step for the other straight line test images. Below, I copy the output of this step applied to "straight_lines1.jpg".

![alt text][image4]

* In section 2.4 I check whether "warp first, then apply binary thresholding" or "apply thresholding first, then warp" results better results. I decide to go with "apply thresholding first, then warp".

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Section 2.5.2. in the notebook contains the implementation of functions ´find_points_in_lanes_from_prevState´ and ´find_points_in_lanes_from_lostState´ to identify which points in the warped binary image belong to the left and right lane lines.

* Function ´find_points_in_lanes_from_lostState´ process the binary image in isolation from lane line point detection in previous cycles. It starts by estimating lane lines positions at the bottom of the image by searching for the absolute maximum of histograms of 40% bottom of the image at the left and right sides of the middle image column. It then slides a window of 100x100 pixels starting from the estimated lane line positions at the bottom of the image and all the way to the top of the binary image. At each step, the lane line centre is estimated to be located at the weighted average of the section of the histogram inside the window.

* Function ´find_points_in_lanes_from_prevState´ process the binary image starting from an initial estimate of the lane line x-position for each y-value. That information is kept in an object of class Lane, defined in Section 2.5.1 of the notebook. For each row of pixels, I search for points in the binary image in x-intervals of 100 pixels centred at the x-location of each lane stored in the object Lane.

Section 2.5.3. contains the implementation of functions to fit lines to the lane line points. I have implemented two different methods.

* Function ´fit_lane_points_and_calc_curvature´ fits a 2nd order polynominal f(y) = A*y^2 + B*y + C to each lane line with function ´np.polyfit´. Note that in this method the 2nd order polynomial for the left and the right lane lines are calculated independently of each other, hence lane lines may have different curvatures, i.e. be non parallel.

* Function ´fit_parall_lane_points_and_calc_curvature´ also fits 2nd order polynomials to left and right lane lines, but with the constrain that the curvature of the left lane line has to match the curvature of the right lane line for all values of y, i.e. the function finds the parameters A, B, CL and CR s.t. the quadratic models ´x = A*y^2 + B*y + CL´ (for the left line) and ´x = A*y^2 + B*y + CR´ (for right lane) best approximate input points in a lest-squares sense.

Both methods did really well in the project video. In the challange video, however, ´fit_lane_points_and_calc_curvature´ performed somewhat better, and so I went for ´fit_lane_points_and_calc_curvature´ for the final pipeline. The following image shows points identified as belonging to the left and right lane lines of "test3.jpg", together with the polynomial fits, in warped perspective.

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of curvature and the xoffset of the vehicle with respect to center is also done in Section 2.5.3 of notebook "CarND-Advanced-Lane-Lines.ipynb", in functions `fit_lane_points_and_calc_curvature` and `fit_parall_lane_points_and_calc_curvature`.

* For the radius of curvature, I first recalculate the 2nd order polynominal fits to the lane lines, but this time in units of meters, as opposed to pixels. For the conversion, I assume that 720 pixels in y direction (the height of the perspective transformation polygon) correspond to 30 meters, and that 630 pixels in x direction (the base of the perspective transformation polygon) correspond to 3.7 meters, which is the minimum width of a US highway lane.

* I then apply the definition of radius of curvature to the 2nd order polynominal fit to each lane line, evaluated at y = number of image rows.

* The vehicle x-offset from lane centre in pixels is given by `0.5*(x_left_lane_at_bottom + x_right_lane_at_bottom) - 0.5*image_width`, which assumes that the camera is mounted on the centre of the car. We then convert the offset to meters with the same conversion as before, nominally 3.7m/630px.

* For the final pipeline used for video processing, I use a more sofisticated function `calc_curvAndOffset_and_sanityChecks`. This function is defined in Section 2.5.4. The function performs sanity checks on lane width and radius of curvature, and also compares values estimated from the present image to estimations based on previous image frames. If estimates are deemed valid, it then updates the state of object Lane with a filter that gives weight 0.2 to the current estimation and 0.8 to former estimations.

            
####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

* Function `plot_lanes` implemented in Section 2.5.5 of the notebook unwarps the fits curves and colors on the original (undistorted) image the detected lane region in green.

* Next, function `process_image_isolated` in Section 2.5.6 defines the complete pipeline all the way from raw image to calculation and plotting of lane region and radius and curvature and offset values. But it does it by processing images in isolation.

* Function `process_image` in Section 2.5.6 defines the complete pipeline from raw image to lane region, curvature and offset, but does it taking into account previous estimations, with the Lane object state being stored after each cycle in variable `process_image.lane`. Note that after each processing cycle, the estimator can contain a valid estimate or be in "lost state". In particular, after 10 processing cycles (the tunning of parameter `numCyclesLostForRestart`) without producing a valid estimation, the estimator transitions to "lost state" and in the following cycle attempts a lane estimation without relying on previous estimates.

* Finally, function `process_image_for_video` also in Section 2.5.6 simply calls `process_image` and then adds the text with the radius of curvature and offset information to the processed image. This is the highest level function, which is called to process every video frame.

In Section 3 I run the entire pipeline on image "test3.jpg", plotting and logging the outcome of all intermediate steps. The final image with lane region, radius of curvature and position offset annotations, is shown below.

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my processed project video](./project_video_processed.mp4). The video is also available [in youtube](https://www.youtube.com/watch?v=xcgHLMjOHIA).

And here a [link to my processed challenge video, with identical pipeline](./challenge_video_processed.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The different methods considered for the various pipeline stages have been described in quite detail above.
But to sum up:
* I calibrated the camera with chessboard images, implementing a function able to find a different number of inside chessboard corners on each calibration image.
* I explored different color spaces and channels. In the end I used HSV color thresholding as well as gradient magnitude and direction thresholding to convert undistored RGB images into thresholded binary images.
* I defined a perspective transform from camera to birds-eye view.
* I implemented an algorithm based on histograms in sliding windows to identify points of the binary image belong to the left and right lane lines.
* I implemented two methods to fit curves to the detected lane line points: one constraining the lane lines to be parallel to each other, and the othe without the constraint. I used the latter in the final pipeline. I also estimated radius of curvature and vehicle positon offset from the curve fits.
* To improve stability, I implemented a filter to the estimation pipeline. I also introduced two estimating modes: from valid state, and from lost state. I defined a timeout of 10 cycles after which the estimator transitions to estimation from lost state.

The pipeline does an excellent job in the project video. It also does a decent --albeit imperfect job in the challenge video.

In the harder challenge video, however, the performance is unsatisfactory. The main reason is that the road in that video contains several very sharp curves for which the pipeline has not been tuned. However, I believe that by adding a few extra features to the presented pipeline, the algorithm could also do a decent job on the most challenging video. In particular, I propose the following additions:
* The current pipeline doesn't explicitly apply a region mask to the image, since the perspective transform already applies a masking implicitly. However, for the pipeline to be able to deal with both straight and sharply curvy road stretchs, a region mask should be added to the algorithm, and the geometry of the mask shall be changed based on the lane detection in previous circles (e.g. "focusing" on short distances on curvy stretches but using also pixels farther away in straight flat stretches).
* The reason why in the challenge video the detected lane sometimes shows some inexistent bending to the left is that the algorithm of detection of lane points sometimes misses a significant portion of the left lane line. The same binary tresholding tuning that worked excellently for the project video did not work as well with the challenge video. Obviously, there is room for improvement in the binary thresholding and lane line point detection stages of the pipeline.
* Finally, deep learning could be used in the problem of lane detection. Clearly, a neural network can be trained to estimate radius of curvatures and position offsets from raw videos (the labels simply inferred from human driver actions). Collecting data to train a neural network to identify the entire lane region is somewhat more complicated, but it can still be done. Car position estimations (from GPS, IMUs, etc.), together with an assumption for lane width are enough to provide the lane regions as labels to the neural network. Undoubtedly, a NN could be trained to achieve performances superior to those of computer vision algorithms tuned by humans in roads and visibility conditions never before experienced.

