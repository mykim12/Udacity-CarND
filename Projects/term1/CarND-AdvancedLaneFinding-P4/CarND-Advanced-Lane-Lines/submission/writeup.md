## Writeup for P4
---

**Advanced Lane Finding Project**

**Overall, it was a good experience. Followings are checkup items.**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. **[DONE]**
* Apply a distortion correction to raw images. **[DONE]**
* Use color transforms, gradients, etc., to create a thresholded binary image. **[Done]**
* Apply a perspective transform to rectify binary image ("birds-eye view"). **[Done]**
* Detect lane pixels and fit to find the lane boundary. **[Done]**
* Determine the curvature of the lane and vehicle position with respect to center. **[Done]**
* Warp the detected lane boundaries back onto the original image. **[Done]**
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position. **[Done]**

[//]: # (Image References)
[image1]: ./output_images/calibration2_fig.jpg "Undistorted"
[image2]: ./test_images/test2.jpg "test2"
[image2dt]: ./output_images/test2_undistorted.jpg "test2_distortion-corrected"
[image3]: ./output_images/thresholded_binaries_fig.jpg "Binary Example"
[image4]: ./output_images/perceptive_transform_fig.jpg "Warp Example"
[image4a]: ./output_images/perceptive_transform_fig2.jpg "Warp Binary Example"
[image5]: ./output_images/fitting_fig.jpg "Fit Visual"
[image5a]: ./output_images/fitting_next_fig.jpg "Fit Next"
[image6]: ./output_images/final_mapped.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---

### Writeup / README
**[Done!]**

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I basically followed the steps explained on the class. I wrote a class to take care of everything, end-to-end process for undistortion. The name of class is `Distort()`, which can be found in pipeline.py.

First, I read all chessboard images from camera_cal directory, and collect objpoints and imgpoints by using cv2.findChessboardCorners(). objpoints are the same throught the whole images and replicated. During this process, I also store images with corners detected by using cv2.drawChessboardCorners(). Second, with the objoints and imgpoints, the camera matrix and distortion coefficients are calculated by calling calibrate() function in the Distort class which eventually calls cv2.calibrateCamera(). Third, a sample image is undistorted by using cv2.undistort() with the matrix and coefficients. The **sample result** containing `original image`, `cornered image`, and `undistorted image` is saved after plot, and shown below:

![alt text][image1]

The code snippet is as follows:
```python
#------------------------------------------------------------
# STEP 1. CAMERA CALIBRATION
# create Distortion object with calibration images and corner size
#------------------------------------------------------------
DT = Distortion('./camera_cal', (9, 6))
DT.calibrate()
DT.undistort()
DT.showSampleResult()
```

### Pipeline (test images)

#### 1. Provide an example of a distortion-corrected image.
Using the codes implemented on the step `Camera Calibration`, below shows an example of a distortion-corrected image (selected from `test_images/`).

** original image**
![alt text][image2]

** distortion-corrected image**
![alt text][image2dt]

The code snippet is as follows:
```python
#------------------------------------------------------------
# STEP 2. APPLY ON A TEST image
#------------------------------------------------------------
test_img = mpimg.imread("./test_images/test2.jpg")
test_img_undist = DT.undistort(test_img)
mpimg.imsave("./output_images/test2_undistorted.jpg", test_img_undist)
```

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've written the learned methods during the lesson which does the color transforms and gradients calculations for creating the thresholded binary image, in `Threshold()` Class. The class can take either an image as an object or a file. In this experiment, I wanted to use the distortion-corrected image obtained from the previous step, I used `setImage()`.

I've experimented multiple combinations of color and gradient thresholds on the sample image, starting from what's suggested in the class. An example result of **thresholding** an image is shown below: (input image is the distortion-corrected image obtained from **Step 1**).

![alt text][image3]

During the experiment, I found that vertical information can be more important in this case (highway lane finding examples), so I set more strict threshold on *x-direction* gradient, and increased threshold ranges on *y-direction* gradient.

The final parameters I've chosen is as follows:
```python
#------------------------------------------------------------
# STEP 3. GENERATE THRESHOLDED BINARY IMAGE
#------------------------------------------------------------
TH = Threshold(_sobel_kernel=3)
TH.setImage(test_img_undist)
TH.run(_colorthresh=(90,255), _sthresh=(0, 255), \
       _gxthresh=(30, 80), _gythresh=(25, 120), \
       _mthresh=(30, 100), _dthresh=(0.7, 1.3))
```


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For perspective transformation, I've implemented `Warp()` class which takes an image, and set source and destination points, then finally transform the image. The source and destination points calulation I've selected are shown below:
```python
#------------------------------------------------------------
# DEF setPoints()
# set src points and destination points
#------------------------------------------------------------
def setPoints(self):
    offset = 100
    x, y = self.imsize
    cx, cy = x/2, y/2
    self.src = np.float32([ [(cx - offset/2 - 15), cy + offset],
                            [(x / 6) - 20, y],
                            [(x * 5 / 6 + offset), y],
                            [(cx + offset/2 + 20), cy + offset]    ])

    self.dst = np.float32([ [(x / 4), 0],
                            [(x / 4), y],
                            [(x * 3 / 4), y],
                            [(x * 3 / 4), 0]    ])
```

Actual coordinates are as follows:

| Source        | Destination   |
|:-------------:|:-------------:|
| 575, 460      | 320, 0        |
| 193, 720      | 320, 720      |
| 1167, 720     | 960, 720      |
| 710, 460      | 960, 0        |

An example result from Warp() procedure is shown as follows:

![alt text][image4]

As shown above, curved lanes are transformed to top-down view image and looks parallel.
Another result on the thresholded binary image is shown below:

![alt text][image4a]



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I followed steps from the class, and using histogram values, by sliding windows, tried to fit lane-line positions with a polynomial. The implementation is in `Polynomial()` Class. An example fitting from the top-down binary-warped image obtained above is shown as follows:

![alt text][image5]

Also, the following result is showing the fitting by reusing previous fit positions:

![alt text][image5a]
*(The same binary warped image is used)*

Code snippets from warping distortion-corrected & thresholded image, to fitting lane lines are as follows:

```python
#------------------------------------------------------------
# STEP 5. LOCATE LANE LINES & FIT A POLYNOMIAL
#------------------------------------------------------------
WP.setImage(TH.combined, _isgray=True)
WP.warp()
WP.visualize(_savef="./output_images/perceptive_transform_fig2.jpg")

wimg = WP.warped
PN = Polynomial(_bwimg=wimg)
PN.slideNfitPolynomial()
PN.visualize()

PN.fit_next(_newimg=wimg)
PN.visualize_next()

```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented `measureCurvature_and_mapLane()` function inside `Polynomial()` Class, to calculate the radius of curvature of the lane, and the position of the vehicle.

```python
PN.measureCurvature_and_mapLane()
# value output
# 1201.20614824 692.912494597
# 279.208739196 m 226.739041469 m

```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

It is also implemented inside `measureCurvature_and_mapLane()` function inside `Polynomial()` Class.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](video1)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In most of cases, I followed what's explained during the class first, and then explored out for better results. Some issues delayed me but most of the case, it was about correctly referring right variables throughout the pipeline. However, it also took much time to figure out the best combination of thresholding which can be generalized throughout the video frames. A disadvantage of my pipeline can be a memory problem, if lots of images are to be loaded. Thus, to make the pipeline more robust, batching operation can be implemented.
