## Writeup for Vehicle Detection and Tracking project

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_and_not_car_sample.png
[image2]: ./output_images/HOG_sample_1.png
[image2a]: ./output_images/HOG_sample_2.png
[image2b]: ./output_images/HOG_sample_2_origimg.png
[image3]: ./output_images/final_features_norm.png
[image4a]: ./output_images/detectionWindows.png
[image4b]: ./output_images/detectionWindows_2.png
[image4c]: ./output_images/detectionWindows_3.png
[image4]: ../examples/sliding_window.jpg
[image5]: ../examples/bboxes_and_heat.png
[image6]: ../examples/labels_map.png
[image7]: ../examples/output_bboxes.png
[video1]: ../project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First of all, I downloaded **vehicles.zip** and **non-vehicles.zip** and extracted them under `../data` directory. This is a symbolic link of the actual location, and I'll assume the reviewer will also have the dataset. (it's too big to upload). Then, I implemented `Data()` Class which can load images and also visualize sample images.
```python
# at around line 13
class Data(object):
    def __init__(self, _dpath="../data"):
        self.data_path = _dpath

        self.cars = []
        self.non_cars = []
        self.data_dict = {}
```

The following figure shows an example set of car and non-car images:

![alt text][image1]

To extract HOG features, `skimage.hog()` is used and by some experiments, such as changing parameters like `orientations`, `cells_per_block`, and `pixels_per_cell`, and trying out multiple random images, the following results were obtained:

```python
# around line 110
#------------------------------------------------------------
# DEF HOG_expr
# - some HOG experiments
#------------------------------------------------------------
def HOG_expr(self):
```

*(`orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` were chosen)*

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried several different choice of parameters on a sample image:
![alt text][image2b] and as shown on the figure below, when I increase `cells_per_block` from original setting **2** to **4** and **6**, the hog result seems to be similar in this case *(the 1st row of the figure)*. However, when I tried changing `pixels_per_cell` value from **8** to **4** and **16**, the hog results became simpler and bigger as the value increased *(as shown on the 2nd row of the figure)*. I think if the `pixels_per_cell` value is too small, then the size of # of features will also increase, resulting in high computational overhead. Lastly, the bigger `orientation` values, the more blurry the hog results became *(as shown on the 3rd row of the figure)*. I think this is because if the # of `orientation` bins increase, then each orientation value will be more distributed resulting in less decisive in any directions. In sum, I finally chose `pixels_per_cell=8`, `cell_per_block=2`, and `orient=9`, which produces the good enough hog features.

```python
# around line 152
#------------------------------------------------------------
# DEF HOG_expr2
# - some HOG experiments
#------------------------------------------------------------
def HOG_expr2(self):
```
![alt text][image2a]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I also used color features along with HOG features, and all feature extractions are implemented in `Features()` Class.
```python
# around line 92
class Features(object):
    def __init__(self, _data):
        self.data = _data
```

As mentioned in the class, normalization on each features are applied to gain proper comparable values among multiple types of features.
Here's the sample results showing before/after normalization.

![alt text][image3]

After obtaining features, I trained a linear SVM using color features first. Generic Classifier object is implemented in `Classifier` Class, which takes `Feature` Class object and `_testSize` for training/testing data distribution.
```python
# around line 363
class Classifier(object):
    def __init__(self, _FT, _testSize=0.2):
        self.Feature = _FT
        self.tsize = _testSize
```

I trained the classifier first with color features. After setting feature parameters, simply calling `SVC_ColorFeatures()` function inside `Classifier()` did the job.
```python
# line 440
#------------------------------------------------------------
# STEP 3. CLASSIFIER
#------------------------------------------------------------
CLSF = Classifier(FEAT)
CLSF.Feature.setFeatParams(_cspace='RGB', _ssize=(32, 32),
                           _hist_bins=32, _hist_range=(0, 256))
CLSF.SVC_ColorFeatures(_npredict=10)
```  

and belows are sample training results using **20 cars and non-cars** small dataset with `spatial_size=32`, and `histogram_binsize=32`, for predicting 10 test samples.

```
Total 20 cars and 20 non-cars of size (64, 64, 3) and data type: float32 are collected from data directory ../data
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 3168
0.01 Seconds to train SVC...
Test Accuracy of SVC =  0.875
My SVC predicts:  [ 1.  1.  0.  0.  1.  1.  1.  0.]
For these 10 labels:  [ 1.  1.  0.  0.  1.  1.  0.  0.]
0.00051 Seconds to predict 10 labels with SVC
```
When increased the size of the training data to 200, the accuracy on the 10 predictions **jumped up from 87.5% to 97.5%**!

```
Total 200 cars and 200 non-cars of size (64, 64, 3) and data type: float32 are collected from data directory ../data
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 3168
0.49 Seconds to train SVC...
Test Accuracy of SVC =  0.975
My SVC predicts:  [ 1.  0.  0.  1.  0.  0.  1.  1.  0.  1.]
For these 10 labels:  [ 1.  0.  1.  1.  0.  0.  1.  1.  0.  1.]
0.00059 Seconds to predict 10 labels with SVC
```

Then, I conducted several training using different combination of features, with different parameters.
```python
# train with color features
CLSF.Feature.setFeatParams(_cspace='RGB', _ssize=(32, 32),
                           _hist_bins=32, _hist_range=(0, 256))
CLSF.SVC_ColorFeatures(_npredict=10)

# train with HOG features
CLSF.Feature.setHOGParams(_ppc=8, _cpb=2, _orient=9, _vis=False,
                          _chnl='ALL', _fv=True)
CLSF.SVC_HOGFeatures(_npredict=10)

# train with all features (color & HOG)
CLSF.SVC_AllFeatures(_npredict=10)
```

Training results of the above are as follows:
```
Total 200 cars and 200 non-cars of size (64, 64, 3) and data type: float32 are collected from data directory ../data
======== SVC_ColorFeatures() ========
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 3168
0.45 Seconds to train SVC...
Test Accuracy of SVC =  0.95
My SVC predicts:  [ 1.  1.  1.  0.  1.  0.  0.  1.  0.  0.]
For these 10 labels:  [ 1.  1.  1.  0.  1.  0.  0.  1.  0.  0.]
0.01084 Seconds to predict 10 labels with SVC
======== SVC_HOGFeatures() ========
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 5292
0.04 Seconds to train SVC...
Test Accuracy of SVC =  1.0
My SVC predicts:  [ 0.  0.  1.  0.  1.  1.  0.  1.  0.  1.]
For these 10 labels:  [ 0.  0.  1.  0.  1.  1.  0.  1.  0.  1.]
0.00095 Seconds to predict 10 labels with SVC
======== SVC_AllFeatures() ========
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 8460
0.09 Seconds to train SVC...
Test Accuracy of SVC =  1.0
My SVC predicts:  [ 1.  0.  0.  1.  1.  1.  0.  1.  1.  0.]
For these 10 labels:  [ 1.  0.  0.  1.  1.  1.  0.  1.  1.  0.]
0.00117 Seconds to predict 10 labels with SVC

```

I tried increase ``train data size`` to **500** and also the ``test data size`` to **100**.

```
Total 500 cars and 500 non-cars of size (64, 64, 3) and data type: float32 are collected from data directory ../data
======== SVC_ColorFeatures() ========
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 3168
0.99 Seconds to train SVC...
Test Accuracy of SVC =  0.955
0.0044 Seconds to predict 100 labels with SVC
======== SVC_HOGFeatures() ========
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 5292
0.09 Seconds to train SVC...
Test Accuracy of SVC =  0.995
0.00371 Seconds to predict 100 labels with SVC
======== SVC_AllFeatures() ========
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 8460
0.2 Seconds to train SVC...
Test Accuracy of SVC =  0.99
0.00392 Seconds to predict 100 labels with SVC

```

**In conclusion, combining all features of colors & HOGs performed the best.**



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I started to use `slide_window()` function from the class. It is implemented inside `Detector()` Class.
```python
class Detector(object):
    def __init__(self, _classifier, _yss=[None, None],
                       _xywin=(96, 96), _xyoverlap=(0.5, 0.5)):
        self.classifier = _classifier
        self.y_start_stop = _yss
        self.xy_window = _xywin
        self.xy_overlap = _xyoverlap
```
Also, when I tried running the detection without increasing the size of training data for the classifier, it **didn't produce any hot_windows** as results. Thus, I increased the size of training examples to **5000 each (car/non-car)**, and finally I was able to get some *hot_windows*. However, when I used `y_start_stop=[100, None]` as default, I got this result:

![alt text][image4a]

Well, since there is no cars on this image, it's obvious that there's not much of `hot_windows`, but it was strange that so many `hot_windows` were detected on the sky. So, I set `y_start_stop=[200, 660]` (*ystop for excluding car itself*), then I got this result:

![alt text][image4b]

Now, I tried using another example where I can see some cars:

![alt text][image4c]

Now, I think I got some `hot_window` for potential cars, but not perfect. So, I started to use *scale* factor in searching. In addition, I utilized `find_cars()` function and here is the result:
```python
DET.setYSS((400, 656))
DET.findCars(image, _scale=1.5)
```



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
