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
[image4d]: ./output_images/detection_findCars_1.png
[image4e]: ./output_images/detection_findCars_1-2.png
[image4f]: ./output_images/detection_findCars_3.png
[image5a]: ./output_images/detection_heatMap_figure_2.png
[image5b]: ./output_images/detection_heatMap_figure_Accumulate.png
[video1]: ../project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

 [Done!]

---
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
# around line 391
class Classifier(object):
    def __init__(self, _FT, _testSize=0.2):
        self.Feature = _FT
        self.tsize = _testSize
```

I trained the classifier first with color features. After setting feature parameters, simply calling `SVC_ColorFeatures()` function inside `Classifier()` did the job.
```python
# line 807
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



---
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I started to use `slide_window()` function from the class. It is implemented inside `Detector()` Class.
```python
# around line 481
class Detector(object):
    def __init__(self, _classifier, _yss=[None, None],
                       _xywin=(96, 96), _xyoverlap=(0.5, 0.5)):
        self.classifier = _classifier
        self.y_start_stop = _yss
        self.xy_window = _xywin
        self.xy_overlap = _xyoverlap
```
Also, when I tried running the detection without increasing the size of training data for the classifier, it **didn't produce any hot_windows** as results. Thus, I increased the size of training examples to **5000 each (car/non-car)**, and finally I was able to get some *hot_windows*. When I used `y_start_stop=[200, 660]` I got this result:

![alt text][image4b]

Now, I tried using another example where I can see some cars:

![alt text][image4c]

Now, I think I got some `hot_window` for potential cars, but not great. So, I did some experiments and found a few bugs related to X_scaler, fixed it (details in the latter sections), and started to use *scale* factor in searching. In addition, I utilized `find_cars()` function.
```python
DET.setYSS((400, 656))
DET.findCars(image, _scale=1.5)
```



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I ended up using all the features(color features and hog feature) together to get the best result. Also, I could understand the `StandardScaler()` better during this step, and the transform() by the `X_scaler` is highly recommended to have a high performance. In contrast the results above (from window search), by normalizing it and tuning a bit of the sample sizes for training data, I was able to produce a nice result (see below). One more thing to note here is the `scale` parameter of the window, which also affected the performance. Later, I extended codes to run detection with multiple scales;a sample result will be shown in the next section.

![alt text][image4d]


I've also did more experiment tuning ratio of positive/negative samples, training/testing ratio, also the random variety sets of training data, etc. I implemented optimization routine as follows:
```python
868     acc = 0.0
869     # desired accuracy
870     max_acc = 0.997
871     while(acc < max_acc):
872         # randomly generate training data with random numbers
873         CLSF.Feature.data.load_data(_car_sample=randint(1000, 2800), _noncar_sample=randint(1000, 3800))
874         acc = CLSF.SVC_AllFeatures(_npredict=-1)
875         print("acc: ", acc)
```

And here's some intermideate procedures:
```
[Iter 3]
Total 1205 cars and 3136 non-cars of size (64, 64, 3) and data type: float32 are collected from data directory ../data
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 8460
17.98 Seconds to train SVC...
Test Accuracy of SVC =  0.987
My SVC predicts:  [ 0.  1.  0. ...,  0.  0.  1.]
For these -1 labels:  [ 0.  1.  0. ...,  0.  0.  1.]
0.00865 Seconds to predict -1 labels with SVC

[Iter 4]
Total 2318 cars and 3552 non-cars of size (64, 64, 3) and data type: float32 are collected from data directory ../data
Using spatial binning of: (32, 32) and 32 histogram bins
Feature vector length: 8460
15.39 Seconds to train SVC...
Test Accuracy of SVC =  0.9932
My SVC predicts:  [ 0.  0.  1. ...,  0.  0.  0.]
For these -1 labels:  [ 0.  0.  1. ...,  0.  0.  0.]
0.01149 Seconds to predict -1 labels with SVC
```

Duing the optimization, I also save the trained svc which produces relatively high accuracy with `acc_thr=0.9936`. Finally, I obtained the best model with **`accuracy=0.994`**, and here's the detection result on a test image with multiple scales `scales=[1.2, 1.5, 1.8]`.

##### The best result with `scale=1.2`

![alt text][image4e]

##### The result with all scales (`[1.2, 1.5, 1.8]`)

![alt text][image4f]



---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I tried creating a heatmap of detections, and set threshold to filter out false positives as much as possible. At first, I just set threshold as a fixed value as `2`, and after I obtain a heatmap, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Finally, I generated bounding boxes to cover the area of each blob detected.
A sample result is shown below:

![alt text][image5a]


Then, for video detection, I accumulated heatmap for a short period of time (6 frames), to remove more false positives. The below shows a sample 6-frame heatmaps and detections.


![alt text][image5b]

As one of the last steps, I aslo tried to remove more false positives by filtering out bounding boxes whose ratio of hight and width is not quite right as a car object, also whose area is too small to be a car.

```
 805             # filter out obvious false positives using bbox ratio
 806             bbox_w = np.max(nonzerox) - np.min(nonzerox)
 807             bbox_h = np.max(nonzeroy) - np.min(nonzeroy)
 808 
 809             area = bbox_w * bbox_h
 810             ratio = float(bbox_h) / float(bbox_w)
 811             if area < 3000: continue
 812 
 813             if ratio < 0.3: continue
 814             if ratio > 1.9: continue
```

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One issue as already mentioned above, was about scaling (normalization). Without it, the performance was not good. Also, to make nice codes, I had to design really carefully to appropriately implement all sort of conditions for every different experiments. Even though the final results seem to be good, this framework can fail if car is occluded by other types of objects, resulting in having lack of enough features for prediction. Also, sliding window approach tends to be slower than other methods like detection using fully confolutional network. Another thing to note is that some hyper parameters, such as scale, makes the system vulnerable to be generalized enough for various spectrum of the object size. Bounding box proposals can be replaced with Selective Search which could also make the system more robust than sliding window fashion.


#### 2. False positive handling

I tried additional steps as following:
1. using different color space for each feature (HSV for HOG and Color feature, and LUV for spatial binning). 
2. Also changed parameters of feature processing according to the histogram.
	* spatial bin size `(16, 16)` (from `(32, 32)`)
	* color histogram bin size `64` (from `32`)
	* histogram orientation bin: `10` (from `9`)
3. use mlutiple scaling `[ 0.5, 1.0, 1.5 ]`.
4. use decision_function for SVC instead of predict()
	* set `threshold=0.8`
	* 
```python
806                 test_prediction = self.classifier.svc.decision_function(test_features)
807                 predictions.append(test_prediction)
809 
810                 # draw rectangle if predicted true (car)
811                 #if test_prediction == 1:
812                 if test_prediction > 0.8:
```

5. keep history queue for previous consecutive frames for a bit.

```python
1179     history = deque(maxlen=30)
# in DET.heatMap_acc()
983         # add heatmaps to history
984         _history += heatmaps
```

#### 3. Two Bugs - The Real reason why it generated lots of false positives
During experimenting the '*2.False positive handling*' above, I found a major bug which made my system *'looks like (but not actually)'* generating lots of false positive. The **false positives were filtered out correctly** by my codes, *however*, I had a bug where **I keep drawing the filtered out boxes** in some cases. Also, I found another bug, where **it's not correctly accumulating global heatmaps**. So, I fixed them all, and it looked a lot better.


