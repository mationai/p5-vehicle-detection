**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[img1]: ./examples/car_not_car.png
[img2]: ./output_images/hog9.jpg
[img3]: ./output_images/hog10.jpg
[img4]: ./output_images/hog11.jpg
[img5]: ./output_images/hog12.jpg
[img6]: ./output_images/sliding_windows_330.jpg
[img7]: ./output_images/frame-656-pipeline.jpg
[img8]: ./output_images/singleimage-pipeline.jpg
[img9]: ./output_images/frame-262-Q.jpg
[img10]: ./output_images/frame-266-Q-gone.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### [Template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this writeup report.

### Note on inclusion of line numbers in code
Line numbers will *not* be mentioned in this report if function name is referenced since it is hard to kept line numbers in sync with code changes; where as function names are a bit more stable and is easy to search.

---
### Writeup / README

#### Source Files
The python source files are:
- `run.py` - process `project_video.mp4` and writes labeled `outputvid.mp4` to file.
- `train.py` - train model on training data and saves model to file.
- `config.py` - configuration for training and vehicle detection in video
- `draw_sliding_windows.py` - draw out sliding windows on an image
- `draw_features.py` - draw a hog visualization image
- `lib/detection.py` - vehicle detection library, contains the detection pipeline
- `lib/feature_extraction.py` - feature extraction functions
- `lib/draw.py`, `lib/color_palette.py` - drawing functions and colors
- `lib/np_util.py`, `lib/helpers` - helper functions

#### Data Files
Note the `..` in the data paths. If `data/` is not in `..`, modify `config.py` accordingly.
- `../data/vehicles/` - cars
- `../data/non-vehicles/` - not cars

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code to read in all "vehicle" and "non-vehicle" images is found at top of `train.py`:
```python
cars = [img for img in cars_imgspath]
notcars = [img for img in notcars_imgspath]
car_features = fe.images_features(cars, **defaults)
notcar_features = fe.images_features(notcars, **defaults)
```

`defaults` are defined `config.py` and `images_features()` is found in `feature_extraction.py`. It which loops through images and calls `image_features()` which calls `hog_features()` for hog features for each image.

Exploration of different color spaces and `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) are in table below.

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][img1]
*Fig 1. Car and Not Car*

Here is a visualization of different HOG orientations  

![][img2]
*Fig 2. HOG Orientation=9*

![][img3]
*Fig 3. HOG Orientation=10*

![][img4]
*Fig 4. HOG Orientation=11*

![][img5]
*Fig 5. HOG Orientation=12*

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried a few combinations of parameters and here are the results

| Color Space | Spatial Bins | Color Hist Bins | HOG Orientations | Pixs/Cell, Cells/Blk | Feature Vector Len | Test Accuracies (w/ diff seeds) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LUV | (32,32)| 32 | 12 | 8,2 | 10224 | 0.9921, 0.9893, 0.993 |
| LUV | (32,32)| 32 | 10 | 8,2 | 9048 | 0.9941 |
| LUV | (32,32)| 32 | 9 | 8,2 | 8460 | 0.9916 |
| YCrCb | (32,32)| 32 | 12 | 8,2 | 10224 | 0.9935 |
| YCrCb | (32,32)| 32 | 10 | 8,2 | 9048 | 0.9921 |
| YCrCb | (32,32)| 32 | 9 | 8,2 | 8460 | 0.9921 |

Overall it seems LUV 10 Orientations has the best result. My model uses the first configuration (LUV 12 Orientations) as I did not want to re-run the model on the video again as it takes a while and need to submit this report already. I will try the 10 Orientations configuration however as I like to know if it performs better result.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the following parameters defined in `config.py`:
```
    color_space='LUV',
    orient = 12,  # HOG orientations
    pix_per_cell = 8,  # HOG pixels per cell
    cell_per_block = 2,  # HOG cells per block
    hog_channel = 'ALL',  # Can be 0, 1, 2, or "ALL"
    train_size = (64, 64),  # train image size
    spatial_size = (32, 32),  # Spatial binning dimensions
    hist_bins = 32,  # Number of histogram bins
    hog_feat = True, 
    hist_feat = True, # worst if no hist
    spatial_feat = True, # much worst if no spatial
```

The training is done in `train.py`. The model is saved as file specified in `config.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, positioning of where the windows will slide needs to be determined. Windows, especially smaller ones can not be slided across the screen as doing so can take up to an hour to process just one second of video. The observation that the cars vanish toward a vanishing point and that point is near the center of the screen.

Once this point, or more specifically this horizontal line is determined, it is easy to visualize that the smallest windows will slide horizontally centered along this line and that the same window size does not need to be slided any further up or down from this vertical position. The next deduction is that the next window size up will need to be slided slightly more down vertically than up.

Coding it in a function that returns these bounding box positions will be helpful. This is done in `bbox_rows` function in `feature_extraction.py`. Here instead of starting from smallest window, I start from biggest and work my work until I cross the threshold of 80 x 80. Any smaller than that won't be that useful as the cars are far away.

The `xstep` is the number of steps within a window in the x direction. I start off with 10 steps but the detection is not great, so I ended up with 20. In the y direction, since I start with big windows first, it is easy to think of this in terms of percentage of height to go up. 20% is a good number here. 

To visualize these windows, I created `draw_sliding_windows.py` to draw them overlaying test1.jpg. Below is the result.

![][img6]
*Fig 6. Sliding windows*

`find_hot_wins` method in `detection.py` is where `bbox_rows` is called to find cars With these sliding windows. Boxes of heat that overlap will generate bounding windows of overlap heats. These windows are where cars may be found.  

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is an example of pipeline in action. The top window shows the output of `find_hot_boxes` as described above, in which the small heat box overlaps the bigger one and so the frame generates this new window of possible car. 

This window is matched against all existing cars to see if it can be group with any of them. Since the car has already been found previously, the window is associated with that car and no new car is added as shown in second window.

The last window is a threshold of window of windows associated with the car. More on this below.

![][img7]
*Fig 7. Frame 656 Pipeline*

Here's one where the pipeline is ran on a single image. Since there are no previous car found, the second image shows that the hot windows are added as new detections.

![][img8]
*Fig 8. Pipeline on Single Image*

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project-out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

There are several steps for filter of false positives, all of which are inside `detection.py`.

1. In `find_hot_wins` method, a threshold of 2 is used to filter out non overlapping heat bounding boxes. Value of 1 (no overlap needed) and 3 has been tried. 1 produces many false positives and 3 failed to detect some windows.

2. The heat windows that passes the above test is checked to see if it is too wide, too narrow, or too tall. This is done in lines 234 to 247 of `detect` method.

3. Once the windows passed the above tests, they are added as detected cars in memory but won't be show as detected yet. Cars that are rendered need to pass more purge filters.

4. Those with too many empty consecutive frames (consecutive no-detection new frames) are then removed - lines 290 to 295 in `purge` method.

5. All the windows in past frames (15 frames max) associated with detected cars are overlapped to generate a new window of windows heatmap. If less than 3 windows overlap, the car is not shown as detected - lines 304 to 318 in `final_purge_and_detection_image` method.

6. If the windows of windows is disjoint, the car is removed as windows of car should not be disjoint - lines 320 to 324.

7. Next, if window of windows is too small, narrow, or big, it is removed - lines 330 to 346.

All of above steps has been shown to help remove false positives in the video.

`bboxes_of_heat` function in `feature_extraction.py` is where overlapping bounding boxes are combined. It calls `scipy.ndimage.measurements.label()` which returns an array of labels that is of the same dimension as the image. These label values indicates if which of the heat areas are continuous. This information is used to constructed bounding boxes to cover the area of each blob detected.  

### Here is an example of false position in action

![][img9]
*Fig 1. Frame 262 - possible car 'Q' detected*

The possible car 'Q' removed due to 3 consecutive empty frames while its lifetime frames is only 5
![][img10]
*Fig 1. Frame 266 - possible car 'Q' removed*

---

### Running
Edit `run.py` to specify the video_in file. Set it to False to run pipeline on a single image.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After spending so much time on P4 such that I am too embarrassed to even mention it, I thought P5 may be easier. In a sense it should be as I look back and compare the two. However, due to the long feedback loop of running the video, getting stuck in various errors, and creating a good side and bottom windows system to better troubleshoot and optimize the detection, false positives and purge, the end result is I have once again spent way too much time on this. All this effort may not worth it if some Deep Learning method deem this classical method irrelevant. The model is too brittle. All the magic parameters are tunned to just this one video. There's probably much better classical models out there, but I'm leaning towards believing that Deep Learning methods will surpass classical ones, if not already.

