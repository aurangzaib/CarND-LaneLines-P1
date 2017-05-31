# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image0-1]: ./results/00-original-1496136918.jpg "Scenario"
[image0-2]: ./results/00-original-1496136931.jpg "Scenario"
[image0-3]: ./results/00-original-1496136945.jpg "Scenario"

[image2]: ./results/01-hsv-1496135300.jpg "HSV"
[image3]: ./results/01-hsv-1496135331.jpg "HSV"
[image3-2]: ./results/01-hsv-1496135356.jpg "HSV"

[image4]: ./results/02-yellow-1496135300.jpg "Yellow Mask"
[image5]: ./results/02-yellow-1496135355.jpg "Yellow Mask"

[image6]: ./results/03-white-1496135366.jpg "White Mask"
[image6-2]: ./results/03-white-1496135329.jpg "White Mask"
[image7]: ./results/04-mask-hsv-1496135309.jpg "Mask HSV"
[image7-2]: ./results/04-mask-hsv-1496135361.jpg "Filtered"
[image9]: ./results/07-filtered-1496135354.jpg "Filtered"
[image10]: ./results/07-filtered-1496135303.jpg "Filtered"

[image11]: ./results/08-canny-1496135302.jpg "Edges"
[image12]: ./results/08-canny-1496135366.jpg "Edges"

[image13]: ./results/09-ROI-1496135301.jpg "ROI"
[image13-1]: ./results/09-ROI-1496135329.jpg "ROI"
[image13-2]: ./results/09-ROI-1496135303.jpg "ROI"
[image14]: ./results/10-hough_line_dark-1496215679.jpg "Hough Lines"
[image15]: ./results/10-hough_line_dark-1496215711.jpg "Hough Lines"
[image15-2]: ./results/10-hough_line_dark-1496215737.jpg "Hough Lines"
[image16]: ./results/11-lanes-1496135307.jpg "Lanes"
[image17]: ./results/11-lanes-1496135366.jpg "Lanes"
[image17-2]: ./results/11-lanes-1496135339.jpg "Lanes"

[image18]: ./results/12-resultant-1496135312.jpg "Result"
[image19]: ./results/12-resultant-1496135332.jpg "Result"
[image20]: ./results/12-resultant-1496135364.jpg "Result"

In this project, our goal is to:
    <ul>
        <li>Find the lanes in the images.</li>
        <li>Apply smoothing to draw only smooth lanes.</li>
    </ul>

## The Pipeline:

<br />

The pipeline consists of 10 steps:
    <ol>
        <li>Get the lane info in the image.</li>
        <li>Remove noise from the image using Gaussian Blur Filter.</li>
        <li>Get the edges in the images using Canny Edge Detector</li>
        <li>Remove edges outside of the ROI</li>
        <li>Find Lines using Probabilistic Hough Line Transform: </li>
        <li>Find smooth lanes using Weighted Arithmetic Mean</li>
        <li>Find the coordinates of the Lanes</li>
        <li>Draw Lanes</li>
        <li>Merge results in the original image</li>
    </ol>

<br />

#### Step 1: Get the Lane info in the image:

The original sample images look like:

![alt text][image0-1]
![alt text][image0-2]
![alt text][image0-3]

First, HSV channels information is extracted in order to get the objects in a frame of specific color.

![alt text][image2]
![alt text][image3]
![alt text][image3-2]

Now, the yellow and white range objects are extracted from the image using inRange function.

```python
    mask_yellow = cv.inRange(image_hsv,
                             np.array([20, 100, 100]),  # yellow lower
                             np.array([100, 255, 255]))  # yellow upper

```

![alt text][image4]
![alt text][image5]

```python
    # get region of image within white range
    mask_white = cv.inRange(image_hsv,
                            np.array([0, 0, 240]),  # white lower
                            np.array([255, 255, 255]))  # white lower
```

![alt text][image6]
![alt text][image6-2]

At this point, the results of both the white and yellow masks are merged using bitwise OR operator. Then these masks are applied on the image using bitwise AND operator.

```python
    # merge results of yellow and white - bitwise OR
    mask_hsv = cv.bitwise_or(mask_yellow, mask_white)
    # only retain the information of the lanes in the image - bitwise AND
    masked_image = cv.bitwise_and(image_gray, mask_hsv)
```

![alt text][image7]

<br />

#### Step 2: Remove noise from the image using Gaussian Blur Filter

Guassian Blur Filter is used to remove the noise from the image.
```python
def remove_image_noise(image, kernel=(3, 3)):
    filtered_image = cv.GaussianBlur(image,  # source
                                     kernel,  # kernel size
                                     0)  # border type
    return filtered_image
```                                     
![alt text][image9]
![alt text][image10]

<br />

#### Step 3: Find the edges in the images using Canny Edge Detector

Canny Edge Detector is used to find the edges. The edges are the points where the grayscale values change quite sharply.

```python
def get_edges(image, low, high):
    edges = cv.Canny(image,
                     low,  # low threshold
                     high)  # high threshold
    return edges
```                

![alt text][image11]
![alt text][image12]

<br />

#### Step 4: Remove edges outside of the ROI

There can be several objects in an image which are not of interest, e.g. trees etc. A region of interest (ROI) needs to be created to remove the extra details in the image beside the lanes.

```python
def get_masked_edges(width, height_bottom, height_top, edges, ignore_mask_color=255):
    # blank matrix
    mask = np.zeros_like(edges)
    # ROI vertices
    vertices = np.array([[(width * 0.05, height_bottom),  # left bottom
                          (width / 3., height_top),  # left top
                          (2 * width / 3., height_top),  # right top
                          (width * 0.95, height_bottom)]],  # right bottom
                        dtype=np.int32)
    # create ROI
    cv.fillPoly(mask, vertices, ignore_mask_color)
    # remove the parts of image which are not within ROI - bitwise AND
    masked_edges = cv.bitwise_and(edges, mask)
    return masked_edges
```
![alt text][image13]
![alt text][image13-1]
![alt text][image13-2]

<br />

#### Step 5: Find Lines using Probabilistic Hough Line Transform: 

Then, Probabilistic Hough Line Transform is used to get the lines. It provides the start and end points of each detected line.
```python
def get_hough_lines(image, rho=2, theta=1, voting=25, min_line_length=20, max_line_gap=1):
    return cv.HoughLinesP(image, rho, theta * np.pi / 180,  # degree to radian
                          voting,  # minimum voting in hough accumulator
                          min_line_length,  # min line length in pixels
                          max_line_gap  # gap b/w lines in pixels
                          )
```
![alt text][image14]
![alt text][image15]
![alt text][image15-2]

<br />

#### Step 6: Find smooth lanes using Weighted Arithmetic Mean

Hough Line Transform provides several lines of different slopes and lengths. The ideal solution is to have only two lines, one for left lane and the other for right lane.

Since Hough lines vary in length, the most important and useful line are those which are longest in length because the smaller lines might be due to the noise and random calculations.

[https://en.wikipedia.org/wiki/Weighted_arithmetic_mean]

The weighted average can be used where the line length can be used as a weight.
```python
def get_weighted_lanes(detected_lines):
    right_slope_intercept = []
    right_length = []
    left_slope_intercept = []
    left_length = []
    for line in detected_lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            line_slope = float(y2 - y1) / float(x2 - x1)
            line_angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            line_intercept = y1 - line_slope * x1
            line_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if line_slope < 0:
                left_slope_intercept.append((line_slope, line_intercept))
                left_length.append(line_length)
            elif line_slope > 0:
                right_slope_intercept.append((line_slope, line_intercept))
                right_length.append(line_length)
    sop_left, sop_right = np.dot(left_length, left_slope_intercept), np.dot(right_length, right_slope_intercept)
    left_weight, right_weight = np.sum(left_length), np.sum(right_length)
    # handling cases when no line is detected by hough transform
    left_bound = len(left_length) > 0 and sop_left[0] < -0.5
    right_bound = len(right_length) > 0 and sop_right[0] > 0.5
    left = (sop_left / left_weight) if left_bound else None
    right = (sop_right / right_weight) if right_bound else None
    save_history(left, right)
    return left, right
```

<br />

#### Step 7: Find the coordinates of the lanes:

At this point, we have the slopes and the intercepts of both lanes. Now we need to calculates the start and end coordinates for both lanes.

Additionally, we also need to handle the cases when there is no Hough lines are found. This problem is solved by keeping the history of the slopes and intercepts for the frames and using the average of the them in case no Hough lines are found.

```python
# height_bottom, height_top, left, right
def get_coordinates(h_bottom, h_top, l, r):
    if l is None:
        l = [np.mean(left_slope_history[-10:]), np.mean(left_intercept_history[-10:])]
    if r is None:
        r = [np.mean(right_slope_history[-10:]), np.mean(right_intercept_history[-10:])]
    slope, intercept = l
    x_l1,x_l2 = int((h_bottom - intercept) / slope), int((h_top - intercept) / slope)
    slope, intercept = r
    x_r1, x_r2 = int((h_bottom - intercept) / slope), int((h_top - intercept) / slope)
    y1, y2 = int(h_bottom), int(h_top)
    return ((x_l1, y1), (x_l2, y2)), ((x_r1, y1), (x_r2, y2))
```

<br />

#### Step 8: Draw Lanes:

Now, we can draw the detected lanes on a blank image for debugging and testing. This step is not necessary and the results can directly be drawn on the actual image without first drawing on the blank image.

```python
def draw_weighted_lanes(edges, left_coordinates, right_coordinates, color=(0, 0, 255), thickness=5):
    lane_image = np.zeros((edges.shape[0],  # --> height
                           edges.shape[1],  # --> width
                           3), dtype=np.uint8)
    for coordinates in (left_coordinates, right_coordinates):
        start_coordinate, end_coordinate = coordinates
        cv.line(lane_image,  # source
                start_coordinate, end_coordinate,  # (x2, y2)
                color, thickness
                )
    return lane_image 
```

![alt text][image16]
![alt text][image17]
![alt text][image17-2]

<br />

#### Step 9: Merge results in the original image

The detected and smooth lanes can now be drawn on the actual image using addWeighted function. It merges two images with configurable weights (importance of an individual image).

```python
def get_resultant_image(source1, source2, alpha=0.8, beta=1, gamma=0):
    resultant = cv.addWeighted(source1,
                               alpha,  # weight of the first array elements
                               source2,  # source 2
                               beta,  # weight of the second array elements
                               gamma)  # scalar added to each sum
    return resultant
```

![alt text][image18]
![alt text][image19]
![alt text][image20]

<br />

## Suggest possible improvements to the pipeline

<ol>
    <li>
        Hough Line Transform can be used in combination of line stitching techniques in which a curve lane can be approximated using convolution of several small line.
    </li>
    <li>
        Kalman Filter can be used to predict the future values in case of occlusion or when the lane markings are missing.
    </li>
</ol>

<br />


## Identify potential shortcomings with the current pipeline

<ol>
    <li>
        In case when no lines are detected by Hough Transform, the history is used to keep extrapolating the lines, but it has the drawback that if lanes are bent sharply or there is a bump on the road then the predictions using the history could be quite wrong.
    </li>
    <li>
        Hough Line Transform is only useful when the lanes are not bending. The current pipeline will have issues with curvy and snake type roads.
    </li>
<ol>

<br />