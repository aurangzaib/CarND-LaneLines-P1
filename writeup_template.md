# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image0-1]: ./results/00-original-1496136918.jpg "Scenario"
[image0-2]: ./results/00-original-1496136931.jpg "Scenario"
[image0-3]: ./results/00-original-1496136945.jpg "Scenario"
[image2]: ./results/01-hsv-1496135300.jpg "HSV"
[image3]: ./results/01-hsv-1496135331.jpg "HSV"
[image3]: ./results/01-hsv-1496135356.jpg "HSV"
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
[image14]: ./results/10-hough_line_dark-1496215679.jpg "Hough Lines"
[image15]: ./results/10-hough_line_dark-1496215711.jpg "Hough Lines"
[image15-2]: ./results/10-hough_line_dark-1496215730.jpg "Hough Lines"
[image16]: ./results/11-lanes-1496135307.jpg "Lanes"
[image17]: ./results/11-lanes-1496135366.jpg "Lanes"

[image18]: ./results/12-resultant-1496135312.jpg "Result"
[image19]: ./results/12-resultant-1496135332.jpg "Result"
[image20]: ./results/12-resultant-1496135364.jpg "Result"

In this project, our goal is to:
    <ul>
        <li>Find the lanes in the images.</li>
        <li>Apply smoothing to draw only smooth lanes.</li>
    </ul>

Below, I will go through each step of the pipeline.

### Pipeline

The Pipeline consists of 10 steps:
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
## Step 1: Get the Lane info in the image:

First thing first, the original sample images look like:

![alt text][image0-1]
![alt text][image0-2]
![alt text][image0-3]

I first extracted the HSV channels information in order to get the objects in a frame of specific colors.

![alt text][image2]
![alt text][image3]

Now I extracted the yellow and white range objects from the image using inRange function. Then I merged the results of both masks.
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

At this point, I merged the results of both the white and yellow masks using bitwise OR operator. Then these masks are applied on the image using bitwise AND operator.

```python
    # merge results of yellow and white - bitwise OR
    mask_hsv = cv.bitwise_or(mask_yellow, mask_white)
    # only retain the information of the lanes in the image - bitwise AND
    masked_image = cv.bitwise_and(image_gray, mask_hsv)
```
![alt text][image7]

## Step 2: Remove noise from the image using Gaussian Blur Filter

Guassian Blur Filter is used to remove the noise from the image.
```python
    filtered_image = cv.GaussianBlur(image,  # source
                                     kernel,  # kernel size
                                     0)  # border type
```                                     
![alt text][image9]
![alt text][image10]

## Step 3: Get the edges in the images using Canny Edge Detector
```python
    edges = cv.Canny(image,
                     low,  # low threshold
                     high)  # high threshold
```                
![alt text][image11]
![alt text][image12]

## Step 4: Remove edges outside of the ROI
```python
    # blank matrix
    mask = np.zeros_like(edges)
    # ROI vertices
    vertices = np.array([[(width * 0.05, height_bottom),  # left bottom
                          (width / 3., height_top),  # left top
                          (2 * width / 3., height_top),  # right top
                          (width * 0.95, height_bottom)]],  # right bottom
                        dtype=np.int32)
    
    # create ROI
    cv.fillPoly(mask,  # image
                vertices,  # coordinates
                ignore_mask_color)
    # remove the parts of image which are not within ROI - bitwise AND
    masked_edges = cv.bitwise_and(edges, mask)
```
![alt text][image13]

## Step 5: Find Lines using Probabilistic Hough Line Transform: 

Then, Probabilistic Hough Line Transform is used to get the lines.
```python
    return cv.HoughLinesP(image,  # source
                          rho,
                          theta * np.pi / 180,  # degree to radian
                          voting,  # minimum voting in hough accumulator
                          min_line_length,  # min line length in pixels
                          max_line_gap  # gap b/w lines in pixels
                          )
```
![alt text][image14]
![alt text][image15]
![alt text][image15-2]

## Step 6: Find smooth lanes using Weighted Arithmetic Mean

```python
def get_weighted_lanes(detected_lines):
    # hough line transform provides several detected lines
    # we want to reduce all the to just 2 lines
    # lines will be weighted w.r.t the length of the lines
    # resultant lines are called lanes (left and right)
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

## Step 7: Find the coordinates of the lanes:

At this point, we have the slopes and the intercepts of both lanes. Now we need to calculates the start and end coordinates for both lanes.

```python
def get_coordinates(height_bottom, height_top, left, right):
    if left is None:
        left = [np.mean(left_slope_history[-10:]), np.mean(left_intercept_history[-10:])]
    if right is None:
        right = [np.mean(right_slope_history[-10:]), np.mean(right_intercept_history[-10:])]

    slope, intercept = left
    x_l1 = int((height_bottom - intercept) / slope)
    x_l2 = int((height_top - intercept) / slope)

    slope, intercept = right
    x_r1 = int((height_bottom - intercept) / slope)
    x_r2 = int((height_top - intercept) / slope)

    y1 = int(height_bottom)
    y2 = int(height_top)
    return ((x_l1, y1), (x_l2, y2)), ((x_r1, y1), (x_r2, y2))
```
![alt text][image16]
![alt text][image17]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

## Step 8: Draw Lanes:

```python
def draw_weighted_lanes(edges, left_coordinates, right_coordinates,
                        color=(0, 0, 255), thickness=5):
    lane_image = np.zeros((edges.shape[0],  # --> height
                           edges.shape[1],  # --> width
                           3),
                          dtype=np.uint8)
    for coordinates in (left_coordinates, right_coordinates):
        start_coordinate, end_coordinate = coordinates
        cv.line(lane_image,  # source
                start_coordinate,  # (x1, y1)
                end_coordinate,  # (x2, y2)
                color,  # RGB color --> red
                thickness  # line thickness
                )
    if SHOW_DEBUG_IMAGES:
        save_image(lane_image, "11-lanes")
        plt.imshow(lane_image)
    return lane_image 
```

## Step 9: Merge results in the original image

```python
def get_resultant_image(source1, source2, alpha=0.8, beta=1, gamma=0):
    resultant = cv.addWeighted(source1,
                               alpha,  # weight of the first array elements
                               source2,  # source 2
                               beta,  # weight of the second array elements
                               gamma)  # scalar added to each sum
    save_image(resultant, "12-resultant")
    return resultant

```

![alt text][image18]
![alt text][image19]
![alt text][image20]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
