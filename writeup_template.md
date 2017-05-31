# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


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

---
### Pipeline

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

![alt text][image0-1]
![alt text][image0-2]
![alt text][image0-3]

The Pipeline consists of 10 steps:

## Step 1:
    
move the image in HSV domain. 

![alt text][image2]
![alt text][image3]

Get yellow lines

![alt text][image4]
![alt text][image5]

white lines and merge the results using
bitwise OR operator.

![alt text][image6]
![alt text][image6-2]
![alt text][image7]

## Step 2: Remove noise from the image using Gaussian Blur Filter

![alt text][image9]
![alt text][image10]
## Get the edges in the images using Canny Edge Detector
![alt text][image11]
![alt text][image12]

## Remove edges outside of the ROI
![alt text][image13]

## Apply Probabilistic Hough Line Transform 
![alt text][image14]
![alt text][image15]
![alt text][image15-2]

## Get Weighted Lane using Weighted Arithmetic Mean
![alt text][image16]
![alt text][image17]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

## Draw Lanes on the images
![alt text][image18]
![alt text][image19]
![alt text][image20]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
