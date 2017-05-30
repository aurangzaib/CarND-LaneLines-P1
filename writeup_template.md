# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

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
[image6]: ./results/03-white-1496135329.jpg "White Mask"

[image7]: ./results/04-mask-hsv-1496135309.jpg "Mask HDV"
[image8]: ./results/04-mask-hsv-1496135361.jpg "Mask HDV"

[image9]: ./results/07-filtered-1496135354.jpg "Filtered"
[image10]: ./results/07-filtered-1496135303.jpg "Filtered"

[image11]: ./results/08-canny-1496135302.jpg "Edges"
[image12]: ./results/08-canny-1496135366.jpg "Edges"

[image13]: ./results/09-ROI-1496135301.jpg "ROI"

[image16]: ./results/11-lanes-1496135307.jpg "Lanes"
[image17]: ./results/11-lanes-1496135366.jpg "Lanes"

[image18]: ./results/12-resultant-1496135312.jpg "Result"
[image19]: ./results/12-resultant-1496135332.jpg "Result"
[image20]: ./results/12-resultant-1496135364.jpg "Result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image0-1]
![alt text][image0-2]
![alt text][image0-3]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
