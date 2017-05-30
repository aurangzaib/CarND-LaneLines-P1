import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv


def perform_lane_detection():
    video_cap = cv.VideoCapture(
        '/Users/siddiqui/Documents/Projects/self-drive/CarND-LaneLines-P1/test_videos/solidYellowLeft.mp4')
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            print("cannot read frame")
            return -1
        detect_lane(frame)


def detect_lane(image):
    # image = cv.imread("test_images/solidYellowCurve2.jpg")
    image = cv.resize(image, None, fx=0.5, fy=0.6, interpolation=cv.INTER_LINEAR)
    # gray scale and noise reduction
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # define range of blue color in HSV
    cv.imshow("hsv", image_hsv)

    mask_yellow = cv.inRange(image_hsv,
                             np.array([20, 100, 100]),  # yellow lower
                             np.array([100, 255, 255]))  # yellow upper

    mask_white = cv.inRange(image_hsv,
                            np.array([0, 0, 240]),  # white lower
                            np.array([255, 255, 255]))  # white lower

    mask_hsv = cv.bitwise_or(mask_yellow, mask_white)
    cv.imshow("1- white and yellow masking", image_hsv)

    # only retain the information of the lines in the image
    masked_image = cv.bitwise_and(image_gray, mask_hsv)

    # remove noise using gaussian filter with 9x9 kernel size
    gray = cv.GaussianBlur(masked_image,  # source
                           (9, 9),  # kernel size
                           0)  # border type
    cv.imshow("2- after noise removal", masked_image)

    # find the edges
    edges = cv.Canny(gray,
                     50,  # low threshold
                     150)  # high threshold
    cv.imshow("3- canny edge detector", edges)
    width = image.shape[1]
    height = image.shape[0]

    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    vertices = np.array([[(width * 0.05, height),  # left bottom
                          (width / 4., height * .589),  # left top
                          (3 * width / 4., height * .589),  # right top
                          (width * 0.95, height)]],  # right bottom
                        dtype=np.int32)

    cv.fillPoly(mask,  # image
                vertices,  # coordinates
                ignore_mask_color)
    cv.imshow("ROI", mask)
    # remove the parts of image which are not within vertices
    masked_edges = cv.bitwise_and(edges, mask)

    # find lines using hough
    detected_lines = cv.HoughLinesP(masked_edges,  # source
                                    2,  # rho --> 1 pixel
                                    1 * np.pi / 180,  # theta in radian (1 degree)
                                    15,  # min voting
                                    10,  # min line length in pixels
                                    1  # max line gap in pixels
                                    )

    # creating a blank to draw lines on
    line_image = np.zeros((masked_edges.shape[0],
                           masked_edges.shape[1],
                           3),
                          dtype=np.uint8)

    left_line = []
    # several lines are detected
    # each line has start and end points
    print("line length before:", detected_lines.size)
    for line in detected_lines:
        for x1, y1, x2, y2 in line:
            # if x2 == x1:
            #     continue
            slope = (y2 - y1) / (x2 - x1)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    print("line length after:", len(left_line))
    for line in detected_lines:
        for x1, y1, x2, y2 in line:
            if angle != 0 and length >= 1:
                cv.line(line_image,  # source
                        (x1, y1),  # start point
                        (x2, y2),  # end point
                        (0, 0, 255),  # BGR color --> red
                        5  # line thickness
                        )

    cv.imshow("hough line", line_image)
    # gray images can't draw colored features on it
    # this is a way to do it, using dstack and weighted sum
    # color_edges = np.dstack((edges, edges, edges))
    line_edges = cv.addWeighted(line_image,  # source 1
                                0.8,  # alpha --> weight of the first array elements
                                image,  # source 2
                                1,  # beta --> weight of the second array elements
                                0)  # gamma --> scalar added to each sum

    cv.imshow("result", line_edges)
    cv.waitKey(1)


perform_lane_detection()
