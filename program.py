import numpy as np
import cv2 as cv
import time

SHOW_DEBUG_IMAGES = False


def save_image(image, name="unknown", path="result_images/"):
    stamp = str(int(time.time()))
    cv.imwrite(path + name + "-" + stamp + ".jpg", image)


def perform_lane_detection(path="test_videos/challenge.mp4"):
    video_cap = cv.VideoCapture(path)
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if not success:
            print("cannot read frame")
            return -1
        lanes_detection(frame)


def get_height_width(image):
    return image.shape[1], image.shape[0], image.shape[0] * .62


def get_masked_image(image):
    # HSV image
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # gray-scale image
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # get region of image within yellow range
    mask_yellow = cv.inRange(image_hsv,
                             np.array([20, 100, 100]),  # yellow lower
                             np.array([100, 255, 255]))  # yellow upper

    # get region of image within white range
    mask_white = cv.inRange(image_hsv,
                            np.array([0, 0, 240]),  # white lower
                            np.array([255, 255, 255]))  # white lower

    # merge results of yellow and white - bitwise OR
    mask_hsv = cv.bitwise_or(mask_yellow, mask_white)
    # only retain the information of the lanes in the image - bitwise AND
    masked_image = cv.bitwise_and(image_gray, mask_hsv)

    if SHOW_DEBUG_IMAGES:
        save_image(image_hsv, "01-hsv")
        save_image(mask_yellow, "02-yellow")
        save_image(mask_white, "03-white")
        save_image(mask_hsv, "04-mask hsv")
        save_image(mask_hsv, "05-mask hsv")
        cv.imshow("06-masked image", masked_image)
    return masked_image


def remove_image_noise(image, kernel=(3, 3)):
    filtered_image = cv.GaussianBlur(image,  # source
                                     kernel,  # kernel size
                                     0)  # border type
    if SHOW_DEBUG_IMAGES:
        save_image(filtered_image, "07-filtered")
        cv.imshow("2- after noise removal", filtered_image)
    return filtered_image


def get_edges(image, low, high):
    edges = cv.Canny(image,
                     low,  # low threshold
                     high)  # high threshold
    if SHOW_DEBUG_IMAGES:
        save_image(edges, "08-canny")
        cv.imshow("3- canny edge detector", edges)
    return edges


def get_masked_edges(width, height_bottom, height_top, edges, ignore_mask_color=255):
    # blank matrix
    mask = np.zeros_like(edges)
    # ROI vertices
    vertices = np.array([[(width * 0.05, height_bottom),  # left bottom
                          (width / 4., height_top),  # left top
                          (3 * width / 4., height_top),  # right top
                          (width * 0.95, height_bottom)]],  # right bottom
                        dtype=np.int32)
    # create ROI
    cv.fillPoly(mask,  # image
                vertices,  # coordinates
                ignore_mask_color)
    # remove the parts of image which are not within ROI - bitwise AND
    masked_edges = cv.bitwise_and(edges, mask)
    if SHOW_DEBUG_IMAGES:
        save_image(mask, "09-ROI")
        save_image(masked_edges, "10-masked_edges")
        cv.imshow("ROI", mask)
        cv.imshow("masked edges", masked_edges)

    return masked_edges


def get_hough_lines(image, rho=2, theta=1, voting=15,
                    min_line_length=10, max_line_gap=1):
    return cv.HoughLinesP(image,  # source
                          rho,
                          theta * np.pi / 180,  # degree to radian
                          voting,  # minimum voting in hough accumulator
                          min_line_length,  # min line length in pixels
                          max_line_gap  # gap b/w lines in pixels
                          )


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
        # get start and end coordinates of each line
        for x1, y1, x2, y2 in line:
            # ignore same coordinate to avoid infinite slope
            if x1 == x2:
                continue
            # slope of line
            line_slope = float(y2 - y1) / float(x2 - x1)
            # angle of deviation
            line_angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            # y intercept
            line_intercept = y1 - line_slope * x1
            # line length, used for weight
            line_length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            # in openCV, image is a matrix
            # bottom y values are higher than top y values
            # it means slope is negative for left line

            # left slope = delta_y --> -ve, delta_x --> +ve
            # right slope = delta_y --> -ve, delta_x --> -ve
            if line_slope < 0:
                left_slope_intercept.append((line_slope, line_intercept))
                left_length.append(line_length)
            else:
                right_slope_intercept.append((line_slope, line_intercept))
                right_length.append(line_length)

    # weighted mean
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    # simple arithmetic mean doesn't take into account the ...
    # ...significance of individual lines which causes fluctuation & noise
    # length of each line can be used as a weight
    # weight average --> (sum of product) / (sum of weight)
    # sop --> sum of product
    sop_left = np.dot(left_length, left_slope_intercept)
    sop_right = np.dot(right_length, right_slope_intercept)
    left_weight = np.sum(left_length)
    right_weight = np.sum(right_length)
    # handling cases when no line is detected by hough transform
    left = (sop_left / left_weight) if len(left_length) > 0 else None
    right = (sop_right / right_weight) if len(right_length) > 0 else None

    # resultant is 2 lanes out of several hough lines
    return left, right


def get_coordinates(height_bottom, height_top, left, right):
    # start and end x coordinates for left lane
    # x --> (y - intercept) / slope
    slope, intercept = left
    left_x1 = int((height_bottom - intercept) / slope)
    left_x2 = int((height_top - intercept) / slope)
    # start and end x coordinates for right lane
    slope, intercept = right
    right_x1 = int((height_bottom - intercept) / slope)
    right_x2 = int((height_top - intercept) / slope)
    # y coordinates
    y1 = int(height_bottom)
    y2 = int(height_top)

    return ((left_x1, y1), (left_x2, y2)), ((right_x1, y1), (right_x2, y2))


def draw_lanes(edges, left_coordinates, right_coordinates,
               color=(0, 0, 255), thickness=5):
    # creating a blank to draw lines on
    lane_image = np.zeros((edges.shape[0],  # --> height
                           edges.shape[1],  # --> width
                           3),
                          dtype=np.uint8)
    for coordinates in (left_coordinates, right_coordinates):
        start_coordinate, end_coordinate = coordinates
        cv.line(lane_image,  # source
                start_coordinate,  # (x1, y1)
                end_coordinate,  # (x2, y2)
                color,  # BGR color --> red
                thickness  # line thickness
                )
    save_image(lane_image, "11-lanes")
    cv.imshow("smooth lines", lane_image)
    return lane_image


def get_resultant_image(source1, source2, alpha=0.8, beta=1, gamma=0):
    resultant = cv.addWeighted(source1,
                               alpha,  # weight of the first array elements
                               source2,  # source 2
                               beta,  # weight of the second array elements
                               gamma)  # scalar added to each sum
    save_image(resultant, "12-resultant")
    return resultant


def lanes_detection(image):
    # STEP - 0: resize frame to better visualize
    image = cv.resize(image, None, fx=0.5, fy=0.6, interpolation=cv.INTER_LINEAR)
    save_image(image, "original")
    # STEP - 1: get the width and height of image
    width, height_bottom, height_top = get_height_width(image)
    # STEP - 2: get the white and yellow lanes from the image
    masked_image = get_masked_image(image)
    # STEP - 3: remove noise using gaussian filter with 9x9 kernel size
    filtered_image = remove_image_noise(masked_image, kernel=(9, 9))
    # STEP - 4: get canny edge
    edges = get_edges(filtered_image, low=50, high=150)
    # STEP - 5: remove edges outside of ROI
    masked_edges = get_masked_edges(width, height_bottom, height_top, edges)
    # STEP - 6: find lines using hough
    hough_lines = get_hough_lines(masked_edges)
    # STEP - 7: get left and right lanes from several hough lines
    left_lane, right_lane = get_weighted_lanes(hough_lines)
    # STEP - 8: get the coordinates of the left and right lanes
    left_coordinates, right_coordinates = get_coordinates(height_bottom,
                                                          height_top,
                                                          left_lane,
                                                          right_lane)
    # STEP - 9: draw lanes on a blank image
    lane_image = draw_lanes(masked_edges, left_coordinates, right_coordinates)
    # STEP - 10: draw lanes on original image
    weighted_image = get_resultant_image(lane_image, image)
    cv.imshow("result", weighted_image)
    cv.waitKey(1)


perform_lane_detection()
