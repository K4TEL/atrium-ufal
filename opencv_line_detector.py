import math
from pathlib import Path

import cv2 as cv
import numpy as np


def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line


def hough_lines(image_filename: Path, visual: bool = False):
    # Loads an image
    src = cv.imread(cv.samples.findFile(str(image_filename)), cv.IMREAD_GRAYSCALE)

    blurred = cv.GaussianBlur(src, (5, 5), 0)

    dst = cv.Canny(blurred, 100, 150, None, 5)

    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 10, None, 200, 60)
    # print(linesP)

    if visual:

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        lane_lP = lane_lines(cdstP, linesP)
        line_image = np.zeros_like(cdstP)
        for l in lane_lP:
            if l is not None:
                cv.line(line_image, (l[0]), (l[1]), (255, 0, 0), 1, cv.LINE_AA)
        cv.addWeighted(cdstP, 1.0, line_image, 1.0, 0.0)

        if len(src.shape) == 2:  # grayscale image
            src = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
        if len(cdstP.shape) == 2:  # grayscale image
            cdstP = cv.cvtColor(cdstP, cv.COLOR_GRAY2BGR)

        result_small = cv.resize(cv.hconcat([src, cdstP]), (1600, 1000), cv.INTER_AREA)

        cv.imshow(f"{image_filename.stem}", result_small)
        cv.waitKey(0)
        cv.destroyAllWindows()

    print(f"Found {len(linesP)} lines on {image_filename.stem}")
    return linesP, src.shape

def lines_lenght(lines: list) -> list:
    all_lenght = []
    xds, yds = [], []
    for l in lines:
        points = l[0]
        start = [points[0], points[1]]
        end = [points[2], points[3]]

        x_diff, y_diff = end[0] - start[0], end[1] - start[1]

        l_lenght = math.sqrt(x_diff**2 + y_diff**2)
        all_lenght.append(l_lenght)
        xds.append(abs(x_diff))
        yds.append(abs(y_diff))

    return all_lenght, xds, yds

def page_visual_analysis(image_filename: Path) -> list:

    detected_lines, img_shape = hough_lines(image_filename)

    det_line_len, det_h_diff, det_v_diff = lines_lenght(detected_lines)

    print(img_shape)



    def filter_edges(lines: list, ignore_edges: int = 50) -> list:
        res = []
        filter = []
        for i, l in enumerate(lines):
            points = l[0]
            start = [points[0], points[1]]
            end = [points[2], points[3]]

            w, h = img_shape[0], img_shape[1]

            l_lenght = 0

            if ignore_edges < end[0] < w-ignore_edges and ignore_edges < end[1] < w-ignore_edges and \
                    ignore_edges < start[0] < w-ignore_edges and ignore_edges < start[1] < h-ignore_edges:
                res.append(l)  # non edge
            else:

                if det_h_diff[i] > w / 2 and ignore_edges < end[0] < w-ignore_edges and \
                        ignore_edges < start[0] < w-ignore_edges:
                    res.append(l)  # very long horizontal

                if det_v_diff[i] > h / 2 and ignore_edges < end[1] < h-ignore_edges and \
                        ignore_edges < start[1] < h-ignore_edges:
                    res.append(l)  # very long vertical

                filter.append(l)

        print(f"From {len(lines)} found {len(res)} non-edge lines and {len(filter)} on the edge")
        return res, filter

    lines, filtered_lines = filter_edges(detected_lines, 100)
    det_line_len, horizontal_diff, vertical_diff = lines_lenght(lines)

    # print(max(det_line_len), min(det_line_len))
    # print(max(horizontal_diff), min(horizontal_diff))
    # print(max(vertical_diff), min(vertical_diff))

    long_horiz = max(horizontal_diff) > img_shape[0] / 2 if len(horizontal_diff) > 0 else False
    long_vert = max(vertical_diff) > img_shape[1] / 2 if len(vertical_diff) > 0 else False

    return long_horiz, long_vert




# fi = Path("/lnet/work/people/lutsai/pythonProject/pages_src/CTX199706756/6591569c-2e8c-4db6-a7a9-84ab997c7f34-12.png")
# hough_lines(fi)
# page_visual_analysis(fi)