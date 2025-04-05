import math
import cv2 as cv
import numpy as np

from common_utils import *


# hough line detection of long and short lines, plotting of the results
def hough_lines(image_filename: Path, output_filename: Path, visual: bool = True) -> (list, list, (int, int)):
    # Loads an image
    src = cv.imread(cv.samples.findFile(str(image_filename)), cv.IMREAD_GRAYSCALE)

    blurred = cv.GaussianBlur(src, (5, 5), 0)

    dst = cv.Canny(blurred, 100, 150, None, 5)

    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    long_lines = cv.HoughLinesP(dst, 1, np.pi / 180, 10, None, 200, 60)

    short_lines = cv.HoughLinesP(dst, 1, np.pi / 180, 10, None, 50, 10)

    if long_lines is not None:
        long_lines, _ = filter_edges(list(long_lines), src.shape, 100)
    else:
        long_lines = []

    if short_lines is not None:
        short_lines, _ = filter_edges(list(short_lines), src.shape, 100)
    else:
        short_lines = []

    if visual:
        if long_lines is not None:
            for i in range(0, len(long_lines)):
                l = long_lines[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

        if short_lines is not None:
            for i in range(0, len(short_lines)):
                l = short_lines[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 2, cv.LINE_AA)

        if len(src.shape) == 2:  # grayscale image
            src = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
        if len(cdstP.shape) == 2:  # grayscale image
            cdstP = cv.cvtColor(cdstP, cv.COLOR_GRAY2BGR)

        result_small = cv.resize(cv.hconcat([src, cdstP]), (1600, 1000), cv.INTER_AREA)

        if not output_filename.is_file():
            cv.imwrite(str(output_filename), result_small)
            print(f"[ + IMG ] \t{output_filename.stem} detected lines plot saved to {output_filename.parent}")

        # cv.imshow(f"{image_filename.stem}", result_small)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    # print(f"\tFound {len(linesP)} lines")
    return long_lines, short_lines, src.shape


# get length and coord differences of the lines
def lines_len_diff(lines: list) -> (list, list, list):
    all_lenght, xds, yds = [], [], []
    for l in lines:
        points = l[0]
        start, end = [points[0], points[1]], [points[2], points[3]]
        x_diff, y_diff = end[0] - start[0], end[1] - start[1]
        l_lenght = math.sqrt(x_diff**2 + y_diff**2)

        all_lenght.append(l_lenght)
        xds.append(abs(x_diff))
        yds.append(abs(y_diff))

    return all_lenght, xds, yds


# filter lines on the edges of the page in specified margin size
def filter_edges(lines: list, img_shape: (int, int), ignore_edges: int = 50) -> (bool, list):
    det_line_len, det_h_diff, det_v_diff = lines_len_diff(lines)
    filtered, res = [], []
    for i, l in enumerate(lines):
        points = l[0]
        start, end = [points[0], points[1]], [points[2], points[3]]
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

            filtered.append(l)

    print(f"[ IMG ]\t{len(res)} non-edge and {len(filtered)} edge lines among the total {len(lines)}")
    return res, filtered


# calling lines detection and post-processing its results
def page_visual_analysis(image_filename: Path, output_filename: Path) -> (bool, bool,  bool, int, int):
    long_lines, short_lines, img_shape = hough_lines(image_filename, output_filename)

    det_line_len, horizontal_diff, vertical_diff = lines_len_diff(long_lines)

    # print(max(det_line_len), min(det_line_len))
    # print(max(horizontal_diff), min(horizontal_diff))
    # print(max(vertical_diff), min(vertical_diff))

    long_horiz = max(horizontal_diff) > img_shape[0] / 2 if len(horizontal_diff) > 0 else False
    long_verts = len([ld for ld in vertical_diff if ld > img_shape[1] / 2]) if len(vertical_diff) > 0 else False
    pictures = len(long_lines) > 1000
    return long_horiz, long_verts, pictures, len(long_lines), len(short_lines)

# TODO
def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv.LUT(image, table)


