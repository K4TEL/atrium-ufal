import math
import mahotas
import cv2 as cv
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from common_utils import *
from pylab import imshow, gray, show


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


def process_image(image_path, N):
    # Read the image
    image = cv.imread(cv.samples.findFile(str(image_path)), cv.IMREAD_GRAYSCALE)
    # 2. Get a histogram of colors
    # color_hist = cv.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    #       cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) ->hist
    # cv.imshow('src', image)
    # cv.waitKey(1)
    #
    plt.plot(hist)
    plt.show()
    #
    # cv.destroyAllWindows()

    print(hist)
    print(len(hist))

    bright = hist[128:, :]
    dark = hist[:128, :]

    print(len(bright), len(dark))

    print(np.argmin(dark), np.argmax(dark))
    print(np.argmin(bright)+128, np.argmax(bright)+128)


    # 3. From 4 corners of the image, get the mean color of square regions with side N
    h, w = image.shape

    # Top-left corner
    tl_corner = image[:N, :N]
    tl_mean = np.mean(tl_corner, axis=(0, 1))

    # Top-right corner
    tr_corner = image[:N, w - N:]
    tr_mean = np.mean(tr_corner, axis=(0, 1))

    # Bottom-left corner
    bl_corner = image[h - N:, :N]
    bl_mean = np.mean(bl_corner, axis=(0, 1))

    # Bottom-right corner
    br_corner = image[h - N:, w - N:]
    br_mean = np.mean(br_corner, axis=(0, 1))

    print(tl_corner)

    img_np = np.array(image)
    print(img_np)
    img_np = img_np.flatten()
    print(img_np)
    img_np = np.where(128 < img_np < 128 + int(np.argmin(bright)), 255, img_np)

    img_np = img_np.reshape(image.shape)

    img.putdata(img_np.flatten())

    img.save('img_thresholded.png')

    raise NotImplementedError

    r, image = cv.threshold(image, int(np.argmin(dark)), 255, cv.THRESH_TOZERO)
    r, image = cv.threshold(image, int(np.argmin(bright)) + 128, 255, cv.THRESH_TRUNC)
    # cv.imshow("Thresholded Image", image)
    # cv.waitKey(1)

    # Clean up windows
    # cv.destroyAllWindows()

    # raise NotImplementedError
    return image

    corner_means = {
        'Top Left': tl_mean,
        'Top Right': tr_mean,
        'Bottom Left': bl_mean,
        'Bottom Right': br_mean
    }

    # 4. Out of the 4 corner colors, pick the dimmest and the brightest values
    dimmest = min(corner_means, key=lambda k: np.linalg.norm(corner_means[k]))
    brightest = max(corner_means, key=lambda k: np.linalg.norm(corner_means[k]))

    print(f"Mean colors of 4 corners: {corner_means}")
    print(f"Dimmest corner: {dimmest}, Mean Color: {corner_means[dimmest]}")
    print(f"Brightest corner: {brightest}, Mean Color: {corner_means[brightest]}")

    # 5. Adjust gamma so that the dimmest corner turns white
    # Approximate gamma to make the dimmest corner white
    dimmest_mean_color = corner_means[dimmest]
    target_color = np.array([255, 255, 255])  # White target

    # Calculate gamma by comparing the norm of dimmest color to the white color intensity
    dimmest_norm = np.linalg.norm(dimmest_mean_color)
    white_norm = np.linalg.norm(target_color)
    gamma_value = white_norm / dimmest_norm

    # Apply gamma correction to the whole image
    gamma_corrected_image = adjust_gamma(image, gamma=gamma_value)

    # Display the gamma corrected image
    cv.imshow("Gamma Corrected Image", gamma_corrected_image)
    cv.waitKey(1)

    # 6. Apply cv.THRESH_TOZERO to set gray pixels to black
    # First convert to grayscale
    gamma_corrected_gray = cv.cvtColor(gamma_corrected_image, cv.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresholded_image = cv.threshold(gamma_corrected_gray, 127, 255, cv.THRESH_TOZERO)
    _, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

    # Display the thresholded image
    # cv.imshow("Thresholded Image", thresholded_image)
    # cv.waitKey(1)

    # Clean up windows
    cv.destroyAllWindows()


fname = "MTX201891173-3"

# Example usage:
fi = Path(f"/home/lutsai/personal_work_ms/pythonProject/pages/train/TEXT/{fname}.png")

with Image.open(fi) as img:
    img.load()

img.show()

image = cv.imread(cv.samples.findFile(str(fi)), cv.IMREAD_GRAYSCALE)
hist = cv.calcHist([image], [0], None, [256], [0, 256])
#       cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) ->hist
# cv.imshow('src', image)
# cv.waitKey(1)
#
# plt.plot(hist)
# plt.show()
#
# cv.destroyAllWindows()

# print(hist)
# print(len(hist))

# bright = hist[128:, :]
# dark = hist[:128, :]
#
# print(len(bright), len(dark))
#
# print(np.argmin(dark), np.argmax(dark))
# print(np.argmin(bright)+128, np.argmax(bright)+128)
# img = img.point(
#     lambda x: 0 if x < np.argmin(dark) else x
#  )
#
# img = img.point(
#     lambda x: 255 if x > np.argmax(bright)+120 else x
#  )
# img.show()

# photo = mahotas.load(str(fi), as_grey=True)
# photo = photo.astype(np.uint8)
img = mahotas.imread(fi, True)
img = img.astype(np.uint8)

T_rc = mahotas.rc(img)
print(T_rc)
gray()
timg = (img > T_rc)
imshow(timg)
# show()
fi = f"/home/lutsai/personal_work_ms/pythonProject/pages/train/TEXT/rc-{fname}.png"
mahotas.imsave(fi, timg)
T_rc = mahotas.otsu(img)
print(T_rc)
gray()
timg = (img > T_rc)
imshow(timg)
# show()
fi = f"/home/lutsai/personal_work_ms/pythonProject/pages/train/TEXT/otsu-{fname}.png"
mahotas.imsave(fi, timg)
# timg = mahotas.thresholding.bernsen(img, 10, 200)
# gray()
# imshow(timg)
# show()
#
# fi = "/home/lutsai/personal_work_ms/pythonProject/pages/train/TEXT/rc_P043_00002_003-3.png"
# mahotas.imsave(fi, timg)




# img.save('img_thresholded.png')

# fi = "/lnet/work/people/lutsai/pythonProject/OCR/ltp-ocr/OCR_demo/img_thresholded.png"
# image = cv.imread(cv.samples.findFile(str(fi)), cv.IMREAD_GRAYSCALE)
#
#
# hist = cv.calcHist([image], [0], None, [256], [0, 256])
#
# plt.plot(hist)
# plt.show()


# img = process_image(fi, 20)  # Adjust 'image.jpg' with your image path and N with your desired side length
#
# cv.imshow("Thresholded Image", img)
# cv.waitKey(0)
#
# # Clean up windows
# cv.destroyAllWindows()

# hough_lines(fi)
# page_visual_analysis(fi)
