import numpy as np
import cv2
from matplotlib import pyplot as plt

from skimage.filters.rank import entropy as ski_entropy
from skimage.morphology import disk


class LowLevelFeatures:

    saturation = -1
    luminance = -1
    contrast = -1
    sharpness = -1

    colour_histogram = None


    def __init__(self):
        pass

    def run(self, image):

        # convert image to HSV (hue, saturation, value)
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = []
        lum = []
        # get all saturation and value (luminance/brightness) values
        for n in range(len(imageHSV)):
            sat.append(imageHSV[:, n][:, 1])
            lum.append(imageHSV[:, n][:, 2])

        # get average value of saturation and luminance across whole image
        self.saturation = np.mean(sat)
        self.luminance = np.mean(lum)

        # print("saturation: ", self.saturation, ", luminance: ", self.luminance)

        # getting Michelson contrast
        # convert image to YUV that preserves brightness values
        imageYUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[:, :, 0]

        # flatten 2D image into 1D array
        # sort the array and get the highest/lowest 5% of values
        result = np.sort(imageYUV.flatten())
        highest = result[-int(np.floor((len(image) + len(image[0])) * .05)):]
        lowest = result[:int(np.floor((len(image) + len(image[0])) * .05))]

        # contrast will be the difference between the average of these percentiles
        self.contrast = (np.mean(highest) - np.mean(lowest))

        # below is the RMS contrast
        # imageGREY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # self.contrast = imageGREY.std()

        # print("contrast: ", self.contrast)

        # getting all the colour channels of the RGB image
        # then displaying the colour histograms with pyplot graphs

        self.colour_histogram = cv2.calcHist([imageHSV], [0], None, [256], [0, 256])

        # imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # hist1 = cv2.calcHist([imageRGB], [0], None, [256], [0, 256])
        # hist2 = cv2.calcHist([imageRGB], [1], None, [256], [0, 256])
        # hist3 = cv2.calcHist([imageRGB], [2], None, [256], [0, 256])
        # self.colour_histogram = (hist1, hist2, hist3)

        # plt.subplot(221), plt.imshow(imageRGB)
        # plt.subplot(222), plt.plot(self.colour_histogram, color='red')
        # plt.xlim([0, 256])
        # plt.show()

        # converting the image to greyscale
        imageGREY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # getting the edges of the image by applying a Laplacian filter
        imageEDGES = cv2.Laplacian(imageGREY, cv2.CV_64F)
        # converting the edges into pyplot representable images
        # edgePIC = cv2.convertScaleAbs(imageEDGES)
        # plt.imshow(edgePIC)
        # plt.show()

        self.sharpness = imageEDGES.var()
        # print("sharpness: ", self.sharpness)

        # this line calculates a visual representation of the entropy
        # it uses a disk (or kernel) of 5. This value can be changed
        # to affect the granularity of the measurements made

        # imageENTROPY = ski_entropy(imageGREY, disk(10))
        # plt.figure(num=None, figsize=(8, 6), dpi=80)
        # plt.imshow(imageENTROPY, cmap='magma')
        # plt.show()

        # below is an attempt to implement hough transforms to detect curved the lines
        # the first block only detects circles, the second detects straight or splines
        # both the straight and spline results look almost identical to the eye though
        # there might be a granularity problem though since the variables are most likely
        # fitted to much smaller (resolution) images

        # imageBLUR = cv2.medianBlur(imageGREY, 5)
        # # Apply hough transform on the image
        # circles = cv2.HoughCircles(imageBLUR, cv2.HOUGH_GRADIENT, 1, image.shape[0] / 64, param1=200, param2=10,
        #                            minRadius=50, maxRadius=100)
        # # Draw detected circles
        # if circles is not None:
        #     circles = np.uint16(np.around(circles))
        #     for i in circles[0, :]:
        #         # Draw outer circle
        #         cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         # Draw inner circle
        #         cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        #
        # cv2.imshow("image", image)
        # cv2.waitKey()

        # edges = cv2.Canny(imageGREY, 150, 200, apertureSize=3)
        # minLineLength = 30
        # maxLineGap = 1
        # lines = cv2.HoughLinesP(edges, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 30, minLineLength, maxLineGap)
        # for x in range(0, len(lines)):
        #     for x1, y1, x2, y2 in lines[x]:
        #         cv2.line(image,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
        #         pts = np.array([[x1, y1], [x2, y2]], np.int32)
        #         # cv2.polylines(image, [pts], True, (0, 255, 0))
        #
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image, "Tracks Detected", (500, 250), font, 0.5, 255)
        # cv2.imshow("Trolley_Problem_Result", image)
        # cv2.imshow('edge', edges)
        # cv2.waitKey()
        return self.saturation, self.luminance, self.contrast, self.sharpness, self.colour_histogram
