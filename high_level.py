import cv2
import numpy as np
import collections
import math
from matplotlib import pyplot as plt
from sympy import symbols, Eq, solve


class HighLevelFeatures:

    width = -1
    height = -1

    diagonal_dominance = -1
    dist_gridlines = []
    dist_powerpoints = []

    entropy = -1

    diagonal_lines = 0
    horizontal_lines = 0
    vertical_lines = 0

    symmetry = -1

    def __init__(self):
        pass

    def get_symmetry(self, image, k, width, height):
        EPSILON = 0.0001
        blur_image = cv2.GaussianBlur(image, (21, 21), 0)
        c = (int(width / 2), int(height / 2))

        theta = 180 / k
        symmetry = []
        for i in range(k):
            angle = theta * (i + 1)

            if angle <= 45:
                adjacent = width / 2
                opposite = abs(adjacent * np.tan(np.deg2rad(angle)))

                x1 = width
                y1 = c[1] - opposite

                x2 = 0
                y2 = c[1] + opposite
            elif angle <= 90:
                opposite = height / 2
                adjacent = abs(opposite / np.tan(np.deg2rad(angle)))

                x1 = c[0] + adjacent
                y1 = 0

                x2 = c[0] - adjacent
                y2 = height
            elif angle <= 135:
                opposite = height / 2
                adjacent = abs(opposite / np.tan(np.deg2rad(angle)))

                x1 = c[0] - adjacent
                y1 = 0

                x2 = c[0] + adjacent
                y2 = height
            elif angle <= 180:
                adjacent = width / 2
                opposite = abs(adjacent * np.tan(np.deg2rad(angle)))

                x1 = width
                y1 = c[1] + opposite

                x2 = 0
                y2 = c[1] - opposite
            else:
                print("cannot get symmetry")
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y2 - (slope * x2)
            d = []

            cv2.line(blur_image, (int(x1), int(y1)), (int(x2), int(y2)), 255, 3)

            for y0 in range(width):
                for x0 in range(height):
                    # if x0 < x1 and
                    if abs(x1 - x2) <= EPSILON or abs(y1 - y2) <= EPSILON:
                        d.append(self.straight_axis(x1, y1, x2, y2, x0, y0, blur_image))
                    else:
                        lx = (y0 - intercept) / slope
                        ly = (slope * x0) + intercept
                        # x1 and x2
                        # y1 and y2: these are both the extremes of the line of symmetry
                        # first check if the min-x and min-y are less than x0, y0

                        min_x = min(x1, x2)
                        min_y = min(y1, y2)
                        # print("check: ", min_x, " | ", x0, " | ", lx)
                        # print("check: ", min_y, " | ", y0, " | ", ly)
                        # print("\n")
                        if min_x <= x0 <= lx or min_y <= y0 <= ly:
                            perp_slope = - (x2 - x1) / (y2 - y1)
                            perp_intercept = ((-perp_slope * x0) + y0)

                            x, y = symbols('x y')
                            eq1 = Eq((slope * x) + intercept - y)
                            eq2 = Eq((perp_slope * x) + perp_intercept - y)
                            intersection = solve((eq1, eq2), (0, 1))

                            if len(intersection) > 0:
                                x_diff = abs(intersection[0] - x0)
                                y_diff = abs(intersection[1] - y0)

                                mirror_x = intersection[x] + x_diff
                                mirror_y = intersection[y] + y_diff
                                if mirror_y < width and mirror_x < height:
                                    d.append(abs(blur_image[y0][x0] - blur_image[int(mirror_y)][int(mirror_x)]))
            filtered_d = list(filter(None, d))
            if len(filtered_d) > 0:
                symmetry.append(sum(filtered_d) / len(filtered_d))

        self.symmetry = min(symmetry)

    @staticmethod
    def straight_axis(x1, y1, x2, y2, x0, y0, image):
        if abs(x1 - x2) <= 0.0001:
            if x0 < x1:
                x_diff = abs(x1 - x0)
                mirror_x = x1 + x_diff
                return abs(image[y0][x0] - image[int(y0)][int(mirror_x) - 1])
        elif abs(y1 - y2) <= 0.0001:
            if y0 < y1:
                y_diff = abs(y1 - y0)
                mirror_y = y1 + y_diff
                return abs(image[y0][x0] - image[int(mirror_y) - 1][int(x0)])
        else:
            return

    def run(self, image):

        self.width = image.shape[1]
        self.height = image.shape[0]

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        # if we would like a *binary* map that we could process for contours,
        # compute convex hull's, extract bounding boxes, etc.,
        # we can additionally threshold the saliency map
        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # performing morphological transformation to remove small salient objects,
        # holes in larger salient objects, and connect together nearby salient
        # objects
        closing_kernel = np.ones((60, 60), np.uint8)
        opening_kernel = np.ones((20, 20), np.uint8)
        imageOPEN = cv2.morphologyEx(threshMap, cv2.MORPH_OPEN, opening_kernel)
        imageCLOSED = cv2.morphologyEx(imageOPEN, cv2.MORPH_CLOSE, closing_kernel)

        # getting coordinates of perimeter of areas of interest
        # also called 'salient areas'
        contours, hierarchy = cv2.findContours(imageCLOSED, 1, 2)

        # getting the area of the contours for comparison of size
        # getting centre of mass of the contours
        # putting the contours into a dictionary with area as key
        image_contours = {}
        for c in contours:
            area = cv2.contourArea(c)
            moment = cv2.moments(c)
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])

            image_contours[area] = (c, (cx, cy))

        # defining the gridlines and powerpoints for Rule of Thirds
        power_points = [(int(self.width / 3), int(self.height / 3)),                                # top left
                        (int((2 * self.width) / 3), int(self.height / 3)),                          # top right
                        (int(self.width / 3), int((2 * self.height) / 3)),                          # bottom left
                        (int((2 * self.width) / 3), int((2 * self.height) / 3))]                    # bottom right

        grid_lines = [((0, int(self.height/3)), (int(self.width), int(self.height/3))),             # horizontal top
                      ((0, int((2*self.height)/3)), (int(self.width), int((2*self.height)/3))),     # horizontal bottom
                      ((int(self.width/3), 0), (int(self.width/3), int(self.height))),              # vertical left
                      ((int((2*self.width)/3), 0), (int((2*self.width)/3), int(self.height)))]      # vertical right

        # contours are ordered in ascending order of area (the largest object is at the end)
        sorted_contours = collections.OrderedDict(sorted(image_contours.items()))
        # creating an image with only contours
        # (black with lines where salient objects are on original image)
        contour_mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(contour_mask, contours, -1, (0, 255, 0), 1)

        self.dist_gridlines = []
        self.dist_powerpoints = []

        for s in reversed(sorted_contours.values()):

            # get minimum area bounding box and the coordinates of the four corners
            rect = cv2.minAreaRect(s[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # check if it is longer than it is wide, determining the orientation
            # get the angle the longer side of the rectangle
            # this provides the visually dominant orientation
            if math.dist(box[3], box[2]) > math.dist(box[1], box[2]):
                p1 = self.midpoint(box[1], box[2])
                p2 = self.midpoint(box[0], box[3])
            else:
                p1 = self.midpoint(box[0], box[1])
                p2 = self.midpoint(box[3], box[2])

            # getting an empty (black) image with only the orientation
            # lines of this current salient object
            orientation_line_mask = np.zeros(image.shape, np.uint8)
            cv2.line(orientation_line_mask, p1, p2, (0, 255, 0), 3)

            # bitwise AND of the two masks, so we are left with just the end
            # points of the orientation lines of this current salient object
            masked_points = cv2.cvtColor(cv2.bitwise_and(contour_mask, orientation_line_mask), cv2.COLOR_BGR2GRAY)
            orientation_points = cv2.findNonZero(masked_points, idx=None)

            bounding_point_1 = (orientation_points[0][0][0],
                                orientation_points[0][0][1])
            bounding_point_2 = (orientation_points[len(orientation_points) - 1][0][0],
                                orientation_points[len(orientation_points) - 1][0][1])

            theta = np.rad2deg(np.arctan(abs(bounding_point_1[1] - bounding_point_2[1]) /
                                         abs(bounding_point_1[0] - bounding_point_2[0])))

            # getting the number of horizontal, vertical,
            # and diagonal lines to compare as ratios to each other
            if theta < 3:
                self.horizontal_lines += 1
            elif 87 < theta < 93:
                self.vertical_lines += 1
            else:
                self.diagonal_lines += 1

            # checking the angle of the orientation line
            # if 'more' horizontal, only check the horizontal grid lines
            # if 'more' vertical, only check the vertical grid lines
            # small bit faster, but also ensures short horizontal lines don't get paired with a vertical
            # grid line since it is on average closer
            gridline_distance = self.width
            if theta > 45:
                gridline_distance = min(min(((abs(bounding_point_1[0] - grid_lines[2][0][0])
                                              + abs(bounding_point_2[0] - grid_lines[2][0][0]))/2),

                                            ((abs(bounding_point_1[0] - grid_lines[3][0][0])
                                              + abs(bounding_point_2[0] - grid_lines[3][0][0]))/2)), gridline_distance)
            else:
                gridline_distance = min(min(((abs(bounding_point_1[1] - grid_lines[0][0][1])
                                              + abs(bounding_point_2[1] - grid_lines[0][0][1]))/2),

                                            ((abs(bounding_point_1[1] - grid_lines[1][0][1])
                                              + abs(bounding_point_2[1] - grid_lines[1][0][1]))/2)), gridline_distance)
            self.dist_gridlines.append(gridline_distance)

            powerpoint_distance = self.width
            for i in range(len(power_points)):
                powerpoint_distance = min(math.dist(power_points[i], s[1]), powerpoint_distance)
            self.dist_powerpoints.append(powerpoint_distance)

        # making a filled binary mask of the salient objects
        filled_contour_mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(filled_contour_mask, contours, -1, (255, 255, 255), 1)
        cv2.floodFill(filled_contour_mask, None, (0, 0), (255, 255, 255))
        filled_contour_mask = cv2.bitwise_not(filled_contour_mask)

        # making a mask of the diagonal from top left to right
        # getting to total number of pixels in both the contour and diagonal masks
        diagonal_mask = np.zeros(image.shape, np.uint8)
        cv2.line(diagonal_mask, (0, 0), (self.width, self.height), (255, 255, 255), 1)
        masked_diagonal_points = cv2.cvtColor(cv2.bitwise_and(filled_contour_mask, diagonal_mask), cv2.COLOR_BGR2GRAY)
        diagonal_left_right = cv2.countNonZero(masked_diagonal_points)

        # making a mask of the diagonal from top right to left
        # getting to total number of pixels in both the contour and diagonal masks
        diagonal_mask = np.zeros(image.shape, np.uint8)
        cv2.line(diagonal_mask, (self.width, 0), (0, self.height), (255, 255, 255), 1)
        masked_diagonal_points = cv2.cvtColor(cv2.bitwise_and(filled_contour_mask, diagonal_mask), cv2.COLOR_BGR2GRAY)
        diagonal_right_left = cv2.countNonZero(masked_diagonal_points)

        # getting which diagonal has the most pixels giving the highest diagonal dominance
        self.diagonal_dominance = max(diagonal_left_right, diagonal_right_left)

        # converting the image to greyscale
        imageGREY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # marg is the marginal distribution of the greyscale image
        # bins corresponds to the bitwise quality, e.g. 256 -> 8 bits
        # NOTE: I've tried other bin values (1024, 2048, 1920) and they all gave the same result
        # NOTE: Giving a lower value of bins (128) lowered the entropy values proportionally
        # you filter out the probabilities that are equal to 0
        # finally you apply the Shannon Entropy formula to get the
        # entropy of the image
        marg = np.histogramdd(np.ravel(imageGREY), bins=256)[0] / imageGREY.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        self.entropy = -np.sum(np.multiply(marg, np.log2(marg)))

        # resize image to speed up measurement
        scale_percent = 40  # percent of original size
        scaled_width = int(image.shape[1] * scale_percent / 100)
        scaled_height = int(image.shape[0] * scale_percent / 100)
        dim = (scaled_width, scaled_height)

        resized = cv2.resize(imageGREY, dim, interpolation=cv2.INTER_AREA)

        self.get_symmetry(self, resized, 2, scaled_width, scaled_height)

        # performing fourier transform on image
        # lower spatial frequencies have been correlated with faster speed
        # of consumption and perception, as well as linked ot higher
        # ratings of aesthetics
        # print(np.fft.fft2(imageGREY))
        # imageFOURIER = np.fft.fftshift(np.fft.fft2(imageGREY))
        # # plotting magnitude spectrum of image
        # # the more white near the centre means more low-spatial frequencies
        #
        # plt.figure(num=None, figsize=(8, 6), dpi=80)
        # plt.imshow(150*np.log(abs(imageFOURIER)), cmap='gray')
        # plt.show()
        if self.horizontal_lines == 0:
            horizontal_vertical_ratio = 0
        elif self.vertical_lines == 0:
            horizontal_vertical_ratio = self.horizontal_lines
        else:
            horizontal_vertical_ratio = self.horizontal_lines / self.vertical_lines

        if self.diagonal_lines == 0:
            diagonal_ratio = (self.vertical_lines + self.horizontal_lines)
        else:
            diagonal_ratio = (self.vertical_lines + self.horizontal_lines) / self.diagonal_lines

        return [np.min(self.dist_gridlines),
                np.min(self.dist_powerpoints),
                self.diagonal_dominance,
                self.entropy,
                self.symmetry,
                horizontal_vertical_ratio,
                diagonal_ratio]

    @staticmethod
    def midpoint(p1, p2):
        return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
