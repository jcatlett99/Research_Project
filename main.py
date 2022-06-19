import numpy as np
import setuptools.command.install

import high_level as highLevel
import low_level as lowLevel
import formatting as form
import clustering as cluster
import temp_file
import pandas as pd
import regression

import os
import cv2
import csv


def main():
    images = []
    csv_content = [["image",
                    "saturation",
                    "luminance",
                    "contrast",
                    "sharpness",
                    "rule_of_thirds_gridlines",
                    "rule_of_thirds_power_points",
                    "diagonal_dominance",
                    "entropy",
                    "symmetry",
                    "horizontal_vertical_line_ration",
                    "straight_diagonal_line_ratio"]]

    # emptying csv file
    file = open('output/table.csv', 'w', newline='')
    file.truncate()
    file.close()
    image_features = []
    ll_features = []
    hl_features = []
    histograms = []

    n = 1
    for item in os.listdir('moshuir_data/datasets'):
        new_row = []
        new_low_level_row = []
        new_high_level_row = []
        if not item.startswith('.'):

            filename = os.fsdecode(item)
            image = cv2.imread('moshuir_data/datasets/' + filename)

            print("Image: ", filename)
            images.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (45, 45)).flatten())
            new_row.append(filename)
            new_low_level_row.append(filename)
            new_high_level_row.append(filename)

            ll = lowLevel.LowLevelFeatures
            low_level_features = ll.run(ll, image)
            histograms.append(low_level_features[-1])
            new_low_level_row.extend(low_level_features)
            new_row.extend(low_level_features[:len(low_level_features)-1])

            hl = highLevel.HighLevelFeatures
            high_level_features = hl.run(hl, image)
            new_row.extend(high_level_features)
            new_high_level_row.extend(high_level_features)
            print(n, " / 601")
            n += 1

            image_features.append(new_row)
            ll_features.append(new_low_level_row)
            hl_features.append(new_high_level_row)
            csv_content.append(new_row)

    with open('output/table.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_content)

    # data_frame = pd.read_csv('output/table.csv', index_col=0)
    # data_frame['aesthetic'] = np.zeros(len(data_frame))
    #
    # chosen_values = form.read_csv()
    # for k in chosen_values:
    #     data_frame.loc[k]['aesthetic'] = chosen_values[k]
    # data_frame.to_csv('output/table.csv', encoding='utf-8')

    c = cluster.Cluster(image_features, ll_features, histograms, hl_features, images)
    df = c.create_clusters(7, train=True)
    df.to_csv('output/table.csv', encoding='utf-8')

    data_frame = pd.read_csv('output/table.csv', index_col=0)
    temp_file.temp(data_frame)


main()
