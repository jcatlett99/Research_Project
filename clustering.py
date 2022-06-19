import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Cluster:

    image_features = None
    # low_level = [saturation, luminance, contrast, sharpness, entropy]
    low_level = None
    # high_level = [dist(gridlines), dist(powerpoints), diagonal dominance]
    high_level = None

    hist_data = []
    images = []
    labels = []

    def __init__(self, image_features, low_level, histograms, high_level, images):

        self.image_features = image_features

        for sub_list in low_level:
            self.labels.append(sub_list[0])
        self.labels = np.array(self.labels)

        self.low_level = low_level
        self.high_level = high_level
        self.images = np.array(images)
        self.hist_data = np.array(histograms)
        # print(self.labels)

    @staticmethod
    def combine_features(features):
        s = np.delete(np.array(features), 0, 1)
        array = []
        # takes the list of image features and sums them all
        for i in range(len(s)):
            array.append(np.sum(s[i].astype(np.float64)))

        return array

    @staticmethod
    def euclidean_dist(point, centroid):
        assert len(point) == len(centroid)

        s = 0
        for i in range(len(point)):
            s += (point[i] - centroid[i]) ** 2

        return np.sqrt(s)

    def add_colour_hist(self, df):
        baseline = np.ones(len(self.hist_data[0]), dtype=np.float32)
        array = []
        for i in range(len(self.hist_data)):
            array.append(np.mean(scipy.spatial.distance.jensenshannon(baseline, self.hist_data[i])))
        df["histograms"] = array
        return df

    def create_clusters(self, k, train=False):

        data_frame = pd.read_csv('output/table.csv', index_col=0)
        data_frame_with_colours = self.add_colour_hist(data_frame)
        # data_frame_with_colours = data_frame
        # data_frame_with_colours.drop('aesthetic', axis=1)

        columns = data_frame_with_colours.columns
        # for f1 in range(len(columns)):
        #     for f2 in range(f1, len(columns)):
        #         string = columns[f1] + ':' + columns[f2]
        #         print(string)
        #         df = data_frame_with_colours.filter([columns[f1], columns[f2]], axis=1)
        #         print(df)
        string = "tsne"
        df = data_frame_with_colours
        segmentation_std = StandardScaler().fit_transform(df)

        if train:
            self.plot_pca_values(segmentation_std)

        pca = PCA(n_components=7)
        pca.fit(segmentation_std)
        pca_result = pca.transform(segmentation_std)

        if train:
            self.plot_k_values(pca_result)

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(pca_result)

        df_pca_kmeans = pd.concat([data_frame.reset_index(drop=True), pd.DataFrame(pca_result)], axis=1)
        df_pca_kmeans.columns.values[-3:] = ['Component 1', 'Component 2', 'Component 3']
        df_pca_kmeans['SKPCA'] = kmeans.labels_

        df_pca_kmeans['Segment'] = df_pca_kmeans['SKPCA'].map({
            0: 'first',
            1: 'second',
            2: 'third',
            3: 'fourth'
        })

        x_axis = df_pca_kmeans['Component 1']
        y_axis = df_pca_kmeans['Component 2']
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x_axis, y_axis, hue=df_pca_kmeans['Segment'])
        plt.savefig("output/clusters.png")

        novelty = []
        typicality = []
        centroids = kmeans.cluster_centers_
        for i in range(len(kmeans.labels_)):
            novelty.append(self.euclidean_dist(pca_result[i], centroids[kmeans.labels_[i]]))
            t = 0
            for c in centroids:
                t += self.euclidean_dist(pca_result[i], c)
            typicality.append(t / len(centroids))

        self.tsne_vis(pca_result, data_frame, string)

        data_frame_with_colours["novelty"] = novelty / max(novelty)
        data_frame_with_colours["typicality"] = typicality / max(typicality)
        return data_frame_with_colours

    def plot_pca_values(self, segmentation_std):
        pca_test = PCA()
        pca_test.fit(segmentation_std)
        plt.figure(figsize=(10, 8))
        plt.xlabel("components")
        plt.ylabel("cumulative explained variance")
        plt.plot(range(1, len(pca_test.explained_variance_ratio_.cumsum()) + 1),
                 pca_test.explained_variance_ratio_.cumsum(),
                 marker='o', linestyle='--')
        plt.savefig("output/component_variance.png")

    def plot_k_values(self, pca_result):
        # within cluster sum of squares
        wcss = []
        # go through and determine wcss for different k values
        for i in range(1, int(len(self.labels) / 2)):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(pca_result)
            wcss.append(kmeans.inertia_)

        # show wcss for all k and determine optimum k using elbow method
        plt.figure(figsize=(10, 8))
        plt.xlabel("clusters")
        plt.ylabel("wcss")
        plt.plot(range(1, int(len(self.labels)/2)), wcss, marker='o', linestyle='--')
        plt.savefig("output/k_wcss.png")

    def tsne_vis(self, pca_result, df, n):

        tsne = TSNE(n_components=2, perplexity=40.0)
        tsne_result = tsne.fit_transform(pca_result)
        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

        images = [np.reshape(i, (45, 45)) for i in self.images]
        figsize = (20, 20)
        image_zoom = 0.7

        fig, ax = plt.subplots(figsize=figsize)
        artists = []
        index = 0
        for xy, i in zip(tsne_result_scaled, images):
            x0, y0 = xy
            # img = self.frame_image(i, 30)
            img = OffsetImage(i, zoom=image_zoom)
            frame = False
            # if int(df.loc[self.labels[index]]['aesthetic']) == 1:
            #     frame = True
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=frame)
            artists.append(ax.add_artist(ab))
            index += 1
        ax.update_datalim(tsne_result_scaled)
        ax.autoscale()
        plt.savefig('output/tsne_clusters' + n + '.png')
