import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class Recommendation:
    def __init__(self):
        pass

    def read_file(self):
        cargo = pd.read_csv('temple dataset/cargo.csv', header=None)
        return cargo

    def Agglomerative_CompleteLink_cluster(self, X: pd.DataFrame, number_of_clusters: int):
        return AgglomerativeClustering(n_clusters=number_of_clusters, linkage='complete').fit(X)

    def show_plt(self, data, X: pd.DataFrame, title: str):
        labels = data.labels_

        colors = {0: 'rosybrown', 1: 'lightcoral', 2: 'indianred', 3: 'brown', 4: 'firebrick', 5: 'maroon',
                  6: 'darkred',
                  7: 'mistyrose', 8: 'salmon', 9: 'tomato', 10: 'darksalmon', 11: 'coral', 12: 'orangered', 13: 'olive',
                  14: 'yellow', 15: 'olivedrab', 16: 'yellowgreen', 17: 'darkolivegreen', 18: 'greenyellow',
                  19: 'chartreuse',
                  20: 'lawngreen', 21: 'honeydew', 22: 'darkseagreen', 23: 'palegreen', 24: 'lightgreen',
                  25: 'forestgreen',
                  26: 'limegreen', 27: 'darkgreen', 28: 'springgreen', 29: 'aquamarine', 30: 'lightseagreen',
                  31: 'mediumspringgreen', 32: 'mediumaquamarine', 33: 'aquamarine', 34: 'aqua', 35: 'cyan', 36: 'navy',
                  37: 'crimson', 38: 'pink', 39: 'palevioletred', 40: 'plum', 41: 'violet', 42: 'purple', 43: 'indigo',
                  44: 'darkblue', 45: 'mediumblue', 46: 'blue', 47: 'darkviolet', -1: 'black'}

        # Building the colour vector for each data point
        cvec = [colors[l] for l in labels]

        # Plotting P1 on the X-Axis and P2 on the Y-Axis
        # according to the colour vector defined
        plt.figure(figsize=(9, 9))
        plt.title(title)
        plt.scatter(X['C1'], X['C2'], c=cvec)

        plt.show()


r = Recommendation()
c = r.read_file().iloc[:100, 2:4]
c.columns = [f'C{n}' for n in range(1, 3)]
print(c.head())
c = c.dropna()
model = r.Agglomerative_CompleteLink_cluster(c, 10)
r.show_plt(model, c, 'Agglomerative Complete Link')
