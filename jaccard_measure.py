import numpy as np
import pickle
from scipy.linalg import norm
from itertools import permutations
from numpy.linalg import inv
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import copy

np.set_printoptions(suppress = True)

def plot_data(X, y, out_file = None, title = None ,show = True):
    """ Simple function to plot your 2D results ! """

    yunique = np.unique(y)

    for yu in yunique:
        Xi = X[y == yu]
        plt.scatter(Xi[:,0],Xi[:,1],label=str(yu))
    plt.legend(loc="best")

    if title is not None:
        plt.title(title)
    if show is True:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file)

    plt.clf()


class JACCARD:
    
    def __init__(self, y1, y2):
        """ 
        ~~~~~~~~ -------- ~~~~~~~~ --------
        > y1 is the reference
        > y2 is the infered labels
        > X is useless !
        ~~~~~~~~ -------- ~~~~~~~~ --------
        """
        assert len(y1) == len(y2)
        #assert len(y1[y1 < 0]) == 0 and len(y2[y2 < 0]) == 0 # no outliers !

        self.y1 = y1
        self.y2 = y2
        self.define_variables()

    def define_variables(self):

        self.unique_y1 = np.unique(self.y1)
        self.unique_y2 = np.unique(self.y2)
        #print("number of target cluster : ",len(self.unique_y1))
        #print("number of predicted cluster : ",len(self.unique_y2)) 

        self.n_sample = len(self.y1)
        self.J = None
        self.random_baseline = None
        tmp = np.arange(self.n_sample, dtype=int)

        self.elem1 = {}
        for y in self.unique_y1:
            self.elem1[y] = set(tmp[(self.y1 == y)]) # storing everything as sets 

        self.elem2 = {}
        for y in self.unique_y2:
            self.elem2[y] = set(tmp[(self.y2 == y)])

    def measure_random_baseline(self):
        """ Measure the jaccard index matrix if random assignments are given """

        jtmp = np.zeros((len(self.unique_y2), len(self.unique_y1)))
        n_sample = 50

        for i in range(n_sample):
            y_tmp = np.arange(0, len(self.unique_y2), dtype=int) # making sure there is at least one of each !
            y_random = np.random.randint(0, len(self.unique_y2),len(self.y1)) 
            y_random[:len(self.unique_y2)] = y_tmp
            tmp_object = JACCARD(self.y1, y_random)
            tmp_object.compute_Jaccard_matrix()
            jtmp += tmp_object.J

        self.random_baseline = jtmp*(1/n_sample)     # --->  each element has a different random value

    def compute_Jaccard_matrix(self):
        """ Computes jaccard index matrix """

        cat_label_1 = self.unique_y1
        cat_label_2 = self.unique_y2
        member1 = self.elem1
        member2 = self.elem2

        # for every infered clusters, check their Jaccard similarity to every other cluster
        self.J = np.zeros((len(cat_label_2), len(cat_label_1)))
        for i, c2 in enumerate(cat_label_2):
            for j, c1 in enumerate(cat_label_1):
                self.J[i,j] = 1.0*len(member1[c1].intersection(member2[c2]))/len(member1[c1].union(member2[c2]))

    def measure_distance(self):
        """ Normalized similarity measure based on Jaccard measures ~ (L1 - L0)
        """
        # measure random case (completely random assignments) 
        self.compute_Jaccard_matrix()
        self.measure_random_baseline()
        self.l1 = np.sum(self.J)
        self.l0 = np.count_nonzero( (self.J - self.random_baseline) > -1e-8) # count the number of elements above random baseline !
        self.dist = self.l1 - self.l0

        return self

def main():
    """ Some trivial example , try and play with the DBSCAN parameters to see how the measure changes """

    np.random.seed(12)
    X, y_true = make_blobs(n_samples = 5000, n_features=2, centers = 10)

    plot_data(X, y_true)

    e = 0.5
    dbscan = DBSCAN(eps=e, min_samples=40)
    y_pred = dbscan.fit_predict(X)
    #pos_out = (y_pred > -0.0001)
    #X_tmp = X[pos_out]
    #y_true_tmp = y_true[pos_out]
    #y_pred_tmp = y_pred[pos_out]
    #info = JACCARD(y_true_tmp, y_pred_tmp).measure_distance()
    info = JACCARD(y_true, y_pred).measure_distance()
    
    print("score = ", info.dist)
    plot_data(X, y_pred, title = "score = %.3f, eps = %.2f" % (info.dist, e))
   

if __name__ == "__main__":
    main()