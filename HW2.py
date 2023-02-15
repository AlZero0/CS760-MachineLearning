import pandas as pd
import numpy as np
import math

# Q1 - Start

D1 = pd.read_table('data/D1.txt', delimiter = ' ', header = None)
D1 = D1.to_numpy()


class Node:
    def __init__(self, fea, threshold, GR):
        self.fea = fea
        self.threshold = threshold
        self.max_GR = max_GR
        self.left = None
        self.right = None


def split(D, node):
    fea = node.fea
    threshold = node.threshold
    max_GR = node.max_GR
    # split whole dataset based on optimal threshold for given feature
    d1 = D[D[:,fea] >= threshold]
    d2 = D[D[:, fea] < threshold]
    return (d1,d2)  # return the two pieces of the original dataset


def DetermineCandidateSplits(D):
    """ Pass a 2D array D and get possible splits for each feature"""
    X = D[:,:-1]
    y = D[:, -1]
    numvars = X.shape[1]
    C = []
    for j in range(numvars):
        xj = X[:,j]
        xj_sorted = sorted(xj)
        C.append(xj_sorted) # since we use all sorted values as possible splits
    return C


def entropy(y_vector):
    p1 = np.mean(y_vector)
    p0 = 1-p1
    if p1*p0 > 0:
        return -p0*math.log2(p0) - p1*math.log2(p1)
    else:
        return 0


def GainRatio(D, fea, threshold):  # x_fea >= threshold;  fea = 0,1; threshold = x_0_1,x_0_2,...,x_1_100
    count = len(D)
    d1 = D[D[:, fea] >= threshold]
    d2 = D[D[:, fea] < threshold]  # d2 is the portion of the dataset that is below the thresh
    if len(d1) == 0 or len(d2) == 0: return 0  # it's not informative if all labels are on one side

    infogain = entropy(D[:, -1]) - (len(d1) / count * entropy(d1[:, -1]) + len(d2[:, -1]) / count * entropy(d2[:, -1]))

    x_Split_mask = D[:, fea] >= threshold
    entropy_Split = entropy(x_Split_mask)

    if entropy_Split > 0:
        return infogain / entropy_Split
    else:
        return 0  # skip trivial splits


def FindBestSplit(D, C):
    y = D[:, -1]
    c = len(D)  # number of rows
    c1 = sum(y)  # count of 1's in dataset
    if c1 == c: return (1, None, 0)  # if number of 1's equals the totality of observations
    if c1 == 0: return (0, None, 0)  # if count of 1's is 0

    fea_vec = [0, 1]
    GR = [[GainRatio(D, f, t) for t in C[f]] for f in fea_vec]
    # [up]: calculate info gain for each threshold in list of thresholds C in feature f
    GR = np.array(GR)
    max_GR = max(max(i) for i in GR)
    if max_GR == 0:  # if max Gain Ratio is 0 i.e. the feature is useless no matter the threshold
        if c1 >= c - c1:  # if count of 1's is more than half of all rows
            return (1, None, max_GR)
        else:
            return (0, None, max_GR)
    # [up]: if feature is useless, label all this (sub)dataset as the majority class
    ind = np.unravel_index(np.argmax(GR, axis=None), GR.shape)
    fea = fea_vec[ind[0]]  # optimal feature to split on
    threshold_opt_ix = ind[1]
    threshold = C[fea][threshold_opt_ix]  # optimal threshold value for the split

    return (fea, threshold, max_GR)  # if feature is not useless, tell me optimal threshold and max GR


def MakeSubtree(D, node):
    C = DetermineCandidateSplits(D)
    d1,d2 = split(D, node)
    f1, t1, GR1 = FindBestSplit(d1,C)  # split d1 now
    f2, t2, GR2 = FindBestSplit(d2,C)  # split d2 now
    if t1 == None: node.left = f1 #if thresh is None (bc all labels same or useless fea) don't split
    else:
        node.left = Node(f1,t1,GR1)
        MakeSubtree(d1, node.left)  # call create tree again, with d1 now vs whole dataset
    if t2 == None: node.right = f2
    else:
        node.right = Node(f2,t2,GR2)
        MakeSubtree(d2, node.right)


# Initializing root node
C = DetermineCandidateSplits(D1)

fea_vec = [0,1]
GR = [[GainRatio(D1, fea, t) for t in C[fea]] for fea in fea_vec]
GR = np.array(GR)
ind = np.unravel_index(np.argmax(GR, axis=None), GR.shape)

fea_opt = fea_vec[ind[0]]
threshold_opt_ix = ind[1]
threshold_opt = C[fea_opt][threshold_opt_ix]
max_GR = max(max(i) for i in GR)

root = Node(fea_opt, threshold_opt, max_GR)
MakeSubtree(D1, root)


# Q1 - End

import copy
def printPreorder(root):
    #global num_indent
    s1 = [root]
    #num_spaces = ""
    while s1:
        s2 = copy.deepcopy(s1)
        s1 = []
        spaces = ""
        for n in s2:
            #num_spaces = num_spaces + " "
            if n != 0 and n != 1:
                # First print the data of node
                print(f'if x{n.fea+1} >= {n.threshold} (GainRatio = {n.max_GR})'),
                spaces = "" + " "

                if n.left != None:
                    #s1 += [n.left]
                    # Then recur on left child
                    printPreorder(n.left)

                if n.right != None:
                    #s1 += [n.right]
                    # Finally recur on right child
                    printPreorder(n.right)
            else:
                print("return ", n)
        print(spaces, "else")

printPreorder(root)


# Q2.2

from plotnine import ggplot, aes, geom_point, ggsave

Q2 = pd.read_table('data/Q2.txt', delimiter = ' ', header = None, names=["x1", "x2", "y"])
Q2plot = ggplot(Q2, aes("x1", "x2")) + geom_point(aes(color="y"))
ggsave(filename='Q2plot.png', plot=Q2plot)


# Q2.3

Druns = pd.read_table('data/Druns.txt', delimiter = ' ', header = None)
Druns = Druns.to_numpy()

# Initializing root node
C = DetermineCandidateSplits(Druns)

fea_vec = [0,1]
GR = [[GainRatio(Druns, fea, t) for t in C[fea]] for fea in fea_vec]
GR = np.array(GR)
ind = np.unravel_index(np.argmax(GR, axis=None), GR.shape)

fea_opt = fea_vec[ind[0]]
threshold_opt_ix = ind[1]
threshold_opt = C[fea_opt][threshold_opt_ix]
max_GR = max(max(i) for i in GR)

root = Node(fea_opt, threshold_opt, max_GR)
MakeSubtree(Druns, root)

printPreorder(root)


# Q2.4

D3 = pd.read_table('data/D3leaves.txt', delimiter = ' ', header = None)
D3 = D3.to_numpy()

# Initializing root node
C = DetermineCandidateSplits(D3)

fea_vec = [0,1]
GR = [[GainRatio(D3, fea, t) for t in C[fea]] for fea in fea_vec]
GR = np.array(GR)
ind = np.unravel_index(np.argmax(GR, axis=None), GR.shape)

fea_opt = fea_vec[ind[0]]
threshold_opt_ix = ind[1]
threshold_opt = C[fea_opt][threshold_opt_ix]
max_GR = max(max(i) for i in GR)

root = Node(fea_opt, threshold_opt, max_GR)
MakeSubtree(D3, root)

printPreorder(root)


# Q2.5.3

D2 = pd.read_table('data/D2.txt', delimiter = ' ', header = None)
D2 = D2.to_numpy()

# Initializing root node
C = DetermineCandidateSplits(D2)

fea_vec = [0,1]
GR = [[GainRatio(D2, fea, t) for t in C[fea]] for fea in fea_vec]
GR = np.array(GR)
ind = np.unravel_index(np.argmax(GR, axis=None), GR.shape)

fea_opt = fea_vec[ind[0]]
threshold_opt_ix = ind[1]
threshold_opt = C[fea_opt][threshold_opt_ix]
max_GR = max(max(i) for i in GR)

root = Node(fea_opt, threshold_opt, max_GR)
MakeSubtree(D2, root)

printPreorder(root)


# Q2.6

from plotnine import labs, geom_hline

# Scatterplots

# D1
D1 = pd.read_table('data/D1.txt', delimiter = ' ', header = None, names=["x1", "x2", "y"])
D1plot = ggplot(D1, aes("x1", "x2")) + geom_point(aes(color="y")) + labs(title="D1 data and decision boundary") + geom_hline(aes(yintercept=0.201829))
ggsave(filename='D1plot.png', plot=D1plot)

# D2
D2 = pd.read_table('data/D2.txt', delimiter = ' ', header = None, names=["x1", "x2", "y"])
D2plot = ggplot(D2, aes("x1", "x2")) + geom_point(aes(color="y")) + labs(title="D2 data")
ggsave(filename='D2plot.png', plot=D2plot)


# Q2.7

Dbig = pd.read_table('data/Dbig.txt', delimiter = ' ', header = None)
Dbig = Dbig.to_numpy()
Dbig_shuffled = Dbig #copy
np.random.shuffle(Dbig_shuffled)
D8192 = Dbig_shuffled[0:8192,:]
test = Dbig_shuffled[8192:10000,:]

Prop_of_ones = np.mean(Dbig[:,-1])
print(Prop_of_ones)

D32 = D8192[0:32,:]
D128 = D8192[0:128,:]
D512 = D8192[0:512,:]
D2048 = D8192[0:2048,:]


# Q3

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

tree32 = DecisionTreeClassifier()
X = D32[:,:-1]
y = D32[:, -1]
tree32.fit(X, y)
n_nodes32 = tree32.tree_.node_count
test_pred32 = tree32.predict(test[:,:-1])
error32 = 1- metrics.accuracy_score(test[:, -1], test_pred32)

tree128 = DecisionTreeClassifier()
X = D128[:,:-1]
y = D128[:, -1]
tree128.fit(X, y)
n_nodes128 = tree128.tree_.node_count
test_pred128 = tree128.predict(test[:,:-1])
error128 = 1- metrics.accuracy_score(test[:, -1], test_pred128)

tree512 = DecisionTreeClassifier()
X = D512[:,:-1]
y = D512[:, -1]
tree512.fit(X, y)
n_nodes512 = tree512.tree_.node_count
test_pred512 = tree512.predict(test[:,:-1])
error512 = 1- metrics.accuracy_score(test[:, -1], test_pred512)

tree2048 = DecisionTreeClassifier()
X = D2048[:,:-1]
y = D2048[:, -1]
tree2048.fit(X, y)
n_nodes2048 = tree2048.tree_.node_count
test_pred2048 = tree2048.predict(test[:,:-1])
error2048 = 1- metrics.accuracy_score(test[:, -1], test_pred2048)

tree8192 = DecisionTreeClassifier()
X = D8192[:,:-1]
y = D8192[:, -1]
tree8192.fit(X, y)
n_nodes8192 = tree8192.tree_.node_count
test_pred8192 = tree8192.predict(test[:,:-1])
error8192 = 1- metrics.accuracy_score(test[:, -1], test_pred8192)

n = [n_nodes32, n_nodes128, n_nodes512, n_nodes2048, n_nodes8192]
errors = [error32, error128, error512, error2048, error8192]
print(n)
print(errors)

from plotnine import geom_line, scale_x_log10

error_data = pd.DataFrame(data = {"n": n, "Error": errors})
LearningCurve = ggplot(error_data, aes(x="n", y="Error")) + geom_point() + geom_line() + scale_x_log10() + labs(title="Learning Curve")
ggsave(filename="LearningCurve.png", plot=LearningCurve)


# Q4

from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

def Lagrange_interpol(n=10, sigma=1, a=-1, b=1):
    train = np.sort(np.random.uniform(low=a, high=b, size=n))
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    poly = lagrange(train + epsilon, np.sin(train))
    test = np.random.uniform(low=a, high=b, size=n)
    y_train = np.sin(train)
    y_pred_train = Polynomial(poly.coef[::-1])(train)
    training_error = sum((y_train - y_pred_train) ** 2) / n
    y_test = np.sin(test)
    y_pred_test = Polynomial(poly.coef[::-1])(test)
    testing_error = sum((y_test - y_pred_test) ** 2) / n
    print(n, " & ", sigma, " & ",  training_error, " & ", testing_error, " \\\\")

print("Part 1\n")
Lagrange_interpol(10, 0)
Lagrange_interpol(20, 0)

print("Part 2\n")
Lagrange_interpol(10, 1)
Lagrange_interpol(10, 3)
Lagrange_interpol(10, 9)