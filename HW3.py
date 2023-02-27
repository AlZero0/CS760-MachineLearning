# Q2

import numpy as np

Q2data = np.array([
    [0, 3, 0],
    [2, 0, 0],
    [0, 1, 3],
    [0, 1, 2],
    [-1, 0, 1],
    [1, 1, 1]
])

nrows = Q2data.shape[0]
newx = np.array([0,0,0])

Edists = np.zeros(nrows)
for i in range(nrows):
    Edists[i] = np.sqrt(sum((Q2data[i,:] - newx)**2))
Edists

# Q3e

coeff = [-3, 1.7, -0.0025]
print(np.roots(coeff))
coeff = [-3, 1.7, -0.21872776601]
print(np.roots(coeff))
coeff = [-3, 1.7, -0.87973722095]
print(np.roots(coeff))

# Q5a
import pandas as pd
from plotnine import ggplot, aes, geom_point, ggsave, geom_line, labs, theme_minimal

x_coord = [0, 0.25, 0.5, 1]
y_coord = [0.33, 0.67, 1, 1]

ROC_data = pd.DataFrame(data = {"FPR": x_coord, "TPR": y_coord})
ROC_Curve = ggplot(ROC_data, aes(x="FPR",y="TPR"))+geom_point(size=5,fill="red")+geom_line(color="red")+labs(title="ROC Curve")+theme_minimal()
#ROC_Curve
ggsave(filename="ROC_Curve.png", plot=ROC_Curve)

ROC_data_array = np.array(ROC_data)
#ROC_data_array
nrows = ROC_data_array.shape[0]
newx = np.array([0,1])
Edists = np.zeros(nrows)
for i in range(nrows):
    Edists[i] = np.sqrt(sum((ROC_data_array[i,:] - newx)**2))
Edists


import numpy as np
data_test = {'x1': list(np.arange(-2, 2.1, 0.1)),
        'x2': list(np.arange(-2, 2.1, 0.1))}
P1_test = [(x1, x2) for x1 in np.arange(-2, 2.1, .1) for x2 in np.arange(-2, 2.1, .1)]


# P1
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from plotnine import ggsave

P1_train = pd.read_table('data/D2z.txt', delimiter = ' ', header = None, names=["x1", "x2", "y"])

plt.scatter(P1_train[["x1"]], P1_train[["x2"]], facecolors='none', edgecolors='black', s = 29)


P1_test = [(x1, x2) for x1 in np.arange(-2, 2.1, 0.1) for x2 in np.arange(-2, 2.1, 0.1)]
x1_test = [i[0] for i in P1_test]
x2_test = [i[1] for i in P1_test]

plt.scatter(x1_test, x2_test, color = "red", s = 2)

plt.title("Train and test points")

plt.style.use('fivethirtyeight')

#plt.show()
plt.savefig("P1plot.png")

# P2

import pandas as pd

email_data = pd.read_csv("data/emails.csv")
email_data.tail()
X = email_data.drop(columns=['Email No.', 'Prediction'])
y = email_data[['Prediction']]

# Fold 1
test_ix_1 = X.index.isin(np.arange(0,1000))

test_X_1 = X[test_ix_1]
train_X_1 = X[~test_ix_1]
test_X_1.insert(0, "Bias", np.ones(1000)) #add column of ones in 1st position
train_X_1.insert(0, "Bias", np.ones(4000)) #add column of ones in 1st position
test_y_1 = np.ravel(y[test_ix_1])
train_y_1 = np.ravel(y[~test_ix_1])

# Fold 2
test_ix_2 = X.index.isin(np.arange(1000,2000))

test_X_2 = X[test_ix_2]
train_X_2 = X[~test_ix_2]
test_X_2.insert(0, "Bias", np.ones(1000)) #add column of ones in 1st position
train_X_2.insert(0, "Bias", np.ones(4000)) #add column of ones in 1st position
test_y_2 = np.ravel(y[test_ix_2])
train_y_2 = np.ravel(y[~test_ix_2])

# Fold 3
test_ix_3 = X.index.isin(np.arange(2000,3000))

test_X_3 = X[test_ix_3]
train_X_3 = X[~test_ix_3]
test_X_3.insert(0, "Bias", np.ones(1000)) #add column of ones in 1st position
train_X_3.insert(0, "Bias", np.ones(4000)) #add column of ones in 1st position
test_y_3 = np.ravel(y[test_ix_3])
train_y_3 = np.ravel(y[~test_ix_3])

# Fold 4
test_ix_4 = X.index.isin(np.arange(3000,4000))

test_X_4 = X[test_ix_4]
train_X_4 = X[~test_ix_4]
test_X_4.insert(0, "Bias", np.ones(1000)) #add column of ones in 1st position
train_X_4.insert(0, "Bias", np.ones(4000)) #add column of ones in 1st position
test_y_4 = np.ravel(y[test_ix_4])
train_y_4 = np.ravel(y[~test_ix_4])

# Fold 5
test_ix_5 = X.index.isin(np.arange(4000,5000))

test_X_5 = X[test_ix_5]
train_X_5 = X[~test_ix_5]
test_X_5.insert(0, "Bias", np.ones(1000)) #add column of ones in 1st position
train_X_5.insert(0, "Bias", np.ones(4000)) #add column of ones in 1st position
test_y_5 = np.ravel(y[test_ix_5])
train_y_5 = np.ravel(y[~test_ix_5])

# P2 cont.

from sklearn.neighbors import KNeighborsClassifier


def KNN_cv_5fold(k, verbose=False):
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fold 1
    knn.fit(train_X_1, train_y_1)
    y_hat_1 = knn.predict(test_X_1)
    accuracy_1 = np.mean(y_hat_1 == test_y_1)
    TP_1 = sum(y_hat_1[test_y_1.astype('bool')])
    Pred_Pos_1 = sum(y_hat_1)
    precision_1 = TP_1 / Pred_Pos_1
    Actual_Pos_1 = sum(test_y_1)
    recall_1 = TP_1 / Actual_Pos_1

    # Fold 2
    knn.fit(train_X_2, train_y_2)
    y_hat_2 = knn.predict(test_X_2)
    accuracy_2 = np.mean(y_hat_2 == test_y_2)
    TP_2 = sum(y_hat_2[test_y_2.astype('bool')])
    Pred_Pos_2 = sum(y_hat_2)
    precision_2 = TP_2 / Pred_Pos_2
    Actual_Pos_2 = sum(test_y_2)
    recall_2 = TP_2 / Actual_Pos_2

    # Fold 3
    knn.fit(train_X_3, train_y_3)
    y_hat_3 = knn.predict(test_X_3)
    accuracy_3 = np.mean(y_hat_3 == test_y_3)
    TP_3 = sum(y_hat_3[test_y_3.astype('bool')])
    Pred_Pos_3 = sum(y_hat_3)
    precision_3 = TP_3 / Pred_Pos_3
    Actual_Pos_3 = sum(test_y_3)
    recall_3 = TP_3 / Actual_Pos_3

    # Fold 4
    knn.fit(train_X_4, train_y_4)
    y_hat_4 = knn.predict(test_X_4)
    accuracy_4 = np.mean(y_hat_4 == test_y_4)
    TP_4 = sum(y_hat_4[test_y_4.astype('bool')])
    Pred_Pos_4 = sum(y_hat_4)
    precision_4 = TP_4 / Pred_Pos_4
    Actual_Pos_4 = sum(test_y_4)
    recall_4 = TP_4 / Actual_Pos_4

    # Fold 5
    knn.fit(train_X_5, train_y_5)
    y_hat_5 = knn.predict(test_X_5)
    accuracy_5 = np.mean(y_hat_5 == test_y_5)
    TP_5 = sum(y_hat_5[test_y_5.astype('bool')])
    Pred_Pos_5 = sum(y_hat_5)
    precision_5 = TP_5 / Pred_Pos_5
    Actual_Pos_5 = sum(test_y_5)
    recall_5 = TP_5 / Actual_Pos_5

    if verbose:
        print(f'Accuracy = {round(accuracy_1, 3)}, Precision = {round(precision_1, 3)}, Recall = {round(recall_1, 3)} ')
        print(f'Accuracy = {round(accuracy_2, 3)}, Precision = {round(precision_2, 3)}, Recall = {round(recall_2, 3)} ')
        print(f'Accuracy = {round(accuracy_3, 3)}, Precision = {round(precision_3, 3)}, Recall = {round(recall_3, 3)} ')
        print(f'Accuracy = {round(accuracy_4, 3)}, Precision = {round(precision_4, 3)}, Recall = {round(recall_4, 3)} ')
        print(f'Accuracy = {round(accuracy_5, 3)}, Precision = {round(precision_5, 3)}, Recall = {round(recall_5, 3)} ')

    out = np.array([[accuracy_1, precision_1, recall_1],
                    [accuracy_2, precision_2, recall_2],
                    [accuracy_3, precision_3, recall_3],
                    [accuracy_4, precision_4, recall_4],
                    [accuracy_5, precision_5, recall_5]]
                   )
    return (out)

KNN_cv_5fold(k=1, verbose=True)

# P3

import numpy as np

# Gradient Descent:
def GD_logistic(X, y, max_iter=500):
    n = X.shape[0]
    d = X.shape[1]
    y = y.reshape(n, 1)
    theta_0 = np.zeros((d, max_iter))
    theta_t = theta_0
    X_norm = np.linalg.norm(np.transpose(X), ord=1)  # matrix 1 norm of X^T (max abs sum across rows)
    L = X_norm  # Lipschitz constant for binary cross-entropy
    alpha = 1 / L

    alpha = alpha / n

    for t in range(max_iter - 1):
        Grad_t = np.dot(np.transpose(X), 1 / (1 + np.exp(-np.dot(X, theta_t[:, t].reshape(d, 1)))) - y)  # d x 1 array
        theta_t[:, (t + 1):(t + 2)] = theta_t[:, t].reshape(d, 1) - alpha * Grad_t  # d x 1 array
    return theta_t[:, t].reshape(d, 1)

theta_vec_1 = GD_logistic(train_X_1, train_y_1)
theta_vec_2 = GD_logistic(train_X_2, train_y_2)
theta_vec_3 = GD_logistic(train_X_3, train_y_3)
theta_vec_4 = GD_logistic(train_X_4, train_y_4)
theta_vec_5 = GD_logistic(train_X_5, train_y_5)

# Fold 1
y_hat_1_logistic_prob = 1/(1+np.exp(-np.dot(test_X_1, theta_vec_1)))
y_hat_1_logistic_binary = (y_hat_1_logistic_prob >= 0.5).astype(int) # 0.5 threshold
accuracy_1_logistic = np.mean(y_hat_1_logistic_binary == test_y_1)
TP_1_logistic = int(sum(y_hat_1_logistic_binary[test_y_1.astype('bool')]))
Pred_Pos_1_logistic = int(sum(y_hat_1_logistic_binary))
precision_1_logistic = TP_1_logistic / Pred_Pos_1_logistic
Actual_Pos_1 = sum(test_y_1)
recall_1_logistic = TP_1_logistic / Actual_Pos_1

# Fold 2
y_hat_2_logistic_prob = 1/(1+np.exp(-np.dot(test_X_2, theta_vec_2)))
y_hat_2_logistic_binary = (y_hat_2_logistic_prob >= 0.5).astype(int) # 0.5 threshold
accuracy_2_logistic = np.mean(y_hat_2_logistic_binary == test_y_2)
TP_2_logistic = int(sum(y_hat_2_logistic_binary[test_y_2.astype('bool')]))
Pred_Pos_2_logistic = int(sum(y_hat_2_logistic_binary))
precision_2_logistic = TP_2_logistic / Pred_Pos_2_logistic
Actual_Pos_2 = sum(test_y_2)
recall_2_logistic = TP_2_logistic / Actual_Pos_2

# Fold 3
y_hat_3_logistic_prob = 1/(1+np.exp(-np.dot(test_X_3, theta_vec_3)))
y_hat_3_logistic_binary = (y_hat_3_logistic_prob >= 0.5).astype(int) # 0.5 threshold
accuracy_3_logistic = np.mean(y_hat_3_logistic_binary == test_y_3)
TP_3_logistic = int(sum(y_hat_3_logistic_binary[test_y_3.astype('bool')]))
Pred_Pos_3_logistic = int(sum(y_hat_3_logistic_binary))
precision_3_logistic = TP_3_logistic / Pred_Pos_3_logistic
Actual_Pos_3 = sum(test_y_3)
recall_3_logistic = TP_3_logistic / Actual_Pos_3

# Fold 4
y_hat_4_logistic_prob = 1/(1+np.exp(-np.dot(test_X_4, theta_vec_4)))
y_hat_4_logistic_binary = (y_hat_4_logistic_prob >= 0.5).astype(int) # 0.5 threshold
accuracy_4_logistic = np.mean(y_hat_4_logistic_binary == test_y_4)
TP_4_logistic = int(sum(y_hat_4_logistic_binary[test_y_4.astype('bool')]))
Pred_Pos_4_logistic = int(sum(y_hat_4_logistic_binary))
precision_4_logistic = TP_4_logistic / Pred_Pos_4_logistic
Actual_Pos_4 = sum(test_y_4)
recall_4_logistic = TP_4_logistic / Actual_Pos_4

# Fold 5
y_hat_5_logistic_prob = 1/(1+np.exp(-np.dot(test_X_5, theta_vec_5)))
y_hat_5_logistic_binary = (y_hat_5_logistic_prob >= 0.5).astype(int) # 0.5 threshold
accuracy_5_logistic = np.mean(y_hat_5_logistic_binary == test_y_5)
TP_5_logistic = int(sum(y_hat_5_logistic_binary[test_y_5.astype('bool')]))
Pred_Pos_5_logistic = int(sum(y_hat_5_logistic_binary))
precision_5_logistic = TP_5_logistic / Pred_Pos_5_logistic
Actual_Pos_5 = sum(test_y_5)
recall_5_logistic = TP_5_logistic / Actual_Pos_5

np.array([[accuracy_1_logistic, precision_1_logistic, recall_1_logistic],
                   [accuracy_2_logistic, precision_2_logistic, recall_2_logistic],
                   [accuracy_3_logistic, precision_3_logistic, recall_3_logistic],
                   [accuracy_4_logistic, precision_4_logistic, recall_4_logistic],
                   [accuracy_5_logistic, precision_5_logistic, recall_5_logistic]]
                  )


# P4

out_k1 = KNN_cv_5fold(k=1)
out_k3 = KNN_cv_5fold(k=3)
out_k5 = KNN_cv_5fold(k=5)
out_k7 = KNN_cv_5fold(k=7)
out_k10 = KNN_cv_5fold(k=10)

avg_accuracy_k1 = np.mean(out_k1, axis=0)[0]
avg_accuracy_k3 = np.mean(out_k3, axis=0)[0]
avg_accuracy_k5 = np.mean(out_k5, axis=0)[0]
avg_accuracy_k7 = np.mean(out_k7, axis=0)[0]
avg_accuracy_k10 = np.mean(out_k10, axis=0)[0]
acc = [avg_accuracy_k1,avg_accuracy_k3,avg_accuracy_k5,avg_accuracy_k7,avg_accuracy_k10]
print(acc)

from plotnine import ggplot, aes, geom_point, geom_line, ggsave, labs, theme_minimal, scale_x_continuous

avg_acc_by_k = pd.DataFrame(data = {"k": [1,3,5,7,10], "Average accuracy": acc})
kNN_CV_plot=ggplot(avg_acc_by_k,aes(x="k",y="Average accuracy"))+geom_point(size=3,fill="blue")+geom_line(color="blue")+labs(title="kNN 5-Fold CV")+scale_x_continuous(breaks=(1,3,5,7,10))+theme_minimal()
ggsave(filename="kNN_CV_plot.png", plot=kNN_CV_plot)


# P5

from matplotlib import pyplot as plt

test_ix_single = X.index.isin(np.arange(4000,5000))

test_X_single = X[test_ix_single]
len(test_X_single)
train_X_single = X[~test_ix_single]
test_X_single.insert(0, "Bias", np.ones(1000)) #add column of ones in 1st position
train_X_single.insert(0, "Bias", np.ones(4000)) #add column of ones in 1st position
test_y_single = np.ravel(y[test_ix_single])
train_y_single = np.ravel(y[~test_ix_single])

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_X_single,train_y_single)
y_hat_single = knn.predict(test_X_single)


from sklearn.metrics import roc_curve, auc

# Compute fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(test_y_single, y_hat_single, drop_intermediate = False)
roc_auc = auc(fpr, tpr)

theta_vec_single = GD_logistic(train_X_single, train_y_single)
y_hat_single_logistic_prob = 1/(1+np.exp(-np.dot(test_X_single, theta_vec_single)))

FPR, TPR, thresh = roc_curve(test_y_single, y_hat_single_logistic_prob)
roc_auc_logistic = auc(FPR, TPR)

# Plot ROC curve
plt.plot(fpr, tpr, label='5NN (AUC = %0.3f)' % roc_auc)
plt.plot(FPR, TPR, label='Logistic (AUC = %0.3f)' % roc_auc_logistic)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate (Positive label: 1)')
plt.ylabel('True Positive Rate (Positive label: 1)')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.style.use('fivethirtyeight')
plt.show()
#plt.savefig("P5plot.png")