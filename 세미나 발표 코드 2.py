##############################################################################
# pip install scikit-learn matplotlib numpy mglearn
##############################################################################
## 라이브러리 ##
import numpy as np
import matplotlib.pyplot as plt
import mglearn
##############################################################################

###############
## Load Data ##
###############

## iris data for classifier
from sklearn.datasets import load_iris
iris = load_iris()
# print(iris.DESCR)

## forge data for classifier
# 인위적인 데이터 셋
X_f, y_f = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X_f[:, 0], X_f[:, 1], y_f)
plt.legend()

## boston data for regressor
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.DESCR)

## wave data for regressor
# 인위적인 데이터 셋
X_w, y_w = mglearn.datasets.make_wave(n_samples = 40)
mglearn.discrete_scatter(X_w, y_w)

#####################################
## training set, test set separate ##
#####################################

from sklearn.model_selection import train_test_split

## iris data
X_i_train, X_i_test, y_i_train, y_i_test = \
    train_test_split(iris.data, iris.target)
    
## forge data
X_f_train, X_f_test, y_f_train, y_f_test = \
    train_test_split(X_f, y_f)

## boston data
X_b_train, X_b_test, y_b_train, y_b_test = \
    train_test_split(boston.data, boston.target)
    
## wave data
X_w_train, X_w_test, y_w_train, y_w_test = \
    train_test_split(X_w, y_w)
    
    
##############################################################################
    
    
#########
## KNN ##
#########
    
####################
## KNN Classifier ##
####################
from sklearn.neighbors import KNeighborsClassifier

# Example
mglearn.plots.plot_knn_classification(n_neighbors = 1)

# Practice 1
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_f_train, y_f_train)
knn.predict([[1, 1]])
knn.predict([[10, 5]])
train_score = knn.score(X_f_train, y_f_train)
test_score = knn.score(X_f_test, y_f_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Practice 2
for n in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_f_train, y_f_train)
    train_score = knn.score(X_f_train, y_f_train)
    test_score = knn.score(X_f_test, y_f_test)
    print("N : {}  /  Train : {:.2f}  /  Test : {:.2f}".format(n, train_score, test_score))

# Visualization
fig, axes = plt.subplots(3, 3, figsize = (10, 10))
n = 1
for ax1 in axes:
    for ax in ax1:
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(X_f_train, y_f_train)
        mglearn.plots.plot_2d_separator(knn, X_f, fill = True, ax = ax, alpha = .4)
        mglearn.discrete_scatter(X_f[:, 0], X_f[:, 1], y_f, ax = ax)
        ax.set_title("%d neighbor"%(n))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        n += 1
fig.tight_layout()

# Practice 3
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_i_train, y_i_train)
knn.predict([[3, 3, 3, 3]])
train_score = knn.score(X_i_train, y_i_train)
test_score = knn.score(X_i_test, y_i_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Practice 4
for n in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_i_train, y_i_train)
    train_score = knn.score(X_i_train, y_i_train)
    test_score = knn.score(X_i_test, y_i_test)
    print("N : {}  /  Train : {:.2f}  /  Test : {:.2f}".format(n, train_score, test_score))

###################
## KNN Regressor ##
###################
from sklearn.neighbors import KNeighborsRegressor

# Example
mglearn.plots.plot_knn_regression(n_neighbors = 1)

# Practice 1
knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_w_train, y_w_train)
knn.predict([[0]])
train_score = knn.score(X_w_train, y_w_train)
test_score = knn.score(X_w_test, y_w_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Practice 2
for n in range(1, 10):
    knn = KNeighborsRegressor(n_neighbors = n)
    knn.fit(X_w_train, y_w_train)
    train_score = knn.score(X_w_train, y_w_train)
    test_score = knn.score(X_w_test, y_w_test)
    print("N : {}  /  Train : {:.2f}  /  Test : {:.2f}".format(n, train_score, test_score))

# Visualization
fig, axes = plt.subplots(3, 3, figsize = (10, 10))
line = np.linspace(min(X_w), max(X_w), 1000)
n = 1
for ax1 in axes:
    for ax in ax1:
        knn = KNeighborsRegressor(n_neighbors = n)
        knn.fit(X_w_train, y_w_train)
        ax.plot(line, knn.predict(line))
        ax.plot(X_w_train, y_w_train, "^", c = "blue", markersize = 5)
        ax.plot(X_w_test, y_w_test, "v", c = "red", markersize = 5)
        ax.set_title("n:{} / train:{:.2f} / test:{:.2f}"\
                     .format(n, knn.score(X_w_train, y_w_train), knn.score(X_w_test, y_w_test)))
        n += 1
        
axes[0][0].legend(["model pred", "train data", "test data"])
fig.tight_layout()

# Practice 3
knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_b_train, y_b_train)
train_score = knn.score(X_b_train, y_b_train)
test_score = knn.score(X_b_test, y_b_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Practice 4
for n in range(1, 10):
    knn = KNeighborsRegressor(n_neighbors = n)
    knn.fit(X_b_train, y_b_train)
    train_score = knn.score(X_b_train, y_b_train)
    test_score = knn.score(X_b_test, y_b_test)
    print("N : {}  /  Train : {:.2f}  /  Test : {:.2f}".format(n, train_score, test_score))
    
    
##############################################################################


############
## Linear ##
############

#######################
## Linear Regression ##
#######################
from sklearn.linear_model import LinearRegression

# Example
mglearn.plots.plot_linear_regression_wave()

# Practice 1
lr = LinearRegression()
lr.fit(X_w_train, y_w_train)
lr.predict([[0]])
train_score = lr.score(X_w_train, y_w_train)
test_score = lr.score(X_w_test, y_w_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Visualization
line = np.linspace(min(X_w), max(X_w), 1000)
mglearn.discrete_scatter(X_w, y_w)
plt.plot(line, lr.predict(line))

# Practice 2
lr = LinearRegression()
lr.fit(X_b_train, y_b_train)
train_score = lr.score(X_b_train, y_b_train)
test_score = lr.score(X_b_test, y_b_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Visualization
plt.plot(lr.coef_, "o")
plt.ylim(-5, 5)
plt.xlabel(boston.feature_names, rotation = 90)


######################
## Ridge Regression ##
######################
from sklearn.linear_model import Ridge

# Practice 1
rg = Ridge()
rg.fit(X_w_train, y_w_train)
rg.predict([[0]])
train_score = rg.score(X_w_train, y_w_train)
test_score = rg.score(X_w_test, y_w_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Visualization
line = np.linspace(min(X_w), max(X_w), 1000)
mglearn.discrete_scatter(X_w, y_w)
plt.plot(line, rg.predict(line))

# Practice 2
Alpha = [0.01, 0.1, 1, 10, 100, 1000]

for A in Alpha:
    rg = Ridge(alpha = A)
    rg.fit(X_w_train, y_w_train)
    train_score = rg.score(X_w_train, y_w_train)
    test_score = rg.score(X_w_test, y_w_test)
    print("Alpha : {}  /  Train : {:.2f}  /  Test : {:.2f}  /  Coef0 : {}"\
          .format(A, train_score, test_score, np.sum(rg.coef_ == 0)))

# Visualization
fig, axes = plt.subplots(2, 3, figsize = (15, 10))
line = np.linspace(min(X_w), max(X_w), 1000)
Alpha = [0.01, 0.1, 1, 10, 100, 1000]
n = 0

for ax1 in axes:
    for ax in ax1:
        rg = Ridge(alpha = Alpha[n])
        rg.fit(X_w_train, y_w_train)
        ax.plot(line, rg.predict(line))
        ax.plot(X_w_train, y_w_train, "^", c = "blue", markersize = 5)
        ax.plot(X_w_test, y_w_test, "v", c = "red", markersize = 5)
        ax.set_title("A:{} / train:{:.2f} / test:{:.2f}"\
                     .format(Alpha[n], rg.score(X_w_train, y_w_train), rg.score(X_w_test, y_w_test)))
        n += 1
axes[0][0].legend(["model pred", "train data", "test data"])
fig.tight_layout()

# Practice 3
rg = Ridge()
rg.fit(X_b_train, y_b_train)
train_score = rg.score(X_b_train, y_b_train)
test_score = rg.score(X_b_test, y_b_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Visualization
plt.plot(boston.feature_names, rg.coef_, "o")
plt.xticks(rotation = 90)

# Practice 4
Alpha = [0.01, 0.1, 1, 10, 100, 1000]

for A in Alpha:
    rg = Ridge(alpha = A)
    rg.fit(X_b_train, y_b_train)
    train_score = rg.score(X_b_train, y_b_train)
    test_score = rg.score(X_b_test, y_b_test)
    print("Alpha : {}  /  Train : {:.2f}  /  Test : {:.2f}  /  Coef0 : {}"\
          .format(A, train_score, test_score, np.sum(rg.coef_ == 0)))
        
# Visualization
fig, axes = plt.subplots(2, 3)
Alpha = [0.01, 0.1, 1, 10, 100, 1000]
n = 0

for ax1 in axes:
    for ax in ax1:
        rg = Ridge(alpha = Alpha[n])
        rg.fit(X_b_train, y_b_train)
        ax.set_ylim(-5, 5)
        ax.plot(rg.coef_, "o")
        ax.set_title("Alpha : {}".format(Alpha[n]))
        n += 1
fig.tight_layout()
        

######################
## Lasso Regression ##
######################
from sklearn.linear_model import Lasso

# Practice 1
ls = Lasso()
ls.fit(X_w_train, y_w_train)
ls.predict([[0]])
train_score = ls.score(X_w_train, y_w_train)
test_score = ls.score(X_w_test, y_w_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Visualization
line = np.linspace(min(X_w), max(X_w), 1000)
mglearn.discrete_scatter(X_w, y_w)
plt.plot(line, ls.predict(line))

# Practice 2
Alpha = [0.01, 0.1, 1, 10, 100, 1000]

for A in Alpha:
    ls = Lasso(alpha = A)
    ls.fit(X_w_train, y_w_train)
    train_score = ls.score(X_w_train, y_w_train)
    test_score = ls.score(X_w_test, y_w_test)
    print("Alpha : {}  /  Train : {:.2f}  /  Test : {:.2f}  /  Coef0 : {}"\
          .format(A, train_score, test_score, np.sum(ls.coef_ == 0)))

# Visualization
fig, axes = plt.subplots(2, 3, figsize = (15, 10))
line = np.linspace(min(X_w), max(X_w), 1000)
Alpha = [0.01, 0.1, 1, 10, 100, 1000]
n = 0

for ax1 in axes:
    for ax in ax1:
        ls = Ridge(alpha = Alpha[n])
        ls.fit(X_w_train, y_w_train)
        ax.plot(line, ls.predict(line))
        ax.plot(X_w_train, y_w_train, "^", c = "blue", markersize = 5)
        ax.plot(X_w_test, y_w_test, "v", c = "red", markersize = 5)
        ax.set_title("A:{} / train:{:.2f} / test:{:.2f}"\
                     .format(Alpha[n], ls.score(X_w_train, y_w_train), ls.score(X_w_test, y_w_test)))
        n += 1
axes[0][0].legend(["model pred", "train data", "test data"])
fig.tight_layout()

# Practice 3
ls = Lasso()
ls.fit(X_b_train, y_b_train)
train_score = ls.score(X_b_train, y_b_train)
test_score = ls.score(X_b_test, y_b_test)
print("Train : {:.2f}  /  Test : {:.2f}  /  Coef0 : {}".format(train_score, test_score, np.sum(ls.coef_ == 0)))

# Practice 4
Alpha = [0.01, 0.1, 1, 10, 100, 1000]
for A in Alpha:
    ls = Lasso(alpha = A)
    ls.fit(X_b_train, y_b_train)
    train_score = ls.score(X_b_train, y_b_train)
    test_score = ls.score(X_b_test, y_b_test)
    print("Alpha : {}  /  Train : {:.2f}  /  Test : {:.2f}  /  Coef0 : {}"\
          .format(A, train_score, test_score, np.sum(ls.coef_ == 0)))

# Visualization
fig, axes = plt.subplots(2, 3)
Alpha = [0.01, 0.1, 1, 10, 100, 1000]
n = 0

for ax1 in axes:
    for ax in ax1:
        ls = Lasso(alpha = Alpha[n])
        ls.fit(X_b_train, y_b_train)
        ax.set_ylim(-5, 5)
        ax.plot(ls.coef_, "o")
        ax.set_title("Alpha : {}".format(Alpha[n]))
        n += 1
fig.tight_layout()


#########################
## Logistic Regression ##
#########################
from sklearn.linear_model import LogisticRegression

# Example (Practice 1 Visualization)
lg = LogisticRegression()
lg.fit(X_f_train, y_f_train)
mglearn.plots.plot_2d_separator(lg, X_f_train, fill = True, alpha = .4)
mglearn.discrete_scatter(X_f[:, 0], X_f[:, 1], y_f)
plt.title("Logistic Regression")
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend()

# Practice 1
lg = LogisticRegression()
lg.fit(X_f_train, y_f_train)
lg.predict([[1, 1]])
lg.predict([[10, 5]])
train_score = lg.score(X_f_train, y_f_train)
test_score = lg.score(X_f_test, y_f_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Practice 2
C = [0.01, 0.1, 1, 10, 100, 1000]
for c in C:
    lg = LogisticRegression(C = c, max_iter = 10000)
    lg.fit(X_f_train, y_f_train)
    train_score = lg.score(X_f_train, y_f_train)
    test_score = lg.score(X_f_test, y_f_test)
    print("C : {}  /  Train : {:.2f}  /  Test : {:.2f}".format(c, train_score, test_score))

# Visualization
fig, axes = plt.subplots(2, 3, figsize = (15, 10))

C = [0.01, 0.1, 1, 10, 100, 1000]
n = 0
for ax1 in axes:
    for ax in ax1:
        lg = LogisticRegression(C = C[n])
        lg.fit(X_f_train, y_f_train)
        mglearn.plots.plot_2d_separator(lg, X_f_train, fill = True, alpha = .4, ax = ax)
        mglearn.discrete_scatter(X_f[:, 0], X_f[:, 1], y_f, ax = ax)
        ax.set_title("C : {}".format(C[n]))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        n += 1
fig.tight_layout()

# Practice 3
lg = LogisticRegression(max_iter = 10000)
lg.fit(X_i_train, y_i_train)
train_score = lg.score(X_i_train, y_i_train)
test_score = lg.score(X_i_test, y_i_test)
print("Train : {:.2f}  /  Test : {:.2f}".format(train_score, test_score))

# Practice 4
C = [0.01, 0.1, 1, 10, 100, 1000]
for c in C:
    lg = LogisticRegression(C = c, max_iter = 10000)
    lg.fit(X_i_train, y_i_train)
    train_score = lg.score(X_i_train, y_i_train)
    test_score = lg.score(X_i_test, y_i_test)
    print("C : {}  /  Train : {:.2f}  /  Test : {:.2f}".format(c, train_score, test_score))