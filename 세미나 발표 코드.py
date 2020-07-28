##########################################################################################
## 필요한 라이브러리 설치 ##
# pip install numpy scipy matplotlib ipython scikit-learn pandas pillow imageio mglearn
#############################################
## 기본 사용 라이브러리 ##
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import os
#############################################
## 지도 학습 ##
# 지도학습의 종류 : 분류 / 회귀
# 분류 : 가능성 있는 클래스 레이블 중 하나를 예측하는 것
# ex) 스팸메일 분류, 붓꽃 분류
# 회귀 : 연속적인 값을 예측하는 것
# ex) 보스턴 집값 예측
#############################################
## 일반화, 과대적합, 과소적합 ##
# 일반화 : 학습된 모델이 처음 보는 데이터에 대해서 정확하게 예측 할 수 있다?
# => 일반화가 잘 되어있다!
# 과대적합 : 훈련 세트의 각 샘플에 너무 가깝게 맞춰진 경우 
# 과소적합 : 데이터의 다양성을 잡아내지 못하고, 정확도도 낮다
#############################################
## 모델 복잡도와 데이터셋 크기의 관계 ##
# 데이터 셋에 다양한 데이터 포인트가 많다?
# => 과대적합 없이 복잡한 모델 생성가능
# 훈련 데이터를 100% 맞추는 것은 좋지 않다.
# <=> 새로운 데이터에는 잘 작동하지 않는다. (일반화하지 않았다)
# 보통 데이터를 training set과 test set으로 나누어서 사용한다
#############################################
## 데이터 셋 ##
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

# iris classification
iris_dataset = load_iris()
print(iris_dataset.DESCR)
print("iris data : \n{}".format(iris_dataset.data[:10, :]))
print("iris feature name : \n{}".format(iris_dataset.feature_names))

# breast cancer classification
cancer = load_breast_cancer()
print(cancer.DESCR)
print(cancer.keys())

# forge classification
X_f, y_f = mglearn.datasets.make_forge()

# wave regression
X_w, y_w = mglearn.datasets.make_wave(n_samples = 40)

# extened boston regression
X_b, y_b = mglearn.datasets.load_extended_boston()

# blobs classification
X_blobs, y_blobs = make_blobs(random_state = 42) # 42

# blobs classification for svm
X_blobs_svm, y_blobs_svm = make_blobs(centers = 4, random_state = 8)
y_blobs_svm = y_blobs_svm % 2

# ram regression
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

# moon classification
X_m, y_m = make_moons(n_samples = 100, noise = 0.25, random_state = 3)

# handcrafted_dataset classification
X_h, y_h = mglearn.tools.make_handcrafted_dataset()

# circle classification
X_circle, y_circle = make_circles(noise = 0.25, factor = .5, random_state = 0)
y_named = np.array(["blue", "red"])[y_circle]
#############################################
## 데이터 나누기 ##
from sklearn.model_selection import train_test_split # train : test = 3 : 1

# iris
X_i_train, X_i_test, y_i_train, y_i_test =\
    train_test_split(iris_dataset.data, iris_dataset.target, random_state = 0)
    
# cancer
X_c_train, X_c_test, y_c_train, y_c_test =\
    train_test_split(cancer.data, cancer.target, random_state = 0)
    
# forge
X_f_train, X_f_test, y_f_train, y_f_test = \
    train_test_split(X_f, y_f, random_state = 0)
    
# wave
X_w_train, X_w_test, y_w_train, y_w_test = \
    train_test_split(X_w, y_w, random_state = 0)

# extened boston
X_b_train, X_b_test, y_b_train, y_b_test = \
    train_test_split(X_b, y_b, random_state = 0)
    
# blobs
X_blobs_train, X_blobs_test, y_blobs_train, y_blobs_test = \
    train_test_split(X_blobs, y_blobs, random_state = 0) 
    
# moons
X_m_train, X_m_test, y_m_train, y_m_test = \
    train_test_split(X_m, y_m, random_state = 0)
    
# circle
X_circle_train, X_circle_test, y_circle_train, y_circle_test = \
    train_test_split(X_circle, y_named, random_state = 0)
#############################################
## iris 데이터 검사 ##
# iris
iris_dataframe = pd.DataFrame(X_i_train, columns = iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c = y_i_train, figsize = (15, 15),\
                           marker = "o", hist_kwds={"bins" : 20}, s = 60, alpha = .8, cmap = mglearn.cm3)
##########################################################################################
### KNN ###
# 예시
mglearn.plots.plot_knn_classification(n_neighbors = 1)
mglearn.plots.plot_knn_classification(n_neighbors = 3)

## KNN-classifier ##
# 가장 가까이 있는 점들을 찾아 투표
from sklearn.neighbors import KNeighborsClassifier

# forge
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_f_train, y_f_train)
clf.score(X_f_test, y_f_test)

# forge visualization
fig, axes = plt.subplots(1, 3, figsize = (10, 3))
for n_neighbor, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors = n_neighbor)
    clf.fit(X_f, y_f)
    mglearn.plots.plot_2d_separator(clf, X_f, fill = True, ax = ax, alpha = .4)
    mglearn.discrete_scatter(X_f[:, 0], X_f[:, 1], y_f, ax = ax)
    ax.set_title("%d neighbor"%(n_neighbor))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc = 3)
    
# iris
knn = KNeighborsClassifier()
knn.fit(X_i_train, y_i_train)
X_new = np.array([[3, 3, 3, 3]])
X_new_pred = iris_dataset.target_names[knn.predict(X_new)]
print("X_new에 대한 예측 : {}".format(X_new_pred))
print("테스트 세트에 대한 정확도 : {:.2f}".format(knn.score(X_i_test, y_i_test)))

# cancer (과대적합, 과소 적합의 예시)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_c_train, y_c_train)
    training_accuracy.append(clf.score(X_c_train, y_c_train))
    test_accuracy.append(clf.score(X_c_test, y_c_test))
    
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("accuracy")
plt.xlabel("n_neigbors")
plt.legend()
#############################################
## KNN-regression ##
# 가장 가까이 있는 점들의 평균
from sklearn.neighbors import KNeighborsRegressor

# 예시
mglearn.plots.plot_knn_regression(n_neighbors = 1)
mglearn.plots.plot_knn_regression(n_neighbors = 3)

# wave
reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(X_w_train, y_w_train)
reg.predict(X_w_test)
reg.score(X_w_test, y_w_test)

# wave visualization
fig, axes = plt.subplots(1, 3, figsize = (15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1) # reshape는 오류때문에 이유를 잘 모르겠음.
for n_neighbor, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors = n_neighbor)
    reg.fit(X_w_train, y_w_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_w_train, y_w_train, "^", c = "blue", markersize = 5)
    ax.plot(X_w_test, y_w_test, "v", c = "red", markersize = 5)
    ax.set_title("n:{} / train:{:.2f} / test:{:.2f}"\
                 .format(n_neighbor, reg.score(X_w_train, y_w_train), reg.score(X_w_test, y_w_test)))
    ax.set_ylabel("target")
    ax.set_xlabel("feature")
axes[0].legend(["model pred", "train data", "test data"])
##########################################################################################
## Linear regression ## 
# 빠르다, 예측이 쉽다, 샘플에 비해 특성이 많을때 잘 작동, 매개변수가 없다
# 모델의 복잡도를 제어할 방법이 없다
# 평균제곱오차를 최소화하는 w, b를 찾는다
#############################################
from sklearn.linear_model import LinearRegression

# 예시
mglearn.plots.plot_linear_regression_wave()

# wave
lr = LinearRegression()
lr.fit(X_w_train, y_w_train)
print("lr.coef : ", lr.coef_)
print("lr.intercept : ", lr.intercept_)
print("훈련 세트에 대한 정확도 : {:.2f}".format(lr.score(X_w_train, y_w_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(lr.score(X_w_test, y_w_test)))

# boston
lr = LinearRegression()
lr.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(lr.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(lr.score(X_b_test, y_b_test)))
#############################################
## Ridge regression ##
# 가중치의 절댓값을 가능한 작게 만든다 (L2)
# 매개변수 alpha로 모델에 제약 가능
# alpha가 클수록 계수를 0에 가깝게 만들어 더 강하게 제어한다
#############################################
from sklearn.linear_model import Ridge

# 제약이 필요한 이유 (학습 곡선)
mglearn.plots.plot_ridge_n_samples()

# boston
ridge = Ridge() # alpha = 1
ridge.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(ridge.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(ridge.score(X_b_test, y_b_test)))
plt.plot(ridge.coef_, "^", label = "ridge alpha = 1")
plt.ylim(-20, 20)
plt.hlines(0, 0, 105)
plt.legend()

ridge10 = Ridge(alpha = 10) # alpha = 10
ridge10.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(ridge10.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(ridge10.score(X_b_test, y_b_test)))
plt.plot(ridge10.coef_, "^", label = "ridge alpha = 10")
plt.ylim(-20, 20)
plt.hlines(0, 0, 105)
plt.legend()

ridge01 = Ridge(alpha = 0.1) # alpha = 0.1
ridge01.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(ridge01.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(ridge01.score(X_b_test, y_b_test)))
plt.plot(ridge01.coef_, "^", label = "ridge alpha = 0.1")
plt.ylim(-20, 20)
plt.hlines(0, 0, 105)
plt.legend()
#############################################
## Lasso regression ##
# 가중치의 절댓값을 가능한 작게 만들고, 어떤 계수는 0이 되기도 함 (L1)
# 매개변수 alpha로 모델에 제약 가능
# alpha가 클수록 강하게 제약
#############################################
from sklearn.linear_model import Lasso

# boston
lasso = Lasso() # alpha = 1
lasso.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(lasso.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(lasso.score(X_b_test, y_b_test)))
print("사용한 특성의 개수 : ", np.sum(lasso.coef_ != 0))
plt.plot(lasso.coef_, "^", label = "lasso alpha = 1")
plt.ylim(-20, 20)
plt.hlines(0, 0, 105)
plt.legend()

lasso001 = Lasso(alpha = 0.01, max_iter = 100000) # alpha = 0.01
lasso001.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(lasso001.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(lasso001.score(X_b_test, y_b_test)))
print("사용한 특성의 개수 : ", np.sum(lasso001.coef_ != 0))
plt.plot(lasso001.coef_, "^", label = "lasso alpha = 0.01")
plt.ylim(-20, 20)
plt.hlines(0, 0, 105)
plt.legend()

lasso00001 = Lasso(alpha = 0.0001, max_iter = 100000) # alpha = 0.0001
lasso00001.fit(X_b_train, y_b_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(lasso00001.score(X_b_train, y_b_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(lasso00001.score(X_b_test, y_b_test)))
print("사용한 특성의 개수 : ", np.sum(lasso00001.coef_ != 0))
plt.plot(lasso00001.coef_, "^", label = "lasso alpha = 0.01")
plt.ylim(-20, 20)
plt.hlines(0, 0, 105)
plt.legend()
#############################################
## Logistic Regression, Support Vector Machine ##
# 매개변수 C로 모델 제약 가능
# C가 클수록 훈련데이터에 맞추려고 함
# C가 작을수록 가중치를 0에 가까워지도록 함
#############################################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 예시
fig, axes = plt.subplots(1, 2, figsize =(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X_f, y_f)
    mglearn.plots.plot_2d_separator(clf, X_f, fill = True, ax = ax, alpha = .4)
    mglearn.discrete_scatter(X_f[:, 0], X_f[:, 1], y_f, ax = ax)
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend()

# 예시 svm 모델
mglearn.plots.plot_linear_svc_regularization() # SVM모델
#############################################
## Logistic Regression ##
# cancer (L2)
logreg = LogisticRegression(max_iter = 10000)
logreg.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.3f}".format(logreg.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.3f}".format(logreg.score(X_c_test, y_c_test)))

logreg100 = LogisticRegression(C = 100, max_iter = 10000)
logreg100.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.3f}".format(logreg100.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.3f}".format(logreg100.score(X_c_test, y_c_test)))

logreg001 = LogisticRegression(C = 0.01, max_iter = 10000)
logreg001.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.3f}".format(logreg001.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.3f}".format(logreg001.score(X_c_test, y_c_test)))

# 그림
plt.plot(logreg.coef_.T, "o", label = "C = 1")
plt.plot(logreg001.coef_.T, "v", label = "C = 0.01")
plt.plot(logreg100.coef_.T, "^", label = "C = 100")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("features")
plt.ylabel("coefficient size")
plt.legend()

# cancer (L1)
for C, marker in zip([0.01, 1, 100], ["v", "o", "^"]):
    lr_l1 = LogisticRegression(solver = "liblinear", C = C, penalty = "l1")
    lr_l1.fit(X_c_train, y_c_train)
    print("C={}인 L1로지스틱 회귀의 훈련 정확도 : {:.2f}".format(C, lr_l1.score(X_c_train, y_c_train)))
    print("C={}인 L1로지스틱 회귀의 테스트 정확도 : {:.2f}".format(C, lr_l1.score(X_c_test, y_c_test)))
    plt.plot(lr_l1.coef_.T, marker, label = "C = {}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("features")
plt.ylabel("coefficient size")
plt.legend(loc = 3)
#############################################
## Linear SVC ##
# blobs
mglearn.discrete_scatter(X_blobs[:, 0], X_blobs[:, 1], y_blobs)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(["class 0", "class 1", "class 2"])

linear_svm = LinearSVC()
linear_svm.fit(X_blobs_train, y_blobs_train)
mglearn.plots.plot_2d_classification(linear_svm, X_blobs, fill = True, alpha = .4) # 색으로 분리
mglearn.discrete_scatter(X_blobs[:, 0], X_blobs[:, 1], y_blobs)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c = color)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
plt.legend(["class 0", "class 1", "class 2", "class 0 border", "class 1 border", "class 2 border"], loc = (1.01, 0.3))
##########################################################################################
### Decision tree ###
# 모델을 시각화 하기 쉽다
# 데이터의 스케일에 구애받지 않는다
# 사전가지치기를 해도 과대적합이 되는 경향이 있다
# 사전 가지치기, 사후 가지치기 (scikit-learn은 사전 가지치기만 제공)로 모델 복잡도 제어
#############################################
## DecisionTreeClassifier ##
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_c_train, y_c_train) # cancer data
print("훈련 세트에 대한 정확도 : {:.2f}".format(tree.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(tree.score(X_c_test, y_c_test)))

tree_d4 = DecisionTreeClassifier(max_depth = 4, random_state = 0)
tree_d4.fit(X_c_train, y_c_train) # cancer data
print("훈련 세트에 대한 정확도 : {:.2f}".format(tree_d4.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(tree_d4.score(X_c_test, y_c_test)))

# DecisionTree visualization (graphviz 오류가 날 수 있음)
from sklearn.tree import export_graphviz
export_graphviz(tree_d4, out_file = "tree.dot", class_names = ["malignity", "positivity"], \
                feature_names = cancer.feature_names, impurity = False, filled = True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

# feature importance
print("feature importance : \n", tree_d4.feature_importances_)

# Feature Importance Visualization
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align = "center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree_d4)
#############################################
## DecisionTreeRegressor ##
# 훈련 데이터의 범위 밖의 포인트에 대해 예측을 할 수 없다.
from sklearn.tree import DecisionTreeRegressor

# ram_prices visualization
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price")

# ram_price
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

X_r_train = data_train.date[:, np.newaxis]
y_r_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_r_train, y_r_train)
linear_reg = LinearRegression().fit(X_r_train, y_r_train)
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# visualization
plt.semilogy(data_train.date, data_train.price, label = "training data")
plt.semilogy(data_test.date, data_test.price, label = "test data")
plt.semilogy(ram_prices.date, price_tree, label = "tree pred", linestyle = "--",  color = "black")
plt.semilogy(ram_prices.date, price_lr, label = "linear reg pred", linestyle = "--")
plt.legend()
#############################################
## Random Forest ##
# 과대적합을 회피할 수 있다
# 성능이 좋다
# 단일트리의 단점 보완, 장점 유지
# 시각적인 부분은 단일 트리가 더 뛰어남
# 차원이 높고, 희소한 데이터에는 잘 작동하지 않으며, 이 경우 선형 모델이 더 적합하다
# 조금씩 다른 여러 결정 트리를 만들고, 결과를 평균 낸다
from sklearn.ensemble import RandomForestClassifier

# moons
forest = RandomForestClassifier(n_estimators = 5, random_state = 0)
forest.fit(X_m_train, y_m_train)

# visualzation
fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_m, y_m, tree, ax = ax)
mglearn.plots.plot_2d_separator(forest, X_m, fill = True, ax = axes[-1, -1], alpha = .4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_m[:, 0], X_m[:, 1], y_m)

# cancer
forest = RandomForestClassifier(n_estimators = 100, random_state = 0)
forest.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(forest.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(forest.score(X_c_test, y_c_test)))

# feature importance visualzation
plot_feature_importances_cancer(forest)
#############################################
## Gradient Boosting ##
# 무작위성이 없고, 강한 사전 가지치기가 적용된다
# 트리 하나에 깊이 5정도의 얕은 트리를 사용하므로 메모리를 적게 쓴다
# 랜덤포레스트보다 매개변수에 더 민감하지만, 더 높은 정확도를 제공한다
# 이진 특성이나, 연속적인 특성에도 잘 동작하지만, 희소한 고차원 데이터에는 잘 동작하지 않는다
# 매개변수 : learning_rate, n_estimators
# learning_rate가 크면 보정을 강하게 하므로 복잡한 모델을 만든다
# n_estimators가 크면 모델의 복잡도가 커지고, 훈련 세트에서의 실수를 더 잘 바로 잡는다
from sklearn.ensemble import GradientBoostingClassifier

# cancer
gbrt = GradientBoostingClassifier(random_state = 0)
gbrt.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(gbrt.score(X_c_train, y_c_train))) # 과대 적합
print("테스트 세트에 대한 정확도 : {:.2f}".format(gbrt.score(X_c_test, y_c_test)))
plot_feature_importances_cancer(gbrt)

# max_depth 제어
gbrt = GradientBoostingClassifier(random_state = 0, max_depth = 1)
gbrt.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(gbrt.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(gbrt.score(X_c_test, y_c_test)))
plot_feature_importances_cancer(gbrt)

# learning_rate 제어
gbrt = GradientBoostingClassifier(random_state = 0, learning_rate = 0.01)
gbrt.fit(X_c_train, y_c_train)
print("훈련 세트에 대한 정확도 : {:.2f}".format(gbrt.score(X_c_train, y_c_train)))
print("테스트 세트에 대한 정확도 : {:.2f}".format(gbrt.score(X_c_test, y_c_test)))
plot_feature_importances_cancer(gbrt)
#############################################
## Bagging ##
# Bootstrap aggregating
# 확률값을 평균하여 예측을 수행, 아니면 빈도가 높은 클래스 레이블로 예측
from preamble import *
from sklearn.ensemble import BaggingClassifier

# bagging + LogisticRegression, cancer
bagging = BaggingClassifier(LogisticRegression(), n_estimators = 100,\
                            oob_score = True, n_jobs = -1, random_state = 0)
bagging.fit(X_c_train, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(bagging.score(X_c_train, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(bagging.score(X_c_test, y_c_test)))
print("OOB 샘플의 정확도 : {:.3f}".format(bagging.oob_score_))

# bagging + DecisionTreeClassifier, moon
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 5, n_jobs = -1, random_state = 0)
bagging.fit(X_m_train, y_m_train)

fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), bagging.estimators_)):
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_m, y_m, tree, ax = ax)

mglearn.plots.plot_2d_separator(bagging, X_m, fill = True, ax = axes[-1, -1], alpha = .4)
axes[-1, -1].set_title("bagging")
mglearn.discrete_scatter(X_m[:, 0], X_m[:, 1], y_m)
plt.show()

# bagging + DecisionTreeClassifier(n_estimators = 100), cancer
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 100, oob_score = True, n_jobs = -1, random_state = 0)
bagging.fit(X_c_train, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(bagging.score(X_c_train, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(bagging.score(X_c_test, y_c_test)))
print("OOB 샘플의 정확도 : {:.3f}".format(bagging.oob_score_))
#############################################
## Extra Tree ##
# 후보 특성을 무작위로 분할한 다음 최적의 분할을 찾는다
# 랜덤포레스트와 다른 방식으로 모델에 무작위성을 부여한다
from sklearn.ensemble import ExtraTreesClassifier

# moon
xtree = ExtraTreesClassifier(n_estimators = 5, n_jobs = -1, random_state = 0)
xtree.fit(X_m_train, y_m_train)

fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), xtree.estimators_)):
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_m, y_m, tree, ax = ax)
    
mglearn.plots.plot_2d_separator(xtree, X_m, fill = True, ax = axes[-1, -1], alpha = .4)
axes[-1, -1].set_title("Extra tree")
mglearn.discrete_scatter(X_m[:, 0], X_m[:, 1], y_m)
plt.show()

# cancer
xtree = ExtraTreesClassifier(n_estimators = 100, n_jobs = -1, random_state = 0)
xtree.fit(X_c_train, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(xtree.score(X_c_train, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(xtree.score(X_c_test, y_c_test)))

# xtree importance
plot_feature_importances_cancer(xtree)
#############################################
## Ada Boost ##
# Adaptive Boosting
# Classifier은 max_depth = 1
# Regressor : max_depth = 3
# 깊이가 1인 트리를 사용해서 직선 경계가 나온다
from sklearn.ensemble import AdaBoostClassifier

# moon
ada = AdaBoostClassifier(n_estimators = 5, random_state = 0)
ada.fit(X_m_train, y_m_train)

fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), ada.estimators_)):
    ax.set_title("tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_m, y_m, tree, ax = ax)
    
mglearn.plots.plot_2d_separator(ada, X_m, fill = True, ax = axes[-1, -1], alpha = .4)
axes[-1, -1].set_title("Ada boost")
mglearn.discrete_scatter(X_m[:, 0], X_m[:, 1], y_m)
plt.show()

# cancer
ada = AdaBoostClassifier(n_estimators = 100, random_state = 0)
ada.fit(X_c_train, y_c_train)

print("훈련 세트 정확도 : {:.3f}".format(ada.score(X_c_train, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(ada.score(X_c_test, y_c_test)))

# Ada Boost feature importance
plot_feature_importances_cancer(ada)
##########################################################################################
### Kernerlized Support Vector Machine ###
# 다양한 데이터셋에서 잘 작동한다
# 데이터의 특성이 몇 개 안되더라도 복잡한 결정 경계를 만들 수 있다.
# 하지만 데이터가 너무 많을 때는 잘 맞지 않다.
# 선형 모델에서의 초평면은 유연하지 못함
# 이를 해결하기 위해서 특성끼리 곱하거나, 특성을 거듭제곱하는 식으로 새로운 특성을 추가한다
#############################################
# 선형 모델의 한계
linear_svm = LinearSVC().fit(X_blobs_svm, y_blobs_svm)
mglearn.plots.plot_2d_separator(linear_svm, X_blobs_svm)
mglearn.discrete_scatter(X_blobs_svm[:, 0], X_blobs_svm[:, 1], y_blobs_svm)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# 3차원 svm 예시
from mpl_toolkits.mplot3d import Axes3D, axes3d

X_new = np.hstack([X_blobs_svm, X_blobs_svm[:, 1:] ** 2]) # 새로운 특성 추가
mask = y_blobs_svm == 0

linear_svm_3d = LinearSVC().fit(X_new, y_blobs_svm)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

figure = plt.figure()
ax = Axes3D(figure, elev = -152, azim = -26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / - coef[2]
ax.plot_surface(XX, YY, ZZ, rstride = 8, alpha = 0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c = "b", cmap = mglearn.cm2, s = 60, edgecolor = "k")
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c = "r", marker = "^", cmap = mglearn.cm2, s = 60, edgecolor = "k")
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_zlabel("feature 1 ** 2")

# 2차원 svm 예시
ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels = [dec.min(), 0, dec.max()], cmap = mglearn.cm2, alpha = 0.5)
mglearn.discrete_scatter(X_blobs_svm[:, 0], X_blobs_svm[:, 1], y_blobs_svm)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
#############################################
## SVM ##
# C가 커질수록 결정경계를 휘어서 정확하게 분류한다
# gamma : 가우시안 커널의 반경을 크게 하여 많은 포인트들이 가까이 있는 것으로 고려
# gamma가 커질수록 하나 하나의 포인트에 민감해진다
from sklearn.svm import SVC

# handcrafted
svm = SVC(kernel = "rbf", C = 10, gamma = 0.1).fit(X_h, y_h)
mglearn.plots.plot_2d_separator(svm, X_h, eps = .5)
mglearn.discrete_scatter(X_h[:, 0], X_h[:, 1], y_h)
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s =15, markeredgewidth = 3)
plt.xlabel("feature 0")
plt.ylabel("feature 1")
fig, axes = plt.subplots(3, 3, figsize = (15, 10))

# 매개변수의 변화
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C = C, log_gamma = gamma, ax = a)
    
axes[0, 0].legend(["class 0", "class 1", "class 0 svm", "class 1 svm"], ncol = 4, loc = (.9, 1.2))

# cancer
svc = SVC()
svc.fit(X_c_train, y_c_train)

print("훈련 세트 정확도 : {:.2f}".format(svc.score(X_c_train, y_c_train)))
print("테스트 세트 정확도 : {:.2f}".format(svc.score(X_c_test, y_c_test)))

# cancer 데이터 박스플롯
plt.boxplot(X_c_train, manage_ticks = False)
plt.yscale("symlog")
plt.xlabel("feature list")
plt.ylabel("feature size")

# cancer 데이터 전처리 for svc
min_on_training = X_c_train.min(axis = 0)
range_on_training = (X_c_train - min_on_training).max(axis = 0)
X_c_train_scaled = (X_c_train - min_on_training) / range_on_training
print("특성별 최솟값 : \n", X_c_train_scaled.min(axis = 0))
print("특성별 최댓값 : \n", X_c_train_scaled.max(axis = 0))
X_c_test_scaled = (X_c_test - min_on_training) / range_on_training

svc = SVC() # C = 1
svc.fit(X_c_train_scaled, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(svc.score(X_c_train_scaled, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(svc.score(X_c_test_scaled, y_c_test)))

svc = SVC(C = 1000) # C = 1000
svc.fit(X_c_train_scaled, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(svc.score(X_c_train_scaled, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(svc.score(X_c_test_scaled, y_c_test)))
##########################################################################################
### Deep Learning ###
# 다른 모델들 보다 높은 성능을 낸다
# 학습이 오래 걸린다
#############################################
# 예시
display(mglearn.plots.plot_logistic_regression_graph())
display(mglearn.plots.plot_single_hidden_layer_graph())
display(mglearn.plots.plot_two_hidden_layer_graph())

# Activation Function
# Leru & tanh
line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label = "tanh")
plt.plot(line, np.maximum(line, 0), linestyle = "--", label = "relu")
plt.legend(loc = "best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
#############################################
## Multilayer perceptrons ##
from sklearn.neural_network import MLPClassifier

# moons
# hidden layer = 100
mlp = MLPClassifier(solver = "lbfgs", random_state = 0)
mlp.fit(X_m_train, y_m_train)
mglearn.plots.plot_2d_separator(mlp, X_m_train, fill = True, alpha = .3)
mglearn.discrete_scatter(X_m_train[:, 0], X_m_train[:, 1], y_m_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# hidden layer = 10
mlp = MLPClassifier(solver = "lbfgs", random_state = 0, hidden_layer_sizes = [10])
mlp.fit(X_m_train, y_m_train)
mglearn.plots.plot_2d_separator(mlp, X_m_train, fill = True, alpha = .3)
mglearn.discrete_scatter(X_m_train[:, 0], X_m_train[:, 1], y_m_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# hidden layer = 10 * 10
mlp = MLPClassifier(solver = "lbfgs", random_state = 0, hidden_layer_sizes = [10, 10])
mlp.fit(X_m_train, y_m_train)
mglearn.plots.plot_2d_separator(mlp, X_m_train, fill = True, alpha = .3)
mglearn.discrete_scatter(X_m_train[:, 0], X_m_train[:, 1], y_m_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# hidden layer = 10 * 10 & activation = tanh
mlp = MLPClassifier(solver = "lbfgs", random_state = 0, hidden_layer_sizes = [10, 10], activation = "tanh")
mlp.fit(X_m_train, y_m_train)
mglearn.plots.plot_2d_separator(mlp, X_m_train, fill = True, alpha = .3)
mglearn.discrete_scatter(X_m_train[:, 0], X_m_train[:, 1], y_m_train)
plt.xlabel("feature 0")
plt.ylabel("feature 1")

# 매개변수의 영향
fig, axes = plt.subplots(2, 4, figsize =(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver = "lbfgs", random_state = 0, hidden_layer_sizes = [n_hidden_nodes, n_hidden_nodes], alpha = alpha)
        mlp.fit(X_m_train, y_m_train)
        mglearn.plots.plot_2d_separator(mlp, X_m_train, fill = True, alpha = .3, ax = ax)
        mglearn.discrete_scatter(X_m_train[:, 0], X_m_train[:, 1], y_m_train, ax = ax)
        ax.set_title("n_hidden = [{}, {}]\nalpha = {:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))

# cancer
mlp = MLPClassifier(random_state = 0)
mlp.fit(X_c_train, y_c_train)
print("훈련 세트 정확도 : {:.2f}".format(mlp.score(X_c_train, y_c_train)))
print("테스트 세트 정확도 : {:.2f}".format(mlp.score(X_c_test, y_c_test)))

# cancer 데이터 전처리
mean_on_train = X_c_train.mean(axis = 0)
std_on_train = X_c_train.std(axis = 0)
X_c_train_scaled = (X_c_train - mean_on_train) / std_on_train
X_c_test_scaled = (X_c_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state = 0)
mlp.fit(X_c_train_scaled, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(mlp.score(X_c_train_scaled, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(mlp.score(X_c_test_scaled, y_c_test)))

mlp = MLPClassifier(max_iter = 1000, random_state = 0)
mlp.fit(X_c_train_scaled, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(mlp.score(X_c_train_scaled, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(mlp.score(X_c_test_scaled, y_c_test)))

mlp = MLPClassifier(max_iter = 1000, alpha = 1, random_state = 0)
mlp.fit(X_c_train_scaled, y_c_train)
print("훈련 세트 정확도 : {:.3f}".format(mlp.score(X_c_train_scaled, y_c_train)))
print("테스트 세트 정확도 : {:.3f}".format(mlp.score(X_c_test_scaled, y_c_test)))

# feature importance
plt.figure(figsize = (20, 5))
plt.imshow(mlp.coefs_[0], interpolation = "none", cmap = "viridis")
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("hidden unit")
plt.ylabel("input feature")
plt.colorbar()
##########################################################################################
### 예측의 불확실성 판단 ###
# 6단원에서 더 자세한 내용을 다룸
#############################################
gbrt = GradientBoostingClassifier(random_state = 0)
gbrt.fit(X_circle_train, y_circle_train)
#############################################
## decision_function ## 
print("X_test.shape : ", X_circle_test.shape)
print("결정 함수 결과 형태 : ", gbrt.decision_function(X_circle_test).shape)
print("결정 함수 : \n", gbrt.decision_function(X_circle_test))
print("임계치와 결정 함수 결과 비교 : \n", gbrt.decision_function(X_circle_test) > 0)
print("예측 : \n", gbrt.predict(X_circle_test))

greater_zero = (gbrt.decision_function(X_circle_test) > 0).astype(int)
pred = gbrt.classes_[greater_zero]
print("pred는 예측 결과와 같다 : ", np.all(pred == gbrt.predict(X_circle_test)))
decision_function = gbrt.decision_function(X_circle_test)
print("결정 함수 최솟 값 : {:.2f}  최댓값 : {:.2f}".format(np.min(decision_function), np.max(decision_function)))

# 결정 경계(좌)와 결정 함수(우)
fig, axes = plt.subplots(1, 2, figsize = (13, 5))
mglearn.tools.plot_2d_separator(gbrt, X_circle, ax = axes[0], alpha = .4, fill = True, cm = mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X_circle, ax = axes[1], alpha = .4, cm = mglearn.ReBl)
for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_circle_test[:, 0], X_circle_test[:, 1], y_circle_test, markers = "^", ax = ax)
    mglearn.discrete_scatter(X_circle_train[:, 0], X_circle_train[:, 1], y_circle_train, markers = "o", ax = ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
cbar = plt.colorbar(scores_image, ax = axes.tolist())
cbar.set_alpha(1)
cbar.draw_all()
axes[0].legend(["test class 0", "test class 1", "training class 0", "training class 1"], ncol = 4, loc = (.1, 1.1))
#############################################
## predict_proba ##
print("확률 값의 형태 : ",gbrt.predict_proba(X_circle_test).shape)
print("예측 확률 : \n", gbrt.predict_proba(X_circle_test))

fig, axes = plt.subplots(1, 2, figsize = (13, 5))

mglearn.tools.plot_2d_separator(gbrt, X_circle, ax = axes[0], alpha = .4, fill = True, cm = mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X_circle, ax = axes[1], alpha = .5, cm = mglearn.ReBl, function = "predict_proba")

for ax in axes:
    # 훈련 포인트와 테스트 포인트를 그리기
    mglearn.discrete_scatter(X_circle_test[:, 0], X_circle_test[:, 1], y_circle_test, markers = "^", ax = ax)
    mglearn.discrete_scatter(X_circle_train[:, 0], X_circle_train[:, 1], y_circle_train, markers = "o", ax = ax)
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    
cbar = plt.colorbar(scores_image, ax = axes.tolist())
cbar.set_alpha(1)
cbar.draw_all()
axes[0].legend(["test class 0", "test class 1", "training class 0", "training class 1"], ncol = 4, loc = (.1, 1.1))
##########################################################################################