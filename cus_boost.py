import numpy as np
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from numpy.lib.function_base import interp
from sklearn.cluster import KMeans
from sklearn.ensemble._weight_boosting import _samme_proba
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import math
import numpy as np
from numpy.core.umath_tests import inner1d
from pandas.compat import numpy
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score, plot_confusion_matrix, recall_score, f1_score, \
    precision_score, accuracy_score, precision_recall_curve, roc_curve, average_precision_score
import sklearn.metrics as metrics
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import OneSidedSelection, TomekLinks, RepeatedEditedNearestNeighbours, NearMiss
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
def plot_AUC(model,X_test,y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print(roc_auc)

def cus_sampler(X_train, y_train, number_of_clusters=23, percentage_to_choose_from_each_cluster=0.5):
    """
    number_of_clusters = 23
    percentage_to_choose_from_each_cluster: 50%
    """

    selected_idx = []
    selected_idx = np.asarray(selected_idx)

    value, counts = np.unique(y_train, return_counts=True)
    minority_class = value[np.argmin(counts)]
    majority_class = value[np.argmax(counts)]

    idx_min = np.where(y_train == minority_class)[0]
    idx_maj = np.where(y_train == majority_class)[0]

    majority_class_instances = X_train[idx_maj]
    majority_class_labels = y_train[idx_maj]

    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(majority_class_instances)

    X_maj = []
    y_maj = []

    points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    for key in points_under_each_cluster.keys():

        points_under_this_cluster = np.array(points_under_each_cluster[key])
        number_of_points_to_choose_from_this_cluster = math.ceil(
            len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)




        selected_points = np.random.choice(points_under_this_cluster,
                                           size=number_of_points_to_choose_from_this_cluster, replace=False)
        X_maj.extend(majority_class_instances[selected_points])
        y_maj.extend(majority_class_labels[selected_points])

        selected_idx = np.append(selected_idx,selected_points)

        # print(len(selected_idx))

        selected_idx = selected_idx.astype(int)


    X_sampled = X_train[selected_idx]
    y_sampled = y_train[selected_idx]


    # X_sampled = np.concatenate((X_train[idx_min], np.array(X_maj)))
    # y_sampled = np.concatenate((y_train[idx_min], np.array(y_maj)))
    #
    # print(X_sampled.shape, y_sampled.shape, selected_idx.shape)

    return X_sampled, y_sampled, selected_idx
class CUSBoostClassifier:
    def __init__(self, n_estimators, depth):
        self.M = n_estimators
        self.depth = depth
       # self.undersampler = RandomUnderSampler(return_indices=True,replacement=False)

        ## Some other samplers to play with ######
        # self.undersampler = EditedNearestNeighbours(return_indices=True,n_neighbors=neighbours)
        # self.undersampler = AllKNN(return_indices=True,n_neighbors=neighbours,n_jobs=4)

    def fit(self, X, Y):
        self.models = []
        self.alphas = []

        N, _ = X.shape
        W = np.ones(N) / N

        for m in range(self.M):
            tree = DecisionTreeClassifier(max_depth=self.depth, splitter='best')


            X_undersampled, y_undersampled, chosen_indices = cus_sampler(X,Y)

            # f_num = pd.Series(y_undersampled)
            # print(f_num.value_counts())
            # f_num = pd.Series(y_undersampled)
            # print(f_num.value_counts())

            tree.fit(X_undersampled, y_undersampled,
                     sample_weight=W[chosen_indices])

            P = tree.predict(X)

            err = np.sum(W[P != Y])

            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.0000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0 :
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1
                except:
                    alpha = 0
                    # W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)

    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX += alpha * tree.predict(X)
        return np.sign(FX), FX

    def predict_proba(self, X):
        # if self.alphas == 'SAMME'
        proba = sum(tree.predict_proba(X) * alpha for tree , alpha in zip(self.models,self.alphas) )


        proba = np.array(proba)


        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba /  normalizer

        # print(proba)
        return proba

    def predict_proba_samme(self, X):
        # if self.alphas == 'SAMME.R'
        proba = sum(_samme_proba(est , 2 ,X) for est in self.models )

        proba = np.array(proba)

        proba = proba / sum(self.alphas)

        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        # proba =  np.linspace(proba)
        # proba = np.array(proba).astype(float)
        proba = proba / normalizer

        # print('proba = ',proba)
        return proba.astype(float)


dataset = 'letter.csv'

df = pd.read_csv(dataset, header=None)
df['label'] = df[df.shape[1] - 1]
#
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])
#
X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

normalization_object = Normalizer()
X = normalization_object.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True)

top_auc = 0
mean_fpr = np.linspace(0, 1, 100)
number_of_clusters = 23
percentage_to_choose_from_each_cluster = 0.25
depth = 5
estimators = 20

current_param_auc = []
current_param_aupr = []
tprs = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                  random_state=42)

classifier = CUSBoostClassifier(depth=depth, n_estimators=estimators)
        # classifier = RusBoost(depth=depth, n_estimators=estimators)
classifier.fit(X_train, y_train)
predictions = classifier.predict_proba_samme(X_test)
#print(type(predictions))
y_pred = classifier.predict(X_test)
#print(y_pred)
npdata = np.array(y_pred)
npdata = npdata.transpose()
#print(npdata.shape)
print('us_boost_Acc: ', accuracy_score(y_test, npdata[:, 0]))
print('cus_boost_Precision: ', precision_score(y_test, npdata[:, 0], average='binary'))
print('cus_boost_Recall: ', recall_score(y_test, npdata[:, 0], average='binary'))
print('cus_boost_F1: ', f1_score(y_test, npdata[:, 0], average='binary'))
print('cus_boost_Gmean: ', geometric_mean_score(y_test, npdata[:, 0]))
print('cus_boost_AUC: ', roc_auc_score(y_test, predictions[:, 1]))

print(metrics.confusion_matrix(y_test, npdata[:, 0]))
#         aupr = average_precision_score(y_test, predictions[:, 1])
#
#         current_param_auc.append(auc)
#
#         current_param_aupr.append(aupr)
#
#         fpr, tpr, thresholds = roc_curve(y_test, predictions[:, 1])
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#
#         current_mean_auc = np.mean(np.array(current_param_auc))
#         current_mean_aupr = np.mean(np.array(current_param_aupr))
#
#         if top_auc < current_mean_auc:
#             top_auc = current_mean_auc
#
#             best_depth = depth
#             best_estimators = estimators
#
#             best_auc = top_auc
#             best_aupr = current_mean_aupr
#
#             best_tpr = np.mean(tprs, axis=0)
#             best_fpr = mean_fpr
#
#             best_precision, best_recall, _ = precision_recall_curve(y_test, predictions[:, 1])
#             best_fpr, best_tpr, thresholds = roc_curve(y_test, predictions[:, 1])
#
#
#         print('ROC: ', top_auc, ' Aupr: ', best_aupr, ' for depth= ', best_depth, ' estimators = ', best_estimators)
# print('ploting', dataset)
# #    plt.clf()
# plt.plot(best_recall, best_precision, lw=2, color='Blue',
#          label='Precision-Recall Curve')
# plt.plot(best_fpr, best_tpr, lw=2, color='red',
#          label='ROC curve')
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.legend(loc="upper right")
# plt.show()

