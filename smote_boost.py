
#from sklearn.utils import shuffle
import numpy as np
from numpy.core.umath_tests import inner1d
from pandas.compat import numpy
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score, plot_confusion_matrix, recall_score, f1_score, \
precision_score, accuracy_score
import sklearn.metrics as metrics
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import OneSidedSelection, TomekLinks, RepeatedEditedNearestNeighbours, NearMiss
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
class SMOTE(object):

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.

        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1),
                                       return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self
class SMOTEBoost(AdaBoostClassifier):

    def __init__(self,
                 n_samples=100,
                 k_neighbors=5,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        self.n_samples = n_samples
        self.algorithm = algorithm
        self.smote = SMOTE(k_neighbors=k_neighbors,
                           random_state=random_state)

        super(SMOTEBoost, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None, minority_target=None):

        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Check parameters.
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            DTYPE = np.float64  # from fast_dict.pxd
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples.
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights.
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive.
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        if minority_target is None:
            # Determine the minority class label.
            stats_c_ = Counter(y)
            maj_c_ = max(stats_c_, key=stats_c_.get)
            min_c_ = min(stats_c_, key=stats_c_.get)
            self.minority_target = min_c_
        else:
            self.minority_target = minority_target

        # Check parameters.
        self._validate_estimator()

        # Clear any previous fit results.
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # SMOTE step.
            X_min = X[np.where(y == self.minority_target)]
            self.smote.fit(X_min)
            X_syn = self.smote.sample(self.n_samples)
            y_syn = np.full(X_syn.shape[0], fill_value=self.minority_target,
                            dtype=np.int64)
            print(X_syn.shape)
            print(y_syn.shape)
            # Normalize synthetic sample weights based on current training set.
            sample_weight_syn = np.empty(X_syn.shape[0], dtype=np.float64)
            sample_weight_syn[:] = 1. / X.shape[0]

            # Combine the original and synthetic samples.
            X = np.vstack((X, X_syn))
            y = np.append(y, y_syn)

            # Combine the weights.
            sample_weight = \
                np.append(sample_weight, sample_weight_syn).reshape(-1, 1)
            sample_weight = \
                np.squeeze(normalize(sample_weight, axis=0, norm='l1'))

            # X, y, sample_weight = shuffle(X, y, sample_weight,
            #                              random_state=random_state)

            # Boosting step.
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination.
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero.
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive.
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize.
                sample_weight /= sample_weight_sum

        return self

def smote_boost():
    clf1 = SMOTEBoost(n_estimators=50)
    clf1.fit(X_train, y_train)
    # clf1.fit(X_train.fillna(0), y_train)
    y_pred = clf1.predict(X_test)
    print('smote_boost_Acc: ', accuracy_score(y_test, y_pred))
    print('smote_boost_Precision: ', precision_score(y_test, y_pred, average='binary'))
    print('smote_boost_Recall: ', recall_score(y_test, y_pred, average='binary'))
    print('smote_boost_F1: ', f1_score(y_test, y_pred, average='binary'))
    print('smote_boost_AUC: ', roc_auc_score(y_test, y_pred))
    plot_AUC(clf1, X_test.fillna(0), y_test)
    # 混淆矩阵
    disp = plot_confusion_matrix(clf1, X_test.fillna(0), y_test,
                                 display_labels=[0, 1],
                                 values_format='',
                                 cmap=plt.cm.Blues
                                 )
    disp.ax_.set_title('confusion matrix')
    plt.show()
    print(metrics.confusion_matrix(y_test, y_pred))
if __name__ == '__main__':
    X = pd.read_excel('all_feature.xlsx', index_col=0)
    y = pd.read_excel('all_label.xlsx')

    new_X = pd.read_excel('non_drug_targetsfea.xlsx', index_col=0)
    # renn()
    X_resampled = pd.read_excel('X_resampled.xlsx', index_col=0)
    y_resampled = pd.read_excel('y_resampled.xlsx', index_col=0)

    # x1 = pd.DataFrame(X.iloc[0:102, :].values)
    # x2 = pd.DataFrame(new_X.iloc[0:, :].values)
    # X_train = pd.concat([x1, x2])
    # print(X_train.shape)
    # x1 = pd.DataFrame(y.iloc[0:102, :].values)
    # x2 = pd.DataFrame(y.iloc[379:18765, :].values)
    # y_train = pd.concat([x1, x2])
    # y_train = y_train.values.ravel()
    # t1 = pd.DataFrame(X.iloc[102:379, :].values)
    # t2 = pd.DataFrame(new_X.iloc[0:, :].values)
    # X_test = pd.concat([t1, t2])
    # t1 = pd.DataFrame(y.iloc[102:379, :].values)
    # t2 = pd.DataFrame(y.iloc[379:18765, :].values)
    # y_test = pd.concat([t1, t2])

    page_blocks = pd.read_excel('page-blocks.xlsx')
    page_blocks_label = pd.read_excel('page-blocks_label.xlsx')
    X_train, X_test, y_train, y_test = train_test_split(page_blocks, page_blocks_label, test_size=0.2,
                                                        random_state=5)
    smote_boost()