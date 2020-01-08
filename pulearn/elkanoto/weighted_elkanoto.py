import numpy as np
import pdb


class WeighUnlabelled(object):
    def __init__(self, estimator, labelled, unlabelled, hold_out_ratio=0.1):
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.labelled = labelled
        self.unlabelled = unlabelled
        self.fit = self.__fit_no_precomputed_kernel
        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}'.format(
            self.estimator,
            self.c,
            self.estimator_fitted,
        )

    def __fit_no_precomputed_kernel(self, X, s):

        positives = np.where(s == 1.0)[0]
        hold_out_size = int(np.ceil(len(positives) * self.hold_out_ratio))

        if len(positives) <= hold_out_size:
            raise (
                'Not enough positive examples to estimate p(s=1|y=1,x). Need at least '
                + str(hold_out_size + 1)
                + '.'
            )

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]

        X = np.delete(X, hold_out, 0)
        s = np.delete(s, hold_out)

        self.estimator.fit(X, s)

        hold_out_predictions = self.estimator.predict_proba(X_hold_out)

        try:
            hold_out_predictions = hold_out_predictions[:, 1]
        except:
            pass

        c = np.mean(hold_out_predictions)

        self.c = c

        self.estimator_fitted = True

    # Returns E[y] which is P(y=1)
    def estimateEy(self, G):

        n = self.labelled
        m = self.labelled + self.unlabelled

        G = G[:, 1]

        np.place(G, G == 1.0, 0.999)

        pdb.set_trace()

        W = (G / (1 - G)) * ((1 - self.c) / self.c)

        return (float(n) + float(W.sum())) / float(m)

    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise Exception(
                'The estimator must be fitted before calling predict_proba(...).'
            )

        n = self.labelled
        m = self.labelled + self.unlabelled
        # self.estimator.predict_proba gives the probability of P(s=1|x) for x belongs to P or U
        probabilistic_predictions = self.estimator.predict_proba(X)

        yEstimate = self.estimateEy(probabilistic_predictions)

        try:
            probabilistic_predictions = probabilistic_predictions[:, 1]
        except:
            pass

        return (probabilistic_predictions * (self.c * yEstimate * m)) / float(
            n
        )

    def predict(self, X, treshold=0.5):
        if not self.estimator_fitted:
            raise Exception(
                'The estimator must be fitted before calling predict(...).'
            )

        return np.array(
            [1.0 if p > treshold else -1.0 for p in self.predict_proba(X)]
        )
