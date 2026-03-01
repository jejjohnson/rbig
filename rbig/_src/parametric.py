from sklearn.preprocessing import QuantileTransformer


class QuantileGaussianTransform:
    def __init__(self, n_quantiles=1000, random_state=None):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self._qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=random_state
        )

    def fit(self, X):
        self._qt.fit(X)
        return self

    def transform(self, X):
        return self._qt.transform(X)

    def inverse_transform(self, X):
        return self._qt.inverse_transform(X)
