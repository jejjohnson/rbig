import numpy as np
from sklearn.utils import check_array
from rbig._src.model import AnnealedRBIG
from rbig._src.metrics import neg_entropy_normal


class RBIGKLD:
    """RBIG applied to two multidimensional variables to estimate KL divergence."""

    def __init__(
        self,
        n_layers=50,
        rotation_type="PCA",
        n_quantiles=1000,
        pdf_extension=10,
        random_state=None,
        verbose=None,
        tolerance=None,
        zero_tolerance=100,
        increment=1.5,
    ):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.n_quantiles = n_quantiles
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.increment = increment

    def fit(self, X, Y):
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)
        if self.pdf_extension is None:
            self.pdf_extension = 10
        mv_g = None
        try:
            while mv_g is None:
                if self.verbose:
                    print(f"PDF Extension: {self.pdf_extension}%")
                try:
                    self.rbig_model_Y = AnnealedRBIG(
                        n_layers=self.n_layers,
                        rotation_type=self.rotation_type,
                        random_state=self.random_state,
                        zero_tolerance=self.zero_tolerance,
                        tolerance=self.tolerance,
                        pdf_extension=self.pdf_extension,
                        n_quantiles=self.n_quantiles,
                    )
                    self.rbig_model_Y.fit(Y)
                    X_transformed = self.rbig_model_Y.transform(X)
                    self.rbig_model_X_trans = AnnealedRBIG(
                        n_layers=self.n_layers,
                        rotation_type=self.rotation_type,
                        random_state=self.random_state,
                        zero_tolerance=self.zero_tolerance,
                        tolerance=self.tolerance,
                        pdf_extension=self.pdf_extension,
                        n_quantiles=self.n_quantiles,
                    )
                    self.rbig_model_X_trans.fit(X_transformed)
                    mv_g = self.rbig_model_X_trans.residual_info.sum()
                except Exception:
                    self.pdf_extension = self.increment * self.pdf_extension
        except KeyboardInterrupt:
            print("Interrupted!")
        self.mv_g = mv_g
        if self.verbose == 2:
            print(f"mv_g: {mv_g}")
            print(f"m_g: {neg_entropy_normal(X_transformed)}")
        self.kld = float(mv_g + neg_entropy_normal(X_transformed).sum())
        return self

    def get_kld(self):
        return self.kld
