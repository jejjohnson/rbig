import numpy as np
from sklearn.utils import check_array
from rbig._src.base import RBIG


class RBIGMI:
    """RBIG applied to two multidimensional variables for mutual information estimation."""

    def __init__(
        self,
        n_layers=50,
        rotation_type="PCA",
        pdf_resolution=1000,
        pdf_extension=None,
        random_state=None,
        verbose=0,
        tolerance=None,
        zero_tolerance=100,
        increment=1.5,
    ):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.increment = 1.5

    def fit(self, X, Y):
        X = check_array(X, ensure_2d=True, copy=True)
        Y = check_array(Y, ensure_2d=True, copy=True)
        if self.pdf_extension is None:
            self.pdf_extension = 10
        fitted = None
        try:
            while fitted is None:
                if self.verbose:
                    print(f"PDF Extension: {self.pdf_extension}%")
                try:
                    self.rbig_model_X = RBIG(
                        n_layers=self.n_layers,
                        rotation_type=self.rotation_type,
                        pdf_resolution=self.pdf_resolution,
                        pdf_extension=self.pdf_extension,
                        verbose=self.verbose,
                        random_state=self.random_state,
                        zero_tolerance=self.zero_tolerance,
                        tolerance=self.tolerance,
                    )
                    X_transformed = self.rbig_model_X.fit_transform(X)
                    self.rbig_model_Y = RBIG(
                        n_layers=self.n_layers,
                        rotation_type=self.rotation_type,
                        pdf_resolution=self.pdf_resolution,
                        pdf_extension=self.pdf_extension,
                        verbose=self.verbose,
                        random_state=self.random_state,
                        zero_tolerance=self.zero_tolerance,
                        tolerance=self.tolerance,
                    )
                    Y_transformed = self.rbig_model_Y.fit_transform(Y)
                    if self.verbose:
                        print(X_transformed.shape, Y_transformed.shape)
                    XY_transformed = np.hstack([X_transformed, Y_transformed])
                    self.rbig_model_XY = RBIG(
                        n_layers=self.n_layers,
                        rotation_type=self.rotation_type,
                        random_state=self.random_state,
                        zero_tolerance=self.zero_tolerance,
                        tolerance=self.tolerance,
                        pdf_resolution=self.pdf_resolution,
                        pdf_extension=self.pdf_extension,
                        verbose=self.verbose,
                    )
                    self.rbig_model_XY.fit(XY_transformed)
                    fitted = True
                except Exception:
                    self.pdf_extension = self.increment * self.pdf_extension
        except KeyboardInterrupt:
            print("Interrupted!")
        return self

    def mutual_information(self):
        """Return mutual information between the two datasets."""
        return self.rbig_model_XY.residual_info.sum()
