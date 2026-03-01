def rbig_fit_transform_xarray(da, features_dim):
    from rbig._src.base import RBIG
    import numpy as np
    data = da.values
    model = RBIG()
    transformed = model.fit_transform(data)
    return transformed, model
