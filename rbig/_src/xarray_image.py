def rbig_image_xarray(da):
    from rbig._src.base import RBIG
    from rbig._src.image import flatten_image, unflatten_image
    import numpy as np
    data = da.values
    orig_shape = data.shape
    flat = data.reshape(-1, orig_shape[-1])
    model = RBIG()
    transformed = model.fit_transform(flat)
    return transformed.reshape(orig_shape), model
