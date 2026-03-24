"""Tests for progress bar integration."""

from unittest.mock import patch

import numpy as np

from rbig import AnnealedRBIG, ImageRBIG


def test_verbose_false_does_not_invoke_tqdm(simple_2d):
    with patch("rbig._src._progress.tqdm") as mock_tqdm:
        model = AnnealedRBIG(n_layers=3, verbose=False)
        model.fit(simple_2d)
        mock_tqdm.assert_not_called()


def test_verbose_true_invokes_tqdm_for_fit(simple_2d):
    with patch(
        "rbig._src._progress.tqdm", wraps=__import__("tqdm").auto.tqdm
    ) as mock_tqdm:
        model = AnnealedRBIG(n_layers=3, verbose=True)
        model.fit(simple_2d)
        assert mock_tqdm.call_count >= 1
        # Check desc kwarg was forwarded
        call_kwargs = mock_tqdm.call_args_list[0][1]
        assert call_kwargs["desc"] == "Fitting RBIG"


def test_verbose_1_does_not_tqdm_transform(simple_2d):
    model = AnnealedRBIG(n_layers=3, verbose=True)
    model.fit(simple_2d)
    with patch("rbig._src._progress.tqdm") as mock_tqdm:
        model.transform(simple_2d)
        mock_tqdm.assert_not_called()


def test_verbose_2_tqdms_transform(simple_2d):
    model = AnnealedRBIG(n_layers=3, verbose=2)
    model.fit(simple_2d)
    with patch(
        "rbig._src._progress.tqdm", wraps=__import__("tqdm").auto.tqdm
    ) as mock_tqdm:
        Xt = model.transform(simple_2d)
        assert mock_tqdm.call_count >= 1
    assert Xt.shape == simple_2d.shape


def test_image_rbig_verbose_fit():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 64))  # 50 images, C=1, H=8, W=8
    model = ImageRBIG(n_layers=2, C=1, H=8, W=8, strategy="dct", verbose=True)
    model.fit(X)
    assert len(model.layers_) == 2
    Xt = model.transform(X)
    assert Xt.shape == X.shape
