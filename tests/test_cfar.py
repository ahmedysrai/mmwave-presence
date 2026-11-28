import numpy as np
import pytest

from mmpresence.cfar import ca_cfar, ca_cfar_threshold


def test_single_look_alpha_is_the_textbook_one():
    # flat floor of 1: the threshold is alpha itself
    n, pfa = 16, 1e-3
    thr = ca_cfar_threshold(np.ones(101), n_train=n // 2, n_guard=2, pfa=pfa)
    assert thr[50] == pytest.approx(n * (pfa ** (-1 / n) - 1), rel=1e-9)


@pytest.mark.parametrize("n_looks", [1, 20])
def test_false_alarm_rate_matches_design_on_independent_cells(n_looks):
    # each cell is the mean of n_looks exponential powers, as after
    # averaging |X|^2 over frames
    rng = np.random.default_rng(0)
    p = rng.exponential(size=(2000, 200, n_looks)).mean(axis=-1).ravel()
    det = ca_cfar(p, n_train=12, n_guard=2, pfa=1e-2, n_looks=n_looks)
    assert det[50:-50].mean() == pytest.approx(1e-2, rel=0.1)


def test_detections_do_not_depend_on_the_noise_level():
    rng = np.random.default_rng(1)
    p = rng.exponential(size=5000)
    a = ca_cfar(p, n_train=8, n_guard=1, pfa=1e-2)
    assert a.any()
    assert np.array_equal(a, ca_cfar(250.0 * p, n_train=8, n_guard=1, pfa=1e-2))


def test_guard_cells_keep_a_spread_target_out_of_its_own_threshold():
    # target over five bins on a flat floor; without guard cells its
    # shoulders land in the training window and lift the threshold past it
    p = np.ones(101)
    p[48:53] = [10.0, 15.0, 20.0, 15.0, 10.0]
    assert not ca_cfar(p, n_train=8, n_guard=0, pfa=1e-3)[50]
    assert ca_cfar(p, n_train=8, n_guard=2, pfa=1e-3)[50]


def test_edge_cells_use_the_training_cells_they_have():
    thr = ca_cfar_threshold(np.ones(30), n_train=8, n_guard=2, pfa=1e-3)
    assert np.isfinite(thr).all()
    # cell 0 has 8 cells on its right only, the middle has 16
    assert thr[0] > thr[15]
    assert ca_cfar_threshold([1.0], n_train=8, n_guard=2, pfa=1e-3)[0] == np.inf
