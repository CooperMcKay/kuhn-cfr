import pytest

from kuhn.cfr import train
from kuhn.evaluate import exploitability, game_value

GAME_VALUE = -1 / 18
TOL = 0.02


@pytest.fixture(scope="module")
def strategy():
    return train(20_000)


def test_strategy_probabilities_sum_to_one(strategy):
    assert len(strategy) == 12
    for probs in strategy.values():
        assert sum(probs) == pytest.approx(1.0)
        assert all(p >= 0 for p in probs)


def bet(strategy, infoset):
    return strategy[infoset][1]


def test_matches_equilibrium_family(strategy):
    alpha = bet(strategy, "J")
    assert 0 - TOL <= alpha <= 1 / 3 + TOL
    # P1 opening
    assert bet(strategy, "Q") == pytest.approx(0, abs=TOL)
    assert bet(strategy, "K") == pytest.approx(3 * alpha, abs=TOL)
    # P1 facing bet after check
    assert bet(strategy, "Jpb") == pytest.approx(0, abs=TOL)
    assert bet(strategy, "Qpb") == pytest.approx(alpha + 1 / 3, abs=TOL)
    assert bet(strategy, "Kpb") == pytest.approx(1, abs=TOL)
    # P2 facing bet
    assert bet(strategy, "Jb") == pytest.approx(0, abs=TOL)
    assert bet(strategy, "Qb") == pytest.approx(1 / 3, abs=TOL)
    assert bet(strategy, "Kb") == pytest.approx(1, abs=TOL)
    # P2 after check
    assert bet(strategy, "Jp") == pytest.approx(1 / 3, abs=TOL)
    assert bet(strategy, "Qp") == pytest.approx(0, abs=TOL)
    assert bet(strategy, "Kp") == pytest.approx(1, abs=TOL)


def test_game_value_is_minus_one_eighteenth(strategy):
    assert game_value(strategy, strategy) == pytest.approx(GAME_VALUE, abs=1e-3)


def test_exploitability_decreases_with_iterations():
    exploits = [exploitability(train(n)) for n in (10, 100, 1_000, 10_000)]
    assert exploits == sorted(exploits, reverse=True)
    assert exploits[-1] < 0.005


def test_train_rejects_nonpositive_iterations():
    with pytest.raises(ValueError):
        train(0)
