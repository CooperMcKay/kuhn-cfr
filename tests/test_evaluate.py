import pytest

from kuhn.baselines import BASELINES, always_bet, always_call
from kuhn.cfr import train
from kuhn.evaluate import (
    best_response_value,
    exploitability,
    game_value,
    head_to_head,
    simulate_win_rate,
)
from kuhn.game import INFOSETS


@pytest.fixture(scope="module")
def strategy():
    return train(20_000)


def test_baselines_cover_every_infoset_with_valid_probs():
    for name, build in BASELINES.items():
        s = build()
        assert set(s) == set(INFOSETS), name
        assert all(sum(p) == pytest.approx(1.0) for p in s.values()), name


def test_always_bet_vs_always_call_is_hand_computable():
    # P1 always bets, P2 always calls: showdown for 2 chips, symmetric over deals.
    assert game_value(always_bet(), always_call()) == pytest.approx(0.0)


def test_best_response_to_always_call_beats_game_value():
    assert best_response_value(always_call(), player=0) > 0


def test_exploitability_of_always_bet_is_positive():
    assert exploitability(always_bet()) > 0.1


def test_equilibrium_is_not_beaten_by_baselines(strategy):
    # An approximate equilibrium can lose at most about its own exploitability.
    slack = 2 * exploitability(strategy)
    for name, build in BASELINES.items():
        assert head_to_head(strategy, build()) >= -slack, name


def test_simulated_win_rate_is_seeded_and_in_range(strategy):
    a = simulate_win_rate(strategy, always_call(), hands=2_000, seed=1)
    b = simulate_win_rate(strategy, always_call(), hands=2_000, seed=1)
    assert a == b
    assert 0.0 <= a <= 1.0
