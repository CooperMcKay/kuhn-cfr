import pytest

from kuhn.game import (
    ACTIONS,
    DEALS,
    INFOSETS,
    acting_player,
    infoset_key,
    is_terminal,
    payoff,
)

J, Q, K = 0, 1, 2


def test_deals_cover_all_ordered_pairs():
    assert len(DEALS) == 6
    assert all(a != b for a, b in DEALS)


def test_twelve_infosets():
    assert len(INFOSETS) == 12


@pytest.mark.parametrize("history", ["pp", "bp", "bb", "pbp", "pbb"])
def test_terminal_histories(history):
    assert is_terminal(history)


@pytest.mark.parametrize("history", ["", "p", "b", "pb"])
def test_nonterminal_histories(history):
    assert not is_terminal(history)


def test_acting_player_alternates():
    assert [acting_player(h) for h in ["", "p", "pb"]] == [0, 1, 0]
    assert acting_player("b") == 1


def test_infoset_key_hides_opponent_card():
    assert infoset_key((K, J), "pb", 0) == "Kpb"
    assert infoset_key((K, J), "pb", 1) == "Jpb"


@pytest.mark.parametrize(
    "cards,history,expected",
    [
        ((K, Q), "pp", 1),  # check-check showdown, P1 wins antes
        ((J, Q), "pp", -1),
        ((J, Q), "bp", 1),  # P2 folds to bet
        ((K, Q), "bb", 2),  # bet-call showdown
        ((J, K), "bb", -2),
        ((K, Q), "pbp", -1),  # P1 folds to check-bet
        ((K, Q), "pbb", 2),
        ((J, Q), "pbb", -2),
    ],
)
def test_payoffs_for_player_one(cards, history, expected):
    assert payoff(cards, history) == expected


def test_payoff_rejects_nonterminal():
    with pytest.raises(ValueError):
        payoff((K, Q), "pb")


def test_actions():
    assert ACTIONS == ("p", "b")
