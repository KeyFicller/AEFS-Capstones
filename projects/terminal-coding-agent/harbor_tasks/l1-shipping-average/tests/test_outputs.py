"""Hidden checks for l1-shipping-average. Never copied into the agent's repo."""

from fees import average_fee


def test_empty_input_returns_zero():
    assert average_fee([]) == 0.0


def test_single_fee_is_its_own_average():
    assert average_fee([7]) == 7.0


def test_two_fees_average():
    assert average_fee([1, 2]) == 1.5


def test_rounds_to_two_decimals():
    assert average_fee([1, 2, 4]) == 2.33
