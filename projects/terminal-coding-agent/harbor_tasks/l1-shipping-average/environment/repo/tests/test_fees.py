from fees import average_fee


def test_average_of_three_fees():
    assert average_fee([10, 20, 30]) == 20.0


def test_average_of_two_equal_fees():
    assert average_fee([5, 5]) == 5.0


def test_average_rounds_to_two_decimals():
    assert average_fee([1, 2, 2]) == 1.67
