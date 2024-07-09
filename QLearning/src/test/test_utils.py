import pytest
import numpy as np

from src.utils import plot_performance, moving_average


def test_moving_average():
    return_values = moving_average(range(5), 2)
    expected_values = np.convolve(range(5), np.ones(2)/2 , mode='same')
    for fun_value, exp_value in zip(return_values, expected_values):
        assert fun_value == exp_value
        
        
        
@pytest.mark.parametrize(
    "input_array, rolling_length, expected_array",
    [
        (np.arange(5), 2, moving_average(np.arange(5),2)),
        (range(100), 20, moving_average(np.arange(100),20)),
        (np.arange(200), 10, moving_average(np.arange(200),10))
    ]
)
def test_moving_average_param(input_array, rolling_length, expected_array):
    return_values = moving_average(input_array, rolling_length)
    for return_values, expected_values in zip(return_values, expected_array):
        assert return_values == expected_values