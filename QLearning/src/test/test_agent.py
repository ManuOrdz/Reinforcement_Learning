from src.agent import QLearning
from src.parameters import Parameters
parameters = Parameters()


def test_decay_epsilon(QLearning):
    agent = QLearning(parameters)
    