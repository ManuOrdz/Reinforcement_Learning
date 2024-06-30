from __future__ import annotations

class QParameters:
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_episodes: int = 10,
        start_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        gamma: float =0.95,
    ):
        """Initiaze parameter for Qlearnig Class

        Args:
            learning_rate (float): The learning rate
            initial_epsilon: The inital epsilon value
            start_epsilon: The start epsilon
            final_epsilon: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes + 1 
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = start_epsilon / (n_episodes/2)
        self.gamma = gamma
