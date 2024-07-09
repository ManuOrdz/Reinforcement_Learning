from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

def plot_performance(reward_moving_average, length_moving_average, training_error_moving_average):
    fig, axs = plt.subplots(ncols=3, tight_layout=True)
    cmap = plt.cm.Set1
    titles = ['Episode Rewards', 'Lenght Episodes', 'Training Error']
    # Plot reward, length and training error with confidence interval
    for indx, (ax, moving_average) in enumerate(zip(axs,[
                                                        reward_moving_average, 
                                                        length_moving_average, 
                                                        training_error_moving_average
                                                        ]
                                                    )
                                                ):
        ax.set_title(titles[indx])
        ci = 1.96 * moving_average.std() / np.sqrt(len(moving_average))
        ax.plot(moving_average, color=cmap(indx))
        ax.fill_between(
            range(len(moving_average)),
            (moving_average-ci),
            (moving_average+ci),
            alpha=0.25,
            color=cmap(indx)
        )
    plt.show()
    
def moving_average(a: np.array, rolling_length: int)-> np.array:
    moving_average = (
        np.convolve(
            a=a, v = np.ones(rolling_length), mode='same'
        ) 
        / rolling_length
    )
    return moving_average

