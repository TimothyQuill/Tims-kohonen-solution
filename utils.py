# Import libraries
import numpy as np
import matplotlib.pyplot as plt

"""
This file contains utility functions that are used in the main.py file.
"""


# Start with a simple way to create colours. This
# will be used to generate the input data for the SOM.
def generate_random_colours(n):
    return np.random.random((n, 3))


def output_image(data, path):
    # Display the image
    plt.imshow(data)
    plt.axis('off')

    # Save the image to a file
    plt.savefig(path)
    plt.close()
