
import os
import time
import numpy as np
from utils import output_image


class Projection:

    """
    This is a parent class for all projection classes.
    It takes n-dimensional data and projects it into a 2D space.

    Currently only includes SOM, but can be extended to include
    more projection types, e.g. K-means
    """

    def __init__(self, data, height, width, iterations=100):
        self.data = data
        self.height = height
        self.width = width
        self.iterations = iterations
        self.weights = self.generate_weights()

    def generate_weights(self):
        """
        As described, there are i*j*k nodes, where:
            - i is the height of the SOM
            - j is the width of the SOM
            - k is the size of the input vector
        """
        return np.random.random((self.height, self.width, self.data.shape[1]))

    def file_name(self):
        """ Returns a string that describes the settings for the projection type """
        raise NotImplementedError

    def project(self):
        """ This is the main function that projects the data into the 2D space """
        raise NotImplementedError

    def run(self):
        """ Wrapper function to handle non-som logic """

        # Start the timer
        start_time = time.time()

        # Run the SOM
        self.project()

        # End the timer
        end_time = time.time()
        run_time = round(end_time - start_time, 1)

        self.save_image(run_time)

    def save_image(self, run_time):
        # Generate a meaningful filename and save
        cwd = os.getcwd()
        file_name = self.file_name()
        path = cwd + '/output/{0}, took {1} seconds.jpg'.format(file_name, run_time)
        output_image(self.weights, path)
