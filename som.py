import numpy as np
from projection import Projection


class SOM(Projection):

    """
    This class implements the Self-Organising Map (SOM) algorithm.
    """

    def __init__(self, data, height, width, iterations=100):
        super().__init__(data, height, width, iterations)

    def calculate_decay(self, t_0, t):
        lambda_ = self.iterations / np.log(t_0)
        return np.exp(-t / lambda_)

    def calculate_influence(self, bmu, radius):
        """
        Calculate the distance of two (x, y) coordinates,
        which decays over time
        """
        # Create a grid of coordinates
        x, y = np.ogrid[0:self.height, 0:self.width]
        # Calculate the Euclidean distance between each node and the BMU
        d_i = (x - bmu[0]) ** 2 + (y - bmu[1]) ** 2
        # Compute the influence
        return np.exp(-d_i / (2 * radius ** 2))

    @staticmethod
    def calculate_learning_rate(t, decay_, a_0=0.1):
        """
        Use the decay value defined for the neighbourhood
        radius to multiply with lr_t-1
        """
        return a_0 if t == 0 else a_0 * decay_

    def calculate_t_0(self):
        return (self.height * self.width) / 2

    def file_name(self):
        return '{0}-by{1} SOM after {2} iters'.format(
            self.height, self.width, self.iterations)

    def find_bmu(self, input_vec):
        """
        To find the BMU, run through each node in the SOM, then:
            a.Grab its weight vector
            b.compare it to the current input vector to find the Euclidean distance
            c.Return the index [i,j] of the node with the shortest distance,
        i.e., the weights of the node that best match the input vector
        """

        # Reshape input_vec to be (1, 1, length(input_vec))
        input_vec = np.reshape(input_vec, (1, 1, -1))
        # Compute all distances from input_vec to each weight vector
        distances = np.linalg.norm(input_vec - self.weights, axis=2)
        # Find the index of the smallest distance
        min_index = np.unravel_index(np.argmin(distances), distances.shape)

        return min_index

    @staticmethod
    def neighbourhood_radius(t_0, decay):
        """ The neighbourhood shrinks over time """
        return t_0 * decay

    def project(self):
        """
        This is the main function that projects the data
        into the 2D space
        """
        t_0 = self.calculate_t_0()

        # For each iteration...
        for t in range(self.iterations):
            decay = self.calculate_decay(t_0, t)
            radius = self.neighbourhood_radius(t_0, decay)
            lr = self.calculate_learning_rate(t, decay)
            # For each sample...
            for input_vec in self.data:
                bmu = self.find_bmu(input_vec)
                influence = self.calculate_influence(bmu, radius)
                self.update_weights(input_vec, influence, lr)

    def update_weights(self, input_vec, influence, lr):
        """
        The weights are impacted by both the learning rate
        and the influence
        """
        # Reshape input_vec for broadcasting
        input_vec = np.reshape(input_vec, (1, 1, -1))
        # Update weights
        self.weights = (self.weights + lr * influence[..., np.newaxis]
                        * (input_vec - self.weights))

