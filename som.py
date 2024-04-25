# Import libraries
import os
import time
import numpy as np
from utils import output_image


"""
As described, there are i*j*k nodes, where:
    - i is the height of the SOM
    - j is the width of the SOM
    - k is the size of the input vector"""
def generate_weights(height, width, vec_size):
    return np.random.random((height,width,vec_size))


# Take two vectors and find their Euclidean distance using broadcasting
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2, axis=2)


"""
To find the BMU, run through each node in the SOM, then:
   a. Grab its weight vector
   b. compare it to the current input vector to find the Euclidean distance
   c. Return the index [i,j] of the node with the shortest distance,
   i.e., the weights of the node that best match the input vector"""
def find_bmu(input_vec, weights):
    # Reshape input_vec to be (1, 1, length(input_vec))
    input_vec = np.reshape(input_vec, (1, 1, -1))
    # Compute all distances from input_vec to each weight vector
    distances = euclidean_distance(input_vec, weights)
    # Find the index of the smallest distance
    min_index = np.unravel_index(np.argmin(distances), distances.shape)

    return min_index


"""
To find the neighbourhood radius at time t, the logic was broken up into a set of
function to make it slightly more efficient: t_0 only needs to be calculated once,
and lambda is used in multiple sections of the process, so it makes sense to call
it outside the neighbourhood formula."""
def calculate_t_0(height, width):
    return (height * width) / 2


def calculate_decay(t_0, t, iterations):
    lambda_ = iterations / np.log(t_0)
    return np.exp(-t / lambda_)


# The neighbourhood shrinks over time
def neighbourhood_radius(t_0, decay):
    return t_0 * decay


# Use the decay value defined for the neighbourhood radius to multiply with lr_t-1
def calculate_learning_rate(t, decay_, a_0=0.1):
    return a_0 if t == 0 else a_0 * decay_


# Calculate the distance of two [x, y] coordinates, which decays over time
def calculate_influence(bmu, radius, height, width):
    # Create a grid of coordinates
    x, y = np.ogrid[0:height, 0:width]
    # Calculate the Euclidean distance between each node and the BMU
    d_i = (x - bmu[0]) ** 2 + (y - bmu[1]) ** 2
    # Compute the influence
    return np.exp(-d_i / (2 * radius ** 2))


# The weights are impacted by both the learning rate and the influence
def update_weights(weights, input_vec, influence, lr):
    # Reshape input_vec for broadcasting
    input_vec = np.reshape(input_vec, (1, 1, -1))
    # Update weights
    return weights + lr * influence[..., np.newaxis] * (input_vec - weights)


def som(input_data, height, width, iterations):

    # Generate random weights
    node_weights = generate_weights(height, width, input_data.shape[1])

    t_0 = calculate_t_0(height, width)

    # For each iteration...
    for t in range(iterations):
        decay = calculate_decay(t_0, t, iterations)
        radius = neighbourhood_radius(t_0, decay)
        lr = calculate_learning_rate(t, decay)
        # For each sample...
        for input_vec in input_data:
            bmu = find_bmu(input_vec, node_weights)
            influence = calculate_influence(bmu, radius, height, width)
            node_weights = update_weights(node_weights, input_vec, influence, lr)

    return node_weights

# Wrapper function to handle non-som logic
def run_som(input_data, height, width, iterations):

    # Start the timer
    start_time = time.time()

    # Run the SOM
    node_weights = som(input_data, height, width, iterations)

    # End the timer
    end_time = time.time()
    run_time = round(end_time - start_time, 1)

    # Save the image to a file
    cwd = os.getcwd()
    path = cwd + '/output/{0}-by{1} SOM after {2} iters, took {3} seconds.jpg'.format(height, width, iterations, run_time)
    output_image(node_weights, path)
