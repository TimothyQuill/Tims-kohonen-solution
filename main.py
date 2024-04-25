import utils
from som import run_som


# Generate 20 random colours
colours = utils.generate_random_colours(20)

# Using 10x10 SOM at 100 iterations...
run_som(colours, 10, 10, 100)

# ...with 200 iterations...
run_som(colours, 10, 10, 200)

# ...with 500 iterations
run_som(colours, 10, 10, 500)

# Using 100x100 SOM at 1000 iterations...
run_som(colours, 100, 100, 1000)
