import utils
from som import SOM


# Generate 20 random colours
colours = utils.generate_random_colours(20)

# Using 10x10 SOM at 100 iterations...
SOM(colours, 10, 10, 100).run()

# ...with 200 iterations...
SOM(colours, 10, 10, 200).run()

# ...with 500 iterations
SOM(colours, 10, 10, 500).run()

# Using 100x100 SOM at 1000 iterations...
SOM(colours, 100, 100, 1_000).run()
