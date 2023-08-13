import matplotlib.pyplot as plt
import numpy as np

def sierpinski(x, y, size, depth):
    if depth == 0:
        # Base case: Draw a triangle
        points = np.array([[x, y], [x + size, y], [x + size / 2, y + size * np.sqrt(3) / 2]])
        plt.fill(points[:, 0], points[:, 1], 'b')
    else:
        # Recursive case: Divide and conquer
        sierpinski(x, y, size / 2, depth - 1)
        sierpinski(x + size / 2, y, size / 2, depth - 1)
        sierpinski(x + size / 4, y + (size * np.sqrt(3)) / 4, size / 2, depth - 1)

# Parameters
size = 6.0  # Size of the main triangle
depth = 5   # Recursive depth

# Create a blank figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# Call the function to generate the Triflake fractal
sierpinski(0, 0, size, depth)

# Show the fractal
plt.show()
