import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from skimage.util import random_noise
import random

# Create directories for saving the images
if not os.path.exists('../dataset/Color/class0'):
    os.makedirs('../dataset/Color/class0')

if not os.path.exists('../dataset/Color/class1'):
    os.makedirs('../dataset/Color/class1')

# Image dimensions
original_size = (2, 2)
target_size = (224, 224)

# Number of images to generate for each class
num_images = 1000

# Color codes for red, yellow, blue, green, and black in RGB
color_dict = {'red': [1, 0, 0], 'yellow': [1, 1, 0], 'blue': [0, 0, 1], 'green': [0, 1, 0], 'black': [0, 0, 0]}

# Function to create an image
def create_image(colors, num_black):
    # Start with a completely colored image
    image = np.zeros((*original_size, 3))
    colored_locs = random.sample([(i, j) for i in range(2) for j in range(2)], 4-num_black)
    for idx, loc in enumerate(colored_locs):
        image[loc] = colors[idx%2]
    
    return image

# Generate images
for i in range(num_images):
    # Class 0: red+yellow
    num_black = random.randint(0, 2)
    image0 = create_image([color_dict['red'], color_dict['yellow']], num_black)
    image0 = random_noise(image0, mode='gaussian')
    image0_resized = resize(image0, target_size)
    plt.imsave(f'../dataset/Color/class0/{i}.png', image0_resized)
    
    # Class 1: blue+green
    num_black = random.randint(0, 2)
    image1 = create_image([color_dict['blue'], color_dict['green']], num_black)
    image1 = random_noise(image1, mode='gaussian')
    image1_resized = resize(image1, target_size)
    plt.imsave(f'../dataset/Color/class1/{i}.png', image1_resized)