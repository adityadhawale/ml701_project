# pip install Augmentor
import Augmentor

p = Augmentor.Pipeline("/Users/AdaTaylor/Desktop/ml_pics/data/caltech-101/101_ObjectCategories")


# We will only do replacements of the images we have 
# so it's easy to keep the ground truth array the same
# and just copy if for multiple sets of images

# Set the probabilities to 1 for deterministic changes
# which is what we want

p.rotate(probability=1, max_left_rotation=00, max_right_rotation=20)
p.zoom(probability=1, min_factor=1.0, max_factor=1.5)

# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=0.5)

# Note that process replaces the pictures that already exist in these dirs
p.process()
