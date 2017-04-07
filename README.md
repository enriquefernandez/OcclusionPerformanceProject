# Performance Analysis of Deep Neural Networks on Objects with Occlusion

![example](occlusion_examples.jpg?raw=true)

This is my final project for 6.861 Aspects of a Computational Theory of Intelligence MIT course (Fall 2016).

In this project I analyze the performance of state of the art pre-trained neural network models (AlexNet, VGG, Inception, ResNet50) in object recognition tasks in the presence of occlusions.

This is done by comparing the object recognition performance of each model before and after adding occlusions of gaussian blur of different sizes and locations. The data set is the Object Detection Dataset (DET) of the ILSVRC. I selected images in that dataset that only contained one object with its bounding box, which reduced the dataset to 7706 images.

The Gaussian Blur occlusions are generated with OpenCV. The predictions are generated using pre-trained Keras models. The results are stored using h5 files (h5py) and the data is analyzed and the figures generated using numpy, matplotlib and seaborn.

If you are interested, you can read the [project report](EnriqueFernandez_6.861project.pdf) and [final presentation](project_presentation_EnriqueFernandez.pdf).

Enrique Fernandez