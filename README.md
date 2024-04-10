# sat_test

This project expects the dataset to be in the root of the project, in the format provided in the test task description

To check work in progress, please refer to notebooks in the working_notebooks folder. Reading order test_task, followed by dataset_making and nn_trying, and SIFT 
to recreate the experiment, please copy the dataset to this project, and run data_processing.py followed by the trainer.py, which will produce a model in the output
My model lies in the output named siamese_model.pt, and the example of using it is in post_train_evaluation.py.


But overall I'll describe the process of work here
First was preparing the dataset, which consisted of creating pairs of images that are similar, and those that are not similar, and marking them correspondingly.
Overall, the task I formulated for myself is a classification of pairs of images, where the image is either 1 if it is a pair of the same location and 0 if it's not
It would require two pieces - feature extraction and classification
Then, followed the choice of ML method.
Feature extraction can be done by some sort of descriptor method, like SIFT, SURF etc. Or it can be done by NNs
Then for SIFT/SURF there are algorithms to measure the distance between key points. For NNs, the option is to build some architecture of NN that would have at least 2 feature extractors and some classification layers on top.

I've tried using descriptors but didn't achieve much success there.
So, this project contains an NN solution.
I've built a Siamese structure for feature extraction followed by a couple of densely connected layers to classify the image, with pairs of images passed into it.
Then, the dataset was split 80-20, augmented with rescaling, flipping, rotating, change of perspective, and color change, representing my expectations of what would be challenges when comparing UAV and satellite images
I've tried different feature extractions (vggs, resnets of various lengths) and different classifier numbers of layers, activation functions
Tried reducing LR, using different optimizers and weights for classes.

In the end, my metrics aren't particularly satisfying to me, as the accuracy of prediction is about 83ish, and it seems like a task where 99% should be easily achievable, I have plans for further work, but I'm not sure about the deadline of the solution, so I have to wrap it up at some point
The solution overall is more prone to overclaiming with only 3% of correct pairs being identified as incorrect and almost 25% of incorrect being identified as correct as a result of testing. I also did not find any evidence of it being the result of overfitting, as errors were somewhat uniform between the train and test split of the data. 

Further work would include:
- Additional data augmentation
- Fine-tuning feature extraction, as they were frozen during the training of this task
- Exploring triplet loss/architecture with more feature extractors, like adding a gray input, or rotated input
- Investigating on a layer basis, what is active when produced similar responses in non-similar pairs




