# sat_test

This project expects dataset to be in root of the project, in the format as provided in test task description

To check work in progress, please refer to notebooks in working_notebooks folder. Reading order test_task, followed by dataset_making and nn_trying, and SIFT 
to recreate the experiment, please copy the dataset to this project, and run data_processing followed by trainer, which will produce model in the output


But overall I'll describe the process of work here
First was preparing the dataset, which consisted of creating pairs of images that are similar, not similar and marking them.
Overall, the task I formulated for myself is classification of pairs of images, where image is either 1 if it is pair of the same location and 0 if it's not
It would require two pieces - feature extraction and classification
Then, followed the choice of ML method.
Feature extraction can be done by some sort of descriptor method, like SIFT, SURF etc. Or it can be done by NNs
Then for SIFT/SURF there are algorithms to measure distance between keypoint. For NNs, the option is to build some architecture of NN that would have at least 2 feature extractors and a some classification layers on top.

I've tried using descriptors, but didn't achieve much success there.
So, this project contains NN solution.
I've built siamese structure for feature extraction followed by couple of densely connected layers to classify the image, with pairs of images passed into it.
Then, for the dataset it was split 80-20, augmented with rescaling, flipping, rotating, change of perspective, and color change, representing my expectations of what would be challenges when comparing UAV and satellite images
I've tried different feature extractions (vggs, resnets of various length) and different classifier number of layers, activation functions
Tried reducing LR, using different optimizers and weights for classes.

In the end, my metrics aren't particularly satisfying to me, as accuracy of prediction is about 83ish, and it seems like a task where 99% should be easily achievable, and I have plans for further work, but I'm not sure on the deadline of the solution, so I have to wrap it up at some point
The solution overall is more prone to overclaiming with only 3% of correct pairs being identified as incorrect and almost 25% of incorrect being identified as correct as a result of testing. I also did not find any evidence of it being the result of overfitting, as errors were somewhat uniformal between train and test split of the data

Further work would include:
- Additional data augmentation
- Fine-tuning feature extraction, as they were frozen during the training of this task




