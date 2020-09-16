# Image-classification-using-Machine-learning
EMNIST image classification on noisy overlapping Captchas
## Tasks and Dataset
### Image classifcation
In this assignment there are two separate tasks.
#### Task 1 
The first task is to train a model to recognize handwritten English letters. The
training dataset consists of images of handwritten English alphabets and the
corresponding label. A couple of example images and labels are shown in Figure
1. The labels correspond to the place of the letter in the English alphabet, for
example: A is 1, B is 2 and so on. In total, there are 26 classes (A-Z)
Figure 1: Example images and labels
The training dataset consists of the following files:
- training-dataset.npz: This files consists of two python lists that consists
of images (as numpy arrays) and corresponding labels (indexed by their
position in the list). 
