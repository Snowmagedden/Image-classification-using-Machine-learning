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

![image](https://user-images.githubusercontent.com/66143690/93344653-35a0b800-f832-11ea-98e4-186825a15b6b.png)

Figure 1: Example images and labels
The training dataset consists of the following files:
- training-dataset.npz: This files consists of two python lists that consists
of images (as numpy arrays) and corresponding labels (indexed by their
position in the list). 

#### Task 2
In the second task you will use the classifiers trained in the first task to identify
a series of 5 letters in an image of size 30 x 150. However, compared to the
training images, these images are noisy and consists of multiple letters (sort of
a captcha) as shown in the Figure 3.

![image](https://user-images.githubusercontent.com/66143690/93345137-c4add000-f832-11ea-95c2-a581322c6940.png)

Figure 3: An example test image. The label for this particular image is
'2118161513'. The letters in the image are: 'urpom'. You can encode/decode
the label as: u = 21, r = 18, p=16, o=15, m=13.

Note that the label for these images are generated by concatenating the position
of true letters in the English alphabet in this image. For example, if the letters
in the image are: 'abcde', the the label for this image will be '0102030405'.

In this task you will be judged based on top-5-accuracy.
Top-5-accuracy: Top-5-accuracy is when a classifier is allowed to make 5 different predictions instead of 1 and even if 1 out of the 5 predicted labels is
correct (same as the true label) then it is qualified as a correct classification.
For example, if the letters in the image are 'abcde' (true label = 0102030405)
and the predicted labels are:

![image](https://user-images.githubusercontent.com/66143690/93345709-6b926c00-f833-11ea-8b99-dd07b5185c82.png)

then this is classified as a correct prediction since one of the predictions of
the classifier is correct. High top-5-accuracy is easier to achieve for a classifier
as the classifier gets 5 chances to predict the label of the image and even if any
1 of the prediction is correct then it is considered a correct prediction.

Test Dataset
- test-dataset.npy: test-dataset.npy is a list of 10000 images. For each image
your designed classifier will make 5 predictions. Each row in the predic-
tion.csv file will consists of 5 predicted labels separated by comma. Thus
4
Machine Learning Assignment
the final prediction.csv is a 10000 x 5 .csv with each row representing
the predictions corresponding to the image on that number. A snap-
shot of how the final prediction.csv should look like is shown in Figure 4.
Make sure you predict the labels of the images in the same order as they
are listed in the test-dataset.npy. As a sanity-check, make sure that the
first two images in the test-dataset.npy correspond to labels:'0412042625'
(dlDzy); '2203261217' (vczlq) and the last two images correspond to labels:
'2002151002' (tBojb); '0612010703' (
aGC) respectively.

![image](https://user-images.githubusercontent.com/66143690/93345897-a85e6300-f833-11ea-8722-3634a7cb1a84.png)

Figure 4: Example prediction.csv. This example shows 4 rows and 5 columns.
The prediction.csv you submit must contain 10000 rows and 5 columns
