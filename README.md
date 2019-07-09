# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project Writeup: Build a Traffic Sign Recognition Classifier

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/random_from_set.jpg "Random"
[image2]: ./report_images/train_histo.jpg "Training Set Histo"
[image3]: ./report_images/valid_histo.jpg "Validation Set Histo"
[image4]: ./report_images/test_histo.jpg "Test Set Histo"
[image5]: ./report_images/accuracy.jpg "Accuracy over Epochs"
[image6]: ./report_images/test_images_s1.jpg "Traffic Signs 1"
[image7]: ./report_images/test_images_s2.jpg "Traffic Signs 2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Step 0: Load The Data

In addition to the main datasets, I also loaded the signnames.csv file as a dictionary for looking up label numbers later.

---

## Step 1: Dataset Summary & Exploration
### Q1. Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

I didn't use Numpy or Pandas for pulling any information about the dataset since, as I found out, all the required information was already encoded in the lengths of various subarrays.

Here's an image that was pulled at random from the training set:

![alt text][image1]

As you can see, the image is labelled "14", which corresponds to a stop sign.

##### Code output:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = 32x32
Number of classes = 43

### Q2. Include an exploratory visualization of the dataset
To simply "get a feel" for what the network will be trained on, I randomly pulled an example from the training set, as well as its associated label. 

Next, to find out the distributions of classes in each of the datasets, I generated a histogram which counted the frequency of labels in each of the training, validation, and test sets. 

One thing that's quite apparent is that there is significantly more data available for the labels that are numerically lower. My immediate guess would be that those signs are simply more common than the others—the lower numbers include speed limit signs, which (intuitively) could be of the more common road signs.

The important part is that the distribution is *generally* similar across the three datasets, and that should ensure training accuracy.

##### Histogram of the training set:

![alt text][image2]

##### Histogram of the validation set:

![alt text][image3]

##### Histogram of the test set:

![alt text][image4]

----

## Step 2: Design and Test a Model Architecture

### Q3. Pre-process the Data Set (normalization, grayscale, etc.)

To preprocess the data, I first shuffled the training set so that the network is not trained on the *order* of the data. Next, I converted all of the images to grayscale by taking the mean of the RGB channels. Finally, I normalized the data using the given technique to bring the pixel values between -1 and 1.

### Model Architecture

I began by using the original LeNet-5 architecture from lecture and only achieved a validation set accuracy of 86%. I decided to play around a bit and added another fully connected layer between the two existing ones with the same number of inputs and outputs (84). This resulted in a validation set accuracy of 88%: an improvement of 2%, or 88 examples out of the validation set size of 4410.

Applying the concepts from the linked Sermanet/LeCun paper significantly improved performance of the network. I was able to achieve a validation set accuracy of over 97% and a test set accuracy of over 95%. During the tuning process, I set the number of epochs absurdly high (500 at times) to analyze the behavior far beyond initial convergence. One thing that I noticed was that the accuracy jumped all over the place as it was converging, and beyond. To solve this, I switched to Leaky ReLU activations for all of the non-linear layers. Convergence noise was significantly reduced.

As an added challenge, I attempted to incorporate an inception module in the second layer by passing the inputs through a 3x3 convolution in addition to the original 5x5, and concatenating the outputs to be sent through the rest of the network. Needless to say, the number of parameters increased significantly because of the "branch and merge" step later in the network. There was an improvement in the validation set accuracy of 0.2%, but no change to the test set accuracy. It did, however, take a **lot** more time to train and evaluate. Oh well...

##### Attempt 1: Please refer to LeNet5.py. Original LeNet-5 network implementation from lab. 
###### Max validation accuracy = 86%


##### Attempt 2: Please refer to LeNet6.py. Added another fully-connected layer and called it LeNet-6.
###### Max validation accuracy = 88%


##### Attempt 3: Please refer to LeNet6v2.py. Implemented the branching scheme from the Sermanet/LeCun paper and used Leaky ReLU activations. 
[Link to the paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
###### Max validation accuracy = 97.4%, Test accuracy  = 95.4% (Best so far)


##### Attempt 4: Please refer to LeCeption.py. Added inception on the second convolution layer.
###### Max validation accuracy = 97.6%, Test accuracy = 95.4%. However, training was noticeably slower.



#### Here's the architecture of the final model (Settled on "Attempt 3" above):

0\. Input (32x32x1 images) 

---

1\. (a) Convolutional. 5x5, 1x1 stride. Input = 32x32x1. Output = 28x28x6

1\. (b) Leaky ReLU activation

1\. (c) Max pooling. 2x2, 2x2 stride. Input = 28x28x6. Output = 14x14x6.

---
2\. (a) Convolutional. 5x5, 1x1 stride, Input = 14x14x6 Output = 10x10x16.

2\. (b) Leaky ReLU activation

2\. (c) Max pooling. 2x2, 2x2 stride. Input = 10x10x16. Output = 5x5x16.

---
3\. Flatten layer 2 and branch

---
4\. (a) Convolutional. 5x5, 1x1 stride. Input = 5x5x16. Output = 1x1x400.

4\. (b) Leaky ReLU activation

---
5\. (a) Flatten layer 4 and stack with output of step 3. Output = 1x1x800.

5\. (b) Dropout layer

---
6\. (a) Fully-connected. Input = 800. Output = 120.

6\. (b) Leaky ReLU activation

6\. (c) Dropout layer

---
7\. (a) Fully-connected. Input = 120. Output = 84.

7\. (b) Leaky ReLU activation

7\. (c) Dropout layer

---
8\. (a) Fully-connected. Input = 84. Output = 43.

8\. (b) Logit output


### Train, Validate and Test the Model

I trained the model using the training set for 50 epochs, evaluating it on the validation set on each one. I plotted the validation set accuracy on a plot at the bottom of this code block. It converges at approximately 97% accuracy, with a maximum of 97.4% attained at epoch 48. I used a learning rate of 0.0007 and a batch size of 96. 

![alt text][image5]

### Output Top 5 Softmax Probabilities For 5 Images Found on the Web
For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

## Evaluation of Model Performance

Here are the five images I found online:

![alt text][image6]
![alt text][image7]

I fed the model into the following code block, along with the 5 test images found online. I used the tf.nn.top_k function to output the top 5 softmax probabilities and the results were unexpectedly accurate. Not only did the model classify each sign correctly, it did so **with a 100% probability for each one**. Of course, extending to thousands more examples will begin to expose errors in the classifier, but this was completely unexpected, to say the least. Perhaps the images I picked were quite clear, and that passing in images taken in sub-optimal conditions might cause classification errors, but that could be a future test to be done.

Therefore, the top 5 output array information is patently useless after the main prediction, since it appears to just output the first 4 labels in the softmax layer with exactly 0.0% probability. I'm assuming this is some type of tie-breaker for equal probabilities (of zero in this case).

#### Output from the evaluation code block:
INFO:tensorflow:Restoring parameters from ./lenet  
Image # 1  
14 : Stop with 100.0 % probability  
0 : Speed limit (20km/h) with 0.0 % probability  
1 : Speed limit (30km/h) with 0.0 % probability  
2 : Speed limit (50km/h) with 0.0 % probability  
3 : Speed limit (60km/h) with 0.0 % probability  
  
Image # 2  
18 : General caution with 100.0 % probability  
0 : Speed limit (20km/h) with 0.0 % probability  
1 : Speed limit (30km/h) with 0.0 % probability  
2 : Speed limit (50km/h) with 0.0 % probability  
3 : Speed limit (60km/h) with 0.0 % probability  
  
Image # 3  
34 : Turn left ahead with 100.0 % probability  
0 : Speed limit (20km/h) with 0.0 % probability  
1 : Speed limit (30km/h) with 0.0 % probability  
2 : Speed limit (50km/h) with 0.0 % probability  
3 : Speed limit (60km/h) with 0.0 % probability  
  
Image # 4  
22 : Bumpy road with 100.0 % probability  
0 : Speed limit (20km/h) with 0.0 % probability  
1 : Speed limit (30km/h) with 0.0 % probability  
2 : Speed limit (50km/h) with 0.0 % probability  
3 : Speed limit (60km/h) with 0.0 % probability  
  
Image # 5  
1 : Speed limit (30km/h) with 100.0 % probability  
0 : Speed limit (20km/h) with 0.0 % probability  
2 : Speed limit (50km/h) with 0.0 % probability  
3 : Speed limit (60km/h) with 0.0 % probability  
4 : Speed limit (70km/h) with 0.0 % probability  
  

