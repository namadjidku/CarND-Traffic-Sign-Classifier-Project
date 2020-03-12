# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Visualization"
[image2]: ./examples/image2.png
[image3]: ./examples/image3.png
[image4]: ./examples/image4.png
[image5]: ./examples/image5.png
[image6]: ./examples/image.png


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python len() function and set to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below are randomly chosen images from the training data set.

![alt text][image1]

To check the distribution of training data among classes, I used Counter from collections and plotted its results using a bar chart. We can see that data is unbalanced.

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step I performed data augmentation to balance the data. Two affine transformations were used: rotation by -15 or 15 degree and adding gaussian noise. As the result, the classes which had number of instances less then the mean were doubled or tripled depending on the initial amount. After augmenting, the distribution of training data is more balanced:

![alt text][image3]

As a first step, I decided to convert the images to grayscale because grayscale images show more contrast. The function cvtColor of the library opencv was used for converting:

```python
cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
```

Here is an example of a traffic sign image before and after grayscaling.

Augmented image            |  Augmented grayscaled image
:-------------------------:|:-------------------------:
![alt text][image4]        |  ![alt text][image5]


As a last step, I normalized the image data to make the data with mean zero and equal variance, for faster convergence. To normalize I used function normalize from opencv library and reshaped the normalized images form (32,32) to (32,32,1) using numpy:

```python
np.expand_dims(cv2.normalize(x,  np.zeros((32, 32, 1)), 0, 255, cv2.NORM_MINMAX), -1)
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU	                |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Dopout                | With keeping probability = 0.5                |
| Fully connected layer | Input = 400, Output = 120                     |
| RELU					|												|
| Dopout                | With keeping probability = 0.5                |
| Fully connected layer | Input = 120, Output = 84                      |
| RELU					|												|
| Fully connected layer | Input = 84 Output = 43                        |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128, 100 epochs, and a learning rate of 0.001. The optimizer used is AdamOptimizer with softmax cross entropy objective function.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.968 
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture I tried was Lenet-5 with rgb normalized images. 
* What were some problems with the initial architecture?

The accuracy on validation set was approximately 90%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Based on the high training accuracy and lower validation accuracy we can claim that LeNeT-5 model suffers from overfitting with traffic sign data. To avoid overfitting I added to the initial structure two dropouts with keeping probability 0.5. Dropout is added after the first pooling layer and befor the last fully-connected layer. To improve training of the network, I tried to balance the training data set via augmenting data. 
* Which parameters were tuned? How were they adjusted and why?

I adjusted the number of epochs (100) since the validation accuracy was increasing. I am still not convinced myself with this value, as the increase should be monotonic, for me it is increasing but with small jumps. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

As I have already described adding dropouts helped to decrease overfitting. Considering the semantic complexity of the images  (the images are quite simple) and their small size (32x32), two convolutional layers performed well. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]  


The images were chosen to test the network against different lighting conditions for traffic signs. The first and third images were chosen as examples of a dark image. The second, fourth and fifth images have similar lighting conditions. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			              |     Prediction	        					| 
|:---------------------------:|:-------------------------------------------:| 
| Speed limit (30km/h)        | Speed limit (30km/h)   						| 
| Dangerous curve to the right| Dangerous curve to the right				|
| Go straight or right		  | Go straight or right						|
| Children crossing	      	  | Children crossing					 	    |
| Stop sign         		  | Stop sign       							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

For the first, third and fifth images, the model is almost absolutely sure in classifying (probability of 0.99). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999                 | Speed limit (30km/h)    					    | 
| 0.693             	| Dangerous curve to the right 					|
| 0.999		            | Go straight or right							|
| 0.711	      	        | Children crossing					 			|
| 0.998				    | Stop sign         							|