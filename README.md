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

#### 1. Statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Visualization of the dataset.

Below are randomly chosen images from the training data set.

![alt text][image1]

Distribution of training data among classes:

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Preprocessing the image data. 

As a first step we performed data augmentation to balance the data. Two affine transformations were used: rotation by -15 or 15 degree and adding gaussian noise. As the result, the classes which had number of instances less then the mean were doubled or tripled depending on the initial amount. After augmenting, the distribution of training data is more balanced:

![alt text][image3]

After augmenting the training dataset, images were converted to the grayscale to represent more contrast. The function cvtColor of the library opencv was used for converting:

```python
cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
```

Example of a traffic sign image before and after grayscaling:

Augmented image            |  Augmented grayscaled image
:-------------------------:|:-------------------------:
![alt text][image4]        |  ![alt text][image5]


At the next stage, we normalized the image data to make the data with mean zero and equal variance using the function normalize from the opencv library. The normalized images were reshaped form (32,32) to (32,32,1) using numpy:

```python
np.expand_dims(cv2.normalize(x,  np.zeros((32, 32, 1)), 0, 255, cv2.NORM_MINMAX), -1)
```

#### 2. Model architecture:

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
 


#### 3. Training model:

The model was trained with a batch size of 128 for 100 epochs and learning rate of 0.001, AdamOptimizer and softmax cross entropy objective function.

#### 4. Approach taken for finding a solution.

The final model's performance results are:
* training set accuracy of 0.989
* validation set accuracy of 0.968 
* test set accuracy of 0.952

The architecture of the NN we used to address classification of the traffic signs is Lenet-5 with rgb normalized images. Based on the high training accuracy (99%) and lower validation accuracy (90%), we concluded that the model suffered from overfitting. To avoid overfitting we added to the initial architecture two dropouts layers with a keeping probability of 0.5. Dropout is added after the first pooling layer and befor the last fully-connected layer. Augmenting the training dataset, adding dropouts and tuning the number of epochs resulted in achieving 96.8% validation accuracy. Considering the semantic complexity of the images (the images are quite simple) and their small size (32x32), two convolutional layers performed well. 
 

### Test a Model on New Images

#### 1. German traffic signs found in web:

![alt text][image6]  


The images were chosen to test the network against different lighting conditions for traffic signs. The first and third images were chosen as examples of a dark image. The second, fourth and fifth images have similar lighting conditions. 

#### 2. Model's predictions.

Prediction results:

| Image			              |     Prediction	        					| 
|:---------------------------:|:-------------------------------------------:| 
| Speed limit (30km/h)        | Speed limit (30km/h)   						| 
| Dangerous curve to the right| Dangerous curve to the right				|
| Go straight or right		  | Go straight or right						|
| Children crossing	      	  | Children crossing					 	    |
| Stop sign         		  | Stop sign       							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Softmax probabilities for each prediction:


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999                 | Speed limit (30km/h)    					    | 
| 0.693             	| Dangerous curve to the right 					|
| 0.999		            | Go straight or right							|
| 0.711	      	        | Children crossing					 			|
| 0.998				    | Stop sign         							|
