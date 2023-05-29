<h1><p align = "center">Music Genre Classification and Recommendation System</p></h1>

## Project Objective:
To perform in-depth analysis of audio files, visualize and automate genre classification.

## Dataset description:
The [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) is a collection of 10 genres with 100 audio files each, all having a length of 30 secs. A total of 1000 audio files in .wav format.

Genre: 'rock', 'classical', 'metal', 'disco', 'blues', 'reggae', 'country', 'hiphop', 'jazz', 'pop'

## Feature Extraction:
* Generated a dataset by extracting audio features from multiple audio files using the [Librosa](https://librosa.org/doc/latest/tutorial.html) library.
* Extracted spectrogram which is a visual way of representing the signal loudness of a signal over time at various frequencies present in a particular waveform.
* Label encoding the genres in the dataset.


## ML Models
#### 1. XGBoost: 
 XGBoost is an ensemble learning method that combines multiple decision trees to make accurate predictions. It uses a gradient boosting algorithm and is known for its efficiency and effectiveness in various machine learning tasks.

#### 2. Stochastic Gradient Descent Classifier: 
  The SGD classifier is a linear classification model that optimizes the objective function using a variant of the stochastic gradient descent algorithm. It updates the model's parameters using a small random subset of the training data, making it suitable for large-scale datasets.

#### 3. MLP Classifier: 
  The Multilayer Perceptron Classifier, is a type of feedforward neural network with multiple layers of nodes. It can learn complex non-linear patterns and is widely used for classification tasks. The classifier trains by optimizing a loss function using backpropagation and gradient descent.

#### 4. Support Vector Classifier: 
  The Support Vector Classifier (SVC) is a supervised learning algorithm that constructs a hyperplane or set of hyperplanes in a high-dimensional feature space. It aims to maximize the margin between different classes, making it effective in both linear and non-linear classification tasks.

#### 5. Random Forest: 
  Random Forest is an ensemble learning method that creates a multitude of decision trees and combines their predictions. It improves accuracy and reduces overfitting by aggregating the predictions of multiple individual trees. Each tree is built using a random subset of the features and training samples.

#### 6. Light Gradient Boosting Machine (LightGBM): 
  LightGBM is a gradient boosting framework that is known for its high efficiency and fast training speed. It uses a tree-based learning algorithm and employs a novel technique called Gradient-based One-Side Sampling (GOSS) to handle large-scale datasets effectively.

#### 7. Deep Neural Networks: 
  DNNs are a class of artificial neural networks that are composed of multiple hidden layers. They can learn hierarchical representations of data and are highly effective in solving complex tasks such as image and speech recognition. Training DNNs often requires a large amount of data and computational resources.

#### 8. Convolutional Neural Networks (CNNs): 
  CNNs are a specific type of deep neural network designed for processing images. CNNs utilize convolutional layers, pooling layers, and fully connected layers to extract and learn spatial hierarchies of features, enabling them to achieve state-of-the-art performance in computer vision tasks.

## Recommendation System
#### 1. KNN (K-Nearest Neighbors): 
KNN is a non-parametric classification algorithm that predicts the class of a sample by considering the classes of its nearest neighbors in the feature space. The algorithm calculates the distance between the new sample and all existing samples, and the majority class among the k nearest neighbors is assigned to the new sample.

#### 2. Cosine Similarity: 
Cosine similarity is a measure of similarity between two vectors in a high-dimensional space. It calculates the cosine of the angle between the vectors, which indicates the similarity in terms of orientation. 
