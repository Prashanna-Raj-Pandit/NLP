# Introduction to Natural Language Processing (NLP)
This repository contains a Python script (introduction_to_nlp.ipynb) that demonstrates fundamental NLP techniques for text classification using the Kaggle NLP Getting Started dataset. The script explores various machine learning and deep learning models to classify tweets as either indicating a real disaster (label 1) or not (label 0). It covers data preprocessing, text vectorization, embedding, model training, evaluation, and ensembling using TensorFlow, scikit-learn, and TensorFlow Hub.
Table of Contents

# Project Overview
Dataset
Installation
Usage
Models
Results
Embedding Visualization

# Project Overview
The goal of this project is to build and compare multiple models for classifying tweets as disaster-related or non-disaster-related. The script includes:

# Data preprocessing and visualization using pandas.
Text vectorization using TensorFlow's TextVectorization layer and TF-IDF.
Word embeddings using TensorFlow's Embedding layer and Universal Sentence Encoder (USE).
Implementation of a baseline model (Naive Bayes with TF-IDF) and deep learning models (Dense, LSTM, GRU, Bidirectional LSTM, Conv1D, and Transfer Learning with USE).
Model ensembling by averaging predictions from multiple models.
Model evaluation using accuracy, precision, recall, and F1-score.
Visualization of learned word embeddings using the TensorFlow Embedding Projector.

# Dataset
The dataset used is the NLP Getting Started dataset, which includes:

train.csv: Training data with tweet texts and target labels (1 for disaster, 0 for non-disaster).
test.csv: Test data with tweet texts (no labels provided in this script).

The dataset is automatically downloaded and unzipped within the script using the provided URLs.
Installation
To run the script, you need Python 3.7+ and the following dependencies:
pip install tensorflow pandas scikit-learn matplotlib tensorflow-hub numpy

Ensure you have an internet connection to download the dataset and helper functions. If running in Google Colab, additional setup for downloading embedding files may be required.
Usage

**Clone this repository:*
git clone https://github.com/your-username/introduction-to-nlp.git
cd introduction-to-nlp


*# Run the script:*
python introduction_to_nlp.ipynb



The script will:

## Download and preprocess the dataset.
Train and evaluate multiple models (Baseline, Dense, LSTM, GRU, Bidirectional, Conv1D, USE, and 10% Transfer Learning).
Save TensorBoard logs to the model_logs directory.
Generate vectors.tsv and metadata.tsv files for embedding visualization.
Save the best-performing model (model_6_TL.h5).


## Visualize embeddings:

Upload vectors.tsv and metadata.tsv to the TensorFlow Embedding Projector.
Alternatively, download these files manually if running in Google Colab.


## View training logs:
tensorboard --logdir model_logs

Navigate to http://localhost:6006 in your browser to view TensorBoard logs.


## Models
The script implements and compares the following models:
**Model 0:** Baseline (Naive Bayes with TF-IDF)

Uses scikit-learn's TfidfVectorizer to convert text to TF-IDF features and MultinomialNB for classification.
Simple and effective for text classification tasks.

**Model 1:** Simple Dense Model

A neural network with a TextVectorization layer, an Embedding layer, a GlobalAveragePooling1D layer, and a Dense output layer.
Built using TensorFlow's Functional API.

**Model 2:** LSTM (Long Short-Term Memory)

Incorporates an LSTM layer after the embedding layer for sequence modeling.
Suitable for capturing long-term dependencies in text.

**Model 3:** GRU (Gated Recurrent Unit)

Uses a GRU layer instead of LSTM, which has fewer parameters but similar capabilities.

**Model 4:** Bidirectional LSTM

Uses a bidirectional LSTM layer to process text from both left-to-right and right-to-left directions.
Captures context from both directions for improved performance.

**Model 5:** Conv1D

Applies a 1D convolutional layer (Conv1D) followed by a GlobalMaxPool1D layer after the embedding layer.
Suitable for extracting local patterns in text data.

**Model 6:** Universal Sentence Encoder (USE) Transfer Learning

Uses a pretrained Universal Sentence Encoder from TensorFlow Hub to generate sentence embeddings.
Adds dense layers for classification, leveraging pretrained weights for better performance.

**Model 7:** 10% Transfer Learning

A clone of Model 6 trained on only 10% of the training data to evaluate performance with limited data.
Demonstrates the effectiveness of transfer learning with smaller datasets.

**Ensemble**

Combines predictions from Model 0 (Baseline), Model 2 (LSTM), and Model 6 (USE) by averaging their prediction probabilities.
Aims to improve performance by leveraging diverse model strengths.

Each model is trained for 5 epochs, and performance is evaluated on a validation set (10% of the training data).
Results
The script evaluates models using accuracy, precision, recall, and F1-score. Below are the sample results from the script (results may vary slightly due to randomness):


![image](https://github.com/user-attachments/assets/5ed0782e-fa1a-4915-a0b3-a9366e5a4567)

![image](https://github.com/user-attachments/assets/98f38dc8-18ea-4593-851d-7520d6f898da)

Observations:

The USE-based Model 6 outperforms other individual models, likely due to its pretrained embeddings capturing rich semantic information.
The baseline Naive Bayes model performs surprisingly well, indicating that TF-IDF features are effective for this task.
Model 7 (10% Transfer Learning) shows robust performance despite using only 10% of the training data, highlighting the strength of transfer learning.

Embedding Visualization
The script generates vectors.tsv and metadata.tsv files for visualizing the learned word embeddings from Model 1 in the TensorFlow Embedding Projector. This allows exploration of semantic relationships between words (e.g., similar words clustering together).
To visualize:

Upload vectors.tsv and metadata.tsv to the TensorFlow Embedding Projector.
Use visualization techniques like PCA or t-SNE to explore the embedding space.
Adjust settings to analyze word clusters and relationships.

<img width="711" alt="image" src="https://github.com/user-attachments/assets/403f4bae-b66b-4441-87be-361f27730ed6" />
<img width="729" alt="image" src="https://github.com/user-attachments/assets/7fabfea3-8d5c-4de9-a03f-129027974e1f" />



Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows the project's structure and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
