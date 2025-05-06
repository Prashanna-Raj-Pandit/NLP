Introduction to Natural Language Processing (NLP)
This repository contains a Python script (introduction_to_nlp.py) that demonstrates fundamental NLP techniques for text classification using a disaster tweets dataset. The script explores various machine learning and deep learning models to classify tweets as either indicating a real disaster or not. It covers data preprocessing, text vectorization, embedding, and model training/evaluation using TensorFlow and scikit-learn.
Table of Contents

Project Overview
Dataset
Installation
Usage
Models
Results
Embedding Visualization
Contributing
License

Project Overview
The goal of this project is to build and compare multiple models for classifying tweets as disaster-related or non-disaster-related. The script includes:

Data preprocessing and visualization using pandas.
Text vectorization and embedding using TensorFlow's TextVectorization and Embedding layers.
Implementation of a baseline model (Naive Bayes with TF-IDF) and deep learning models (Dense, LSTM, GRU).
Model evaluation using accuracy, precision, recall, and F1-score.
Visualization of learned word embeddings using the TensorFlow Embedding Projector.

Dataset
The dataset used is the NLP Getting Started dataset, which includes:

train.csv: Training data with tweet texts and target labels (1 for disaster, 0 for non-disaster).
test.csv: Test data with tweet texts (no labels provided in this script).

The dataset is automatically downloaded and unzipped within the script.
Installation
To run the script, you need Python 3.7+ and the following dependencies:
pip install tensorflow pandas scikit-learn matplotlib

Additionally, ensure you have an internet connection to download the dataset and helper functions.
Usage

Clone this repository:
git clone https://github.com/your-username/introduction-to-nlp.git
cd introduction-to-nlp


Run the script:
python introduction_to_nlp.py


The script will:

Download and preprocess the dataset.
Train and evaluate multiple models.
Save TensorBoard logs to the model_logs directory.
Generate vectors.tsv and metadata.tsv files for embedding visualization.


To visualize embeddings:

Upload vectors.tsv and metadata.tsv to the TensorFlow Embedding Projector.
Alternatively, download these files manually if running in Google Colab.


To view training logs, start TensorBoard:
tensorboard --logdir model_logs

Then, navigate to http://localhost:6006 in your browser.


Models
The script implements and compares the following models:

Model 0: Baseline (Naive Bayes with TF-IDF)

Uses scikit-learn's TfidfVectorizer and MultinomialNB.
Converts text to TF-IDF features and applies a Naive Bayes classifier.


Model 1: Simple Dense Model

A neural network with a TextVectorization layer, an Embedding layer, a GlobalAveragePooling1D layer, and a Dense output layer.
Uses TensorFlow's Functional API.


Model 2: LSTM (Long Short-Term Memory)

Incorporates an LSTM layer after the embedding layer for sequence modeling.
Suitable for capturing long-term dependencies in text.


Model 3: GRU (Gated Recurrent Unit)

Uses a GRU layer instead of LSTM, which has fewer parameters but similar capabilities.



Each model is trained for 5 epochs, and performance is evaluated on a validation set (10% of the training data).
Results
The script evaluates models using accuracy, precision, recall, and F1-score. Sample results (as computed in the script) are summarized below:

Model 0 (Baseline):

Accuracy: ~79.27%
Precision: ~79.39%
Recall: ~79.27%
F1-Score: ~79.15%


Model 1 (Dense):

Accuracy: ~76.51%
Precision: ~76.86%
Recall: ~76.51%
F1-Score: ~76.23%


Model 2 (LSTM):

Accuracy: ~75.62%
Precision: ~75.62%
Recall: ~75.62%
F1-Score: ~75.62%


Model 3 (GRU):

Accuracy: ~76.25%
Precision: ~76.25%
Recall: ~76.25%
F1-Score: ~76.25%



Observation: The baseline Naive Bayes model outperforms the deepPrince of Persia deep learning models, likely due to the simplicity of the dataset and the effectiveness of TF-IDF features for this task.
Embedding Visualization
The script generates files (vectors.tsv and metadata.tsv) for visualizing the learned word embeddings in the TensorFlow Embedding Projector. This allows you to explore semantic relationships between words (e.g., similar words clustering together).
To visualize:

Upload vectors.tsv and metadata.tsv to the TensorFlow Embedding Projector.
Adjust visualization settings (e.g., PCA, t-SNE) to explore the embedding space.

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
