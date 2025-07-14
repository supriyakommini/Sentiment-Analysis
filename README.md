# 🎭 Multimodal Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-yellow.svg)

---

## 📌 Abstract

The **Multimodal Emotion Recognition System** aims to create a robust and accurate pipeline that recognizes human emotions by leveraging **three data modalities**:  
1. **Text** (sentiment analysis),  
2. **Speech** (audio emotion recognition), and  
3. **Facial Expressions** (video/image-based emotion detection).

By integrating **machine learning (ML)** and **deep learning (DL)** techniques across modalities, this system improves accuracy and reliability in emotion recognition applications like virtual assistants, healthcare monitoring, and human-computer interaction.

---

## 🛠️ Technologies & Libraries Used

- **Languages & Frameworks**: Python, Jupyter Notebook, TensorFlow/Keras
- **Computer Vision**: OpenCV, MediaPipe
- **ML/DL Libraries**: Scikit-learn, TensorFlow, Keras
- **Data Processing**: Numpy, Pandas, TF-IDF, Word2Vec, MFCC
- **Visualization**: Matplotlib, Seaborn
- **Sampling Techniques**: SMOTE for class balancing

---

## 📂 Datasets Used

- **📝 Text**: [Kaggle Emotion NLP Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
  - Files: `train.txt`, `test.txt`, `val.txt`
- **🔊 Speech**: [RAVDESS Emotional Speech Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **😐 Facial**: [FER-2013 Emotion Recognition Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🧠 Model Architectures

### 📈 Machine Learning Models
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

### 🧠 Deep Learning Models
- Convolutional Neural Networks (CNNs)
- Gradient Boosting (XGBoost, etc.)

### ✅ Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## ⚙️ Implementation Overview

### 1. **Text Emotion Recognition**
- NLP preprocessing: tokenization, stopword removal, lemmatization
- Feature extraction: TF-IDF and Word2Vec embeddings
- Model training using ML classifiers
- Evaluation using accuracy, F1-score, and confusion matrix

### 2. **Speech Emotion Recognition**
- Feature extraction: MFCC (Mel Frequency Cepstral Coefficients)
- Training using CNNs and ML models (Random Forest, SVM, etc.)
- Audio classification into emotional states

### 3. **Facial Emotion Detection**
- Facial landmark detection using **MediaPipe FaceMesh**
- Feature vector creation based on landmark distances and angles
- Classification using Random Forest and CNN
- **Real-time emotion prediction** using webcam + OpenCV

---

## 📊 Results Summary

| Modality | Best Model         | Accuracy |
|----------|--------------------|----------|
| **Text** | Random Forest       | ~88.9%   |
| **Speech** | CNN               | ~69.0%   |
| **Facial** | Random Forest     | ~55.0%   |

---

## 📁 Project Structure

