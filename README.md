# Classifying Disaster Tweets with Fine-Tuned BERT for Emergency Response

## Project Overview
Deep learning project for classifying disaster-related tweets using BERT and traditional machine learning methods. This system automatically distinguishes between genuine emergency reports and casual conversations to support emergency response efforts.

## Project Goal
Build an automated system that can accurately identify disaster-related tweets with high recall to ensure no genuine emergency requests are missed, while maintaining good precision to minimize false alarms.

## Dataset
Source: Natural Language Processing with Disaster Tweets (Kaggle)

Samples: 7,613 labeled tweets

Classes: Disaster (1) vs Non-Disaster (0)

Features: Tweet text, keywords, location metadata

### Prerequisites
- Python 3.8+
- GPU recommended for BERT training

## Project Structure
disaster_tweets_project/
├── data/                    # Dataset files
├── notebooks/              # Jupyter notebooks
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_eda_visualization.ipynb
│   └── 3_initial_model_training.ipynb
├── src/                    # Python modules
├── models/                 # Saved models
├── requirements.txt        # Dependencies
└── README.md              # Documentation

## Quick Start
Install dependencies: pip install -r requirements.txt

Run notebooks in numerical order

Models will be trained and evaluated automatically

## Current Results
Logistic Regression: 81.9% accuracy, 77.3% F1-score

DistilBERT: 79.4% accuracy, 77.9% F1-score

Key Strength: BERT achieves 85% recall for disaster tweets

## Technologies Used
Python, PyTorch, Transformers

Scikit-learn, Pandas, NLTK

BERT, Logistic Regression, TF-IDF

## Team
Spandana Kummari (skumm01)

Sai Lahari Pathipati (spath01)

## Academic Project
This project is part of a deep learning course term project focused on disaster management applications.