# Disaster Tweet Classification with Deep Learning

## Project Overview
This project implements deep learning models to classify disaster-related tweets for emergency response applications. The system distinguishes between genuine disaster reports and non-disaster tweets using both traditional machine learning and transformer-based approaches.

## Team Members
- Spandana Kummari (skumm01)
- Sai Lahari Pathipati (spath01)

## Project Structure
disaster_tweets_project/
├── data/ # Dataset files
│ ├── train.csv # Original training data
│ ├── test.csv # Test data
│ ├── processed_train.csv # Preprocessed dataset
│ ├── train_split.csv # Training split (80%)
│ └── val_split.csv # Validation split (20%)
├── notebooks/ # Jupyter notebooks (execution order)
│ ├── 1_data_preprocessing.ipynb
│ ├── 2_eda_visualization.ipynb
│ └── 3_initial_model_training.ipynb
├── src/ # Python source modules
│ ├── data_loader.py # Data loading and preprocessing
│ ├── baseline_model.py # Logistic regression implementation
│ ├── bert_model.py # BERT model classes
│ └── init.py
├── models/ # Saved model files
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Dataset
- **Source**: Natural Language Processing with Disaster Tweets (Kaggle)
- **Samples**: 7,613 labeled tweets
- **Classes**: 
  - `1`: Disaster tweets (requests for help, emergency reports)
  - `0`: Non-disaster tweets (general conversations)
- **Features**: tweet text, keyword, location, target label

## Installation & Setup

### Prerequisites
- Python 3.8+
- GPU recommended for BERT training

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd disaster_tweets_project