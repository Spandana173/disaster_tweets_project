# Disaster Tweet Classification with Fine-Tuned BERT for Emergency Response

## Project Overview
This project implements and optimizes deep learning models to classify disaster-related tweets for emergency response applications. The system distinguishes between genuine disaster reports and non-disaster tweets using both traditional machine learning and transformer-based approaches.

* **Team**: Spandana Kummari (skumm01), Sai Lahari Pathipati (spath01)
* **Course**: CS 6375 Deep Learning - Term Project
* **Timeline**: 6-week comprehensive project

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
│ ├── 3_initial_model_training.ipynb
│ ├── 4_hyperparameter_tuning.ipynb
│ └── 5_model_evaluation_comparison.ipynb
├── src/ # Python source modules
│ ├── data_loader.py # Data loading and preprocessing
│ ├── baseline_model.py # Logistic regression implementation
│ ├── bert_model.py # BERT model classes
│ └── init.py
├── models/ # Saved model files
│ └── optimized_bert/ # Fine-tuned BERT model
├── requirements.txt # Python dependencies
└── README.md # Project documentation


##  Dataset
* **Source**: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)
* **Samples**: 7,613 labeled tweets
* **Classes**: 
  * `1` = Disaster tweets (requests for help, emergency reports)
  * `0` = Non-disaster tweets (general conversations)
* **Features**: tweet text, keyword, location, target label

##  Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd disaster_tweets_project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

## File Descriptions

### Notebooks
* `1_data_preprocessing.ipynb` - Data loading, cleaning, and preparation
* `2_eda_visualization.ipynb` - Exploratory data analysis with charts and insights
* `3_initial_model_training.ipynb` - Baseline models and initial BERT implementation
* `4_hyperparameter_tuning.ipynb` - Systematic optimization experiments
* `5_model_evaluation_comparison.ipynb` - Comprehensive model evaluation

### Source Code
* `src/data_loader.py` - Data loading and preprocessing utilities
* `src/baseline_model.py` - Logistic Regression implementation
* `src/bert_model.py` - BERT model classes and training utilities

## Usage Examples

### Quick Prediction
```python
from src.bert_model import BERTClassifier

# Initialize classifier
classifier = BERTClassifier()

# Load optimized model
model_path = 'models/optimized_bert'

# Make predictions
tweets = [
    "Forest fire spreading rapidly near residential area",
    "Just had a great lunch with friends",
    "Earthquake felt in downtown area, buildings shaking"
]

for tweet in tweets:
    prediction, confidence = classifier.predict(tweet, model_path)
    status = "DISASTER" if prediction == 1 else "NON-DISASTER"
    print(f"Tweet: {tweet[:50]}... -> {status} (confidence: {confidence:.3f})")

### Training Custom Model
```python
from src.data_loader import DataLoader
from src.baseline_model import BaselineModel
from src.bert_model import BERTClassifier

# Load and preprocess data
loader = DataLoader('data/train.csv')
processed_data = loader.preprocess_data()

# Train Logistic Regression baseline
baseline = BaselineModel()
baseline.train(processed_data['cleaned_text'], processed_data['target'])

# Train BERT model
bert_classifier = BERTClassifier()
bert_classifier.train(processed_data, epochs=3, learning_rate=2e-5)

## Common Issues & Solutions

### Installation Issues
* **Issue**: Transformers library import errors
  * **Solution**: `pip install transformers torch` or use Google Colab
* **Issue**: NLTK data download failures
  * **Solution**: Run `nltk.download()` commands manually in notebook
* **Issue**: CUDA out of memory errors
  * **Solution**: Reduce batch size to 8 or use Google Colab GPU

### Training Issues
* **Issue**: BERT overfitting (high training accuracy, low validation)
  * **Solution**: Use early stopping, reduce epochs, add dropout
* **Issue**: Model predicting all ones or zeros
  * **Solution**: Ensure proper training loop and loss calculation
* **Issue**: Slow training times
  * **Solution**: Use GPU acceleration and smaller batch sizes

## Results Reproduction

### Exact Reproduction Steps
1. Run notebooks in numerical order (1 → 2 → 3 → 4 → 5)
2. Use random seed 42 for consistent results
3. Ensure GPU availability for BERT training
4. Use default hyperparameters from Notebook 4
5. Follow the same train-validation split (80-20)

### Expected Outputs
* Notebook 1: Cleaned dataset with 7,613 samples
* Notebook 2: Visualizations of data distribution and patterns
* Notebook 3: Baseline accuracy ~82%, Initial BERT ~79%
* Notebook 4: Optimized hyperparameters (lr=2e-5, batch=8, seq_len=64)
* Notebook 5: Final BERT accuracy ~81.2% with 84.7% disaster recall

##  License

This project is developed for **academic purposes** as part of **CS 6375 Deep Learning** coursework. The code is available for educational and research use.