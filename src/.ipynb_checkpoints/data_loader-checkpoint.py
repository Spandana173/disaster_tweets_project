import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def clean_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = ' '.join(text.split())
        return text
    
    def preprocess_data(self):
        processed_df = self.df.copy()
        processed_df['cleaned_text'] = processed_df['text'].apply(self.clean_text)
        processed_df['keyword'] = processed_df['keyword'].fillna('unknown')
        processed_df['location'] = processed_df['location'].fillna('unknown')
        return processed_df