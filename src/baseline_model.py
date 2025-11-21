from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        
    def train(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        
    def predict(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)
    
    def save_model(self, filepath):
        joblib.dump({'vectorizer': self.vectorizer, 'model': self.model}, filepath)