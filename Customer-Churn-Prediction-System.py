import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import joblib

class ChurnPredictionSystem:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_columns = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
        
    def load_and_prepare_data(self, filepath):
        """Load and prepare the telco customer churn dataset."""
        # Load data
        df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
        
        # Clean data
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
        
        # Encode categorical variables
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
            
        # Prepare features and target
        X = df[self.numerical_columns + self.categorical_columns]
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        return X, y
    
    def train_model(self, X, y):
        """Train the churn prediction model."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        return X_test, y_test, y_pred
    
    def analyze_results(self, X_test, y_test, y_pred):
        """Analyze and visualize model results."""
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create feature importance plot
        feature_importance = pd.DataFrame({
            'feature': self.numerical_columns + self.categorical_columns,
            'importance': self.model.named_steps['classifier'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_churn_probability(self, customer_data):
        """Predict churn probability for new customers."""
        return self.model.predict_proba(customer_data)[:, 1]
    
    def save_model(self, filepath):
        """Save the trained model."""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)

# Example usage
def main():
    # Initialize the system
    churn_system = ChurnPredictionSystem()
    
    # Load and prepare data
    print("Loading and preparing data...")
    X, y = churn_system.load_and_prepare_data('telco_churn.csv')
    
    # Train the model
    print("\nTraining model...")
    X_test, y_test, y_pred = churn_system.train_model(X, y)
    
    # Analyze results
    print("\nAnalyzing results...")
    feature_importance = churn_system.analyze_results(X_test, y_test, y_pred)
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Example prediction for a new customer
    print("\nExample prediction for a new customer:")
    new_customer = pd.DataFrame({
        'tenure': [36],
        'MonthlyCharges': [65.0],
        'TotalCharges': [2340.0],
        'gender': [1],  # encoded value
        'InternetService': [2],  # encoded value
        'Contract': [1],  # encoded value
        'PaymentMethod': [3]  # encoded value
    })
    
    churn_prob = churn_system.predict_churn_probability(new_customer)
    print(f"Churn probability: {churn_prob[0]:.2%}")

if __name__ == "__main__":
    main()
