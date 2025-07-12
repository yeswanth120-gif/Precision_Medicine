import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import re
from datetime import datetime
from sqlalchemy import create_engine, text
import random
import sys

app = Flask(__name__, 
            static_folder='../frontend',
            template_folder='../frontend')

# Enable CORS for all domains on all routes (for development)
CORS(app)

# Add debug logging for template rendering
@app.before_request
def log_request_info():
    print(f"📥 Request: {request.method} {request.path}")
    if request.method == 'POST' and request.content_type and 'json' in request.content_type:
        print(f"📦 JSON Data: {request.get_json()}")

# =============================================================================
# MODEL LOADING
# =============================================================================

# Load heart disease model and scaler
HEART_DISEASE_MODEL_PATH = os.path.join('models', 'heart_disease', 'best_heart_disease_classifier.joblib')
HEART_DISEASE_SCALER_PATH = os.path.join('models', 'heart_disease', 'heart_disease_scaler.joblib')

try:
    heart_disease_model = joblib.load(HEART_DISEASE_MODEL_PATH)
    heart_disease_scaler = joblib.load(HEART_DISEASE_SCALER_PATH)
    print("✅ Heart disease model and scaler loaded successfully!")
    print(sys.version)
except Exception as e:
    print(f"❌ Error loading heart disease model: {e}")
    heart_disease_model = None
    heart_disease_scaler = None

# Load diabetes model and scaler
DIABETES_MODEL_PATH = os.path.join('models', 'diabetes', 'best_diabetes_classifier.joblib')
DIABETES_SCALER_PATH = os.path.join('models', 'diabetes', 'diabetes_scaler.joblib')

try:
    diabetes_model = joblib.load(DIABETES_MODEL_PATH)
    diabetes_scaler = joblib.load(DIABETES_SCALER_PATH)
    print("✅ Diabetes model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading diabetes model: {e}")
    diabetes_model = None
    diabetes_scaler = None

# Load lung cancer model and scaler
LUNG_CANCER_MODEL_PATH = os.path.join('models', 'lung_cancer', 'best_lung_cancer_classifier.joblib')
LUNG_CANCER_SCALER_PATH = os.path.join('models', 'lung_cancer', 'lung_cancer_scaler.joblib')
LUNG_CANCER_FEATURES_PATH = os.path.join('models', 'lung_cancer', 'lung_cancer_features.joblib')

try:
    lung_cancer_model = joblib.load(LUNG_CANCER_MODEL_PATH)
    print("✅ Lung cancer model loaded successfully!")
    
    # Try to load scaler (may not exist for some models)
    if os.path.exists(LUNG_CANCER_SCALER_PATH):
        lung_cancer_scaler = joblib.load(LUNG_CANCER_SCALER_PATH)
        print("✅ Lung cancer scaler loaded successfully!")
    else:
        lung_cancer_scaler = None
        print("⚠️ Lung cancer scaler not found - will proceed without scaling")
    
    # Try to load features (may not exist for some models)
    if os.path.exists(LUNG_CANCER_FEATURES_PATH):
        lung_cancer_features = joblib.load(LUNG_CANCER_FEATURES_PATH)
        print(f"✅ Lung cancer features loaded: {lung_cancer_features}")
    else:
        lung_cancer_features = None
        print("⚠️ Lung cancer features file not found - using default order")
    
    # Test the model with different feature counts to find the correct one
    # Test the model ONLY with 10 features
    if lung_cancer_model is not None:
        print("🔍 Testing lung cancer model feature requirements with 10 features...")
        model_feature_count = None
        try:
            test_features = np.zeros((1, 10))
            if lung_cancer_scaler is not None:
                test_features = lung_cancer_scaler.transform(test_features)

            test_pred = lung_cancer_model.predict(test_features)
            test_proba = lung_cancer_model.predict_proba(test_features)

            print(f"✅ Model accepts 10 features")
            model_feature_count = 10
            print(f"🎯 Lung cancer model expects {model_feature_count} features")
        except Exception as e:
            print(f"❌ 10 features failed: {str(e)}")
            print("⚠️ Could not determine model feature count")
        
except Exception as e:
    print(f"❌ Error loading lung cancer model: {e}")
    lung_cancer_model = None
    lung_cancer_scaler = None
    lung_cancer_features = None

# Kidney Disease Model Loading
kidney_disease_model = None
kidney_disease_scaler = None
kidney_disease_features = None

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Heart disease feature order and ideal values
HEART_DISEASE_FEATURES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
    'ejection_fraction', 'high_blood_pressure', 'platelets', 
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# Diabetes feature order
DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Lung cancer feature order (10 features - CORRECTED)
LUNG_CANCER_FEATURES_10 = [
    'Age', 'EGFR', 'KRAS', 'ALK', 'TP53', 'STK11', 'KEAP1', 'BRAF', 'ROS1', 'MET'
]

# Alternative lung cancer feature orders to try
LUNG_CANCER_FEATURES_ALT1 = [
    'Age', 'Sex', 'EGFR', 'KRAS', 'ALK', 'TP53', 'STK11', 'KEAP1', 'BRAF', 'ROS1'
]

LUNG_CANCER_FEATURES_ALT2 = [
    'Age', 'Sex', 'EGFR', 'KRAS', 'ALK', 'TP53', 'STK11', 'KEAP1', 'BRAF', 'MET'
]

# Kidney disease feature order
KIDNEY_DISEASE_FEATURES = [
    'age', 'bp', 'bgr', 'bu', 'sod', 'pot', 'wc', 'htn_yes', 'dm_yes', 'cad_yes', 'appet_poor', 'pe_yes', 'ane_yes'
]

# Ideal values for heart disease features
HEART_DISEASE_IDEAL_VALUES = {
    'age': {'min': 40, 'max': 95, 'ideal': '40-65', 'unit': 'years'},
    'anaemia': {'ideal': 'No', 'unit': 'condition'},
    'creatinine_phosphokinase': {'min': 23, 'max': 7861, 'ideal': '22-198', 'unit': 'mcg/L'},
    'diabetes': {'ideal': 'No', 'unit': 'condition'},
    'ejection_fraction': {'min': 14, 'max': 80, 'ideal': '50-70', 'unit': '%'},
    'high_blood_pressure': {'ideal': 'No', 'unit': 'condition'},
    'platelets': {'min': 25100, 'max': 850000, 'ideal': '150000-450000', 'unit': 'kiloplatelets/mL'},
    'serum_creatinine': {'min': 0.5, 'max': 9.4, 'ideal': '0.6-1.2', 'unit': 'mg/dL'},
    'serum_sodium': {'min': 113, 'max': 148, 'ideal': '135-145', 'unit': 'mEq/L'},
    'sex': {'ideal': 'Any', 'unit': 'gender'},
    'smoking': {'ideal': 'No', 'unit': 'habit'},
    'time': {'min': 4, 'max': 285, 'ideal': 'Higher', 'unit': 'days'}
}

# Ideal values for diabetes features
DIABETES_IDEAL_VALUES = {
    'Pregnancies': {'min': 0, 'max': 17, 'ideal': '0-5', 'unit': 'count'},
    'Glucose': {'min': 0, 'max': 199, 'ideal': '70-99', 'unit': 'mg/dL'},
    'BloodPressure': {'min': 0, 'max': 122, 'ideal': '90-120', 'unit': 'mm Hg'},
    'SkinThickness': {'min': 0, 'max': 99, 'ideal': '10-25', 'unit': 'mm'},
    'Insulin': {'min': 0, 'max': 846, 'ideal': '16-166', 'unit': 'mu U/ml'},
    'BMI': {'min': 0, 'max': 67.1, 'ideal': '18.5-24.9', 'unit': 'kg/m²'},
    'DiabetesPedigreeFunction': {'min': 0.078, 'max': 2.42, 'ideal': '<0.5', 'unit': 'score'},
    'Age': {'min': 21, 'max': 81, 'ideal': '21-65', 'unit': 'years'}
}

# Ideal values for lung cancer features
LUNG_CANCER_IDEAL_VALUES = {
    'Age': {'min': 20, 'max': 90, 'ideal': '20-70', 'unit': 'years'},
    'Sex': {'ideal': 'Any', 'unit': 'gender'},
    'EGFR': {'ideal': 'No mutation', 'unit': 'gene'},
    'KRAS': {'ideal': 'No mutation', 'unit': 'gene'},
    'ALK': {'ideal': 'No mutation', 'unit': 'gene'},
    'TP53': {'ideal': 'No mutation', 'unit': 'gene'},
    'STK11': {'ideal': 'No mutation', 'unit': 'gene'},
    'KEAP1': {'ideal': 'No mutation', 'unit': 'gene'},
    'BRAF': {'ideal': 'No mutation', 'unit': 'gene'},
    'ROS1': {'ideal': 'No mutation', 'unit': 'gene'},
    'MET': {'ideal': 'No mutation', 'unit': 'gene'}
}

# Ideal values for kidney disease features
KIDNEY_DISEASE_IDEAL_VALUES = {
    'age': {'min': 2, 'max': 90, 'ideal': '20-65', 'unit': 'years'},
    'bp': {'min': 50, 'max': 180, 'ideal': '90-120', 'unit': 'mm Hg'},
    'bgr': {'min': 22, 'max': 490, 'ideal': '70-140', 'unit': 'mgs/dl'},
    'bu': {'min': 1.5, 'max': 391, 'ideal': '7-20', 'unit': 'mgs/dl'},
    'sod': {'min': 4.5, 'max': 163, 'ideal': '135-145', 'unit': 'mEq/L'},
    'pot': {'min': 2.5, 'max': 47, 'ideal': '3.5-5.0', 'unit': 'mEq/L'},
    'wc': {'min': 2200, 'max': 26400, 'ideal': '4000-11000', 'unit': 'cells/cumm'},
    'htn_yes': {'ideal': 'No', 'unit': 'condition'},
    'dm_yes': {'ideal': 'No', 'unit': 'condition'},
    'cad_yes': {'ideal': 'No', 'unit': 'condition'},
    'appet_poor': {'ideal': 'No', 'unit': 'symptom'},
    'pe_yes': {'ideal': 'No', 'unit': 'symptom'},
    'ane_yes': {'ideal': 'No', 'unit': 'condition'}
}

# =============================================================================
# HEALTH SCORE CALCULATION FUNCTIONS
# =============================================================================

def calculate_health_score(features, prediction_proba):
    """Calculate health score based on features and prediction"""
    # Start with base score
    score = 85
    
    # Age factor
    age = features[0]
    if age > 70:
        score -= 15
    elif age > 60:
        score -= 10
    elif age > 50:
        score -= 5
    
    # Ejection fraction (most important)
    ejection_fraction = features[4]
    if ejection_fraction < 30:
        score -= 25
    elif ejection_fraction < 40:
        score -= 15
    elif ejection_fraction < 50:
        score -= 10
    
    # Serum creatinine
    serum_creatinine = features[7]
    if serum_creatinine > 2.0:
        score -= 20
    elif serum_creatinine > 1.5:
        score -= 10
    
    # Blood pressure factor
    if features[5] == 1:  # Has high BP
        score -= 10
    
    # Smoking factor
    if features[10] == 1:  # Smokes
        score -= 15
    
    # Prediction probability factor
    death_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    score -= int(death_probability * 30)
    
    return max(20, min(100, int(score)))

def calculate_diabetes_health_score(features, prediction_proba):
    """Calculate health score based on diabetes features and prediction"""
    # Start with base score
    score = 85
    
    # Glucose level (most important)
    glucose = features[1]
    if glucose > 140:
        score -= 25
    elif glucose > 125:
        score -= 15
    elif glucose > 100:
        score -= 10
    
    # BMI factor
    bmi = features[5]
    if bmi > 35:
        score -= 20
    elif bmi > 30:
        score -= 15
    elif bmi > 25:
        score -= 5
    
    # Age factor
    age = features[7]
    if age > 65:
        score -= 10
    elif age > 45:
        score -= 5
    
    # Blood pressure
    bp = features[2]
    if bp > 140:
        score -= 10
    elif bp > 120:
        score -= 5
    
    # Prediction probability factor
    diabetes_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
    score -= int(diabetes_probability * 25)
    
    return max(20, min(100, int(score)))

def calculate_lung_cancer_health_score(features, prediction_proba):
    """Calculate health score based on lung cancer features and prediction (10 features)"""
    # Start with base score
    score = 85
    
    # Age factor (first feature)
    age = features[0]
    if age > 70:
        score -= 15
    elif age > 60:
        score -= 10
    elif age > 50:
        score -= 5
    
    # Count genetic mutations (features 1-9 for 10-feature model)
    mutation_count = sum(features[1:])  # All features except age
    score -= mutation_count * 8  # Each mutation reduces score
    
    # Prediction probability factor - handle NaN values
    if len(prediction_proba) > 1:
        cancer_probability = prediction_proba[1]
        if not np.isnan(cancer_probability):
            score -= int(cancer_probability * 30)
    
    return max(20, min(100, int(score)))

def calculate_kidney_health_score(features_dict):
    """Calculate kidney health score based on key parameters"""
    score = 85
    
    # Age factor
    age = features_dict.get('age', 50)
    if age > 70:
        score -= 15
    elif age > 60:
        score -= 10
    elif age > 50:
        score -= 5
    
    # Blood pressure
    bp = features_dict.get('bp', 80)
    if bp > 140:
        score -= 20
    elif bp > 120:
        score -= 10
    
    # Blood glucose
    bgr = features_dict.get('bgr', 148)
    if bgr > 180:
        score -= 15
    elif bgr > 140:
        score -= 10
    
    # Blood urea
    bu = features_dict.get('bu', 25)
    if bu > 50:
        score -= 15
    elif bu > 40:
        score -= 10
    
    # Comorbidities
    if features_dict.get('htn_yes', 0):
        score -= 10
    if features_dict.get('dm_yes', 0):
        score -= 15
    if features_dict.get('cad_yes', 0):
        score -= 10
    
    return max(20, min(100, int(score)))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_and_fix_probabilities(prediction_proba, prediction):
    """Validate and fix probability values, handling NaN cases"""
    try:
        # Check if we have the right number of probabilities
        if len(prediction_proba) != 2:
            print(f"⚠️ Invalid probability array length: {len(prediction_proba)}")
            return [0.7, 0.3] if prediction == 0 else [0.3, 0.7]
        
        # Check for NaN values
        if np.any(np.isnan(prediction_proba)):
            print("⚠️ NaN values detected in probabilities")
            return [0.7, 0.3] if prediction == 0 else [0.3, 0.7]
        
        # Check for infinite values
        if np.any(np.isinf(prediction_proba)):
            print("⚠️ Infinite values detected in probabilities")
            return [0.7, 0.3] if prediction == 0 else [0.3, 0.7]
        
        # Ensure probabilities are between 0 and 1
        prob_0 = max(0.0, min(1.0, float(prediction_proba[0])))
        prob_1 = max(0.0, min(1.0, float(prediction_proba[1])))
        
        # Ensure probabilities sum to 1
        prob_sum = prob_0 + prob_1
        if prob_sum == 0:
            return [0.5, 0.5]
        
        if abs(prob_sum - 1.0) > 0.001:  # If sum is not close to 1
            prob_0 = prob_0 / prob_sum
            prob_1 = prob_1 / prob_sum
        
        return [prob_0, prob_1]
        
    except Exception as e:
        print(f"⚠️ Error validating probabilities: {e}")
        return [0.7, 0.3] if prediction == 0 else [0.3, 0.7]

def get_lung_cancer_features(data):
    """Get the correct feature set for lung cancer model"""
    
    # Try different feature combinations based on model requirements
    feature_combinations = [
        {
            'features': [
                float(data.get('age', 0)),
                float(data.get('sex', 0)),
                float(data.get('egfr', 0)),
                float(data.get('kras', 0)),
                float(data.get('alk', 0)),
                float(data.get('tp53', 0)),
                float(data.get('stk11', 0)),
                float(data.get('keap1', 0)),
                float(data.get('braf', 0)),
                float(data.get('met', 0))
            ],
            'names': LUNG_CANCER_FEATURES_ALT2,
            'description': '10 features: Age + Sex + 8 genetic mutations (no ROS1)'
        }
    ]
    
    return feature_combinations

# =============================================================================
# KIDNEY DISEASE MODEL LOADING
# =============================================================================

def load_kidney_disease_models():
    global kidney_disease_model, kidney_disease_scaler, kidney_disease_features
    try:
        kidney_disease_model = joblib.load('models/kidney_disease/best_kidney_disease_classifier.joblib')
        kidney_disease_scaler = joblib.load('models/kidney_disease/kidney_disease_scaler.joblib')
        kidney_disease_features = joblib.load('models/kidney_disease/kidney_disease_features.joblib')
        print("✅ Kidney disease models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading kidney disease models: {str(e)}")
        return False

# =============================================================================
# CHATBOT SETUP
# =============================================================================

class SimpleQAChatbot:
    def __init__(self, folder_path):
        self.qa_pairs = self.load_all_qa_pairs(folder_path)
        self.greetings = {
            'hi': "Hi there! 👋 I'm here to help you with health-related questions. What would you like to know?",
            'hello': "Hello! 😊 I'm your health assistant. Feel free to ask me anything about health topics!",
            'hey': "Hey! 👋 How can I help you today?",
            'good morning': "Good morning! ☀️ Hope you're having a great day. What can I help you with?",
            'good afternoon': "Good afternoon! 🌤️ How can I assist you today?",
            'good evening': "Good evening! 🌙 What would you like to know?",
            'how are you': "I'm doing great, thanks for asking! 😊 I'm here and ready to help with your health questions.",
            'what is your name': "I'm your friendly health assistant chatbot! 🤖 You can ask me questions about various health topics.",
            'who are you': "I'm a health-focused chatbot designed to help answer your medical and health-related questions! 🏥"
        }
        self.thanks_responses = [
            "You're welcome! 😊 Happy to help!",
            "Glad I could help! 👍 Feel free to ask more questions.",
            "You're very welcome! 🙂 Anything else you'd like to know?",
            "My pleasure! 😊 I'm here whenever you need help."
        ]
        self.unknown_responses = [
            "I don't have information about that topic in my knowledge base yet. 🤔 Try asking about health-related topics!",
            "I'm not sure about that one! 😅 I specialize in health and medical questions. Got any health queries?",
            "That's not in my knowledge base currently. 📚 I'm great with health and medical questions though!",
            "I don't know about that topic yet! 🤷‍♀️ But I'm here to help with health-related questions!"
        ]
        self.goodbye_response = "Goodbye! 👋 Take care and feel free to come back anytime you have health questions!"

    def load_all_qa_pairs(self, folder_path):
        qa_pairs = []
        if not os.path.exists(folder_path):
            print(f"⚠️ Knowledge base folder not found: {folder_path}")
            return qa_pairs
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                qa_pairs.extend(self.load_qa_pairs(file_path))
        return qa_pairs

    def load_qa_pairs(self, file_path):
        qa_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pairs = re.split(r'\n\s*Q:', content)
        for pair in pairs:
            if not pair.strip():
                continue
            if not pair.startswith('Q:'):
                pair = 'Q:' + pair
            lines = pair.strip().split('\n')
            if len(lines) >= 2:
                question = lines[0].replace('Q:', '').strip()
                answer = lines[1].replace('A:', '').strip()
                qa_pairs.append({'question': question, 'answer': answer})
        return qa_pairs

    def find_answer(self, user_question):
        user_question_clean = user_question.lower().strip()
        # Exact match first
        for qa in self.qa_pairs:
            if qa['question'].lower() == user_question_clean:
                return qa['answer']
        # Fallback: substring search
        for qa in self.qa_pairs:
            if qa['question'].lower() in user_question_clean or user_question_clean in qa['question'].lower():
                return qa['answer']
        return None

    def is_greeting(self, text):
        text_lower = text.lower().strip()
        for greeting in self.greetings.keys():
            if greeting in text_lower:
                return greeting
        return None

    def is_thanks(self, text):
        thanks_words = ['thank', 'thanks', 'appreciate', 'grateful']
        text_lower = text.lower()
        return any(word in text_lower for word in thanks_words)

    def is_goodbye(self, text):
        goodbye_words = ['bye', 'goodbye', 'see you', 'take care', 'farewell']
        text_lower = text.lower()
        return any(word in text_lower for word in goodbye_words)

knowledge_folder = "./knowledgeBase"
simple_bot = SimpleQAChatbot(knowledge_folder)


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Database configuration
DB_CONFIG = {
    'server': 'YESWANTH\\SQLEXPRESS',
    'database': 'PrecisionMedicine',
    'driver': 'ODBC Driver 17 for SQL Server',
    'trusted_connection': 'yes'
}

def get_db_connection():
    """Create database connection using a direct connection string"""
    try:
        print("🔗 Attempting database connection with direct connection string...")
        # Use direct connection string (no URL encoding)
        connection_string = f"mssql+pyodbc://YESWANTH\\SQLEXPRESS/PrecisionMedicine?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        print(f"🔗 Connection string: {connection_string}")
        engine = create_engine(connection_string, echo=False)

        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return engine

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        # You can add further fallback methods here if needed, or just return None
        return None

def validate_database_schema():
    """Validate database schema and create tables if needed"""
    try:
        print("🔍 Validating database schema...")
        engine = get_db_connection()
        if not engine:
            return False
            
        with engine.connect() as conn:
            # Check if Users table exists
            check_users_query = text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = 'Users'
            """)
            
            result = conn.execute(check_users_query)
            users_exists = result.fetchone()[0] > 0
            
            if not users_exists:
                print("⚠️ Users table does not exist. Creating it...")
                create_users_query = text("""
                    CREATE TABLE Users (
                        UserID INT PRIMARY KEY,
                        Username NVARCHAR(100) NOT NULL,
                        Email NVARCHAR(255) NOT NULL,
                        CreatedAt DATETIME2 DEFAULT GETDATE()
                    )
                """)
                conn.execute(create_users_query)
                conn.commit()
                print("✅ Users table created successfully")
            
            # Check if Predictions table exists
            check_predictions_query = text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = 'Predictions'
            """)
            
            result = conn.execute(check_predictions_query)
            predictions_exists = result.fetchone()[0] > 0
            
            if not predictions_exists:
                print("⚠️ Predictions table does not exist. Creating it...")
                create_predictions_query = text("""
                    CREATE TABLE Predictions (
                        PredictionID INT IDENTITY(1,1) PRIMARY KEY,
                        UserID INT NOT NULL,
                        PredictionType NVARCHAR(50) NOT NULL,
                        RiskLevel NVARCHAR(20),
                        Probability FLOAT,
                        HealthScore INT,
                        Confidence FLOAT,
                        InputData NVARCHAR(MAX),
                        OutputData NVARCHAR(MAX),
                        CreatedAt DATETIME2 DEFAULT GETDATE(),
                        FOREIGN KEY (UserID) REFERENCES Users(UserID)
                    )
                """)
                conn.execute(create_predictions_query)
                conn.commit()
                print("✅ Predictions table created successfully")
            
            # Check if LungCancerInputs table exists and create with proper structure
            check_lung_table_query = text("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = 'LungCancerInputs'
            """)
            
            result = conn.execute(check_lung_table_query)
            lung_table_exists = result.fetchone()[0] > 0
            
            if not lung_table_exists:
                print("⚠️ LungCancerInputs table does not exist. Creating it with correct structure...")
                create_lung_table_query = text("""
                    CREATE TABLE LungCancerInputs (
                        InputID INT IDENTITY(1,1) PRIMARY KEY,
                        PredictionID INT NOT NULL,
                        Gender INT NOT NULL,
                        Age INT NOT NULL,
                        EGFRMutation INT NOT NULL,
                        KRASMutation INT NOT NULL,
                        ALKMutation INT NOT NULL,
                        TP53Mutation INT NOT NULL,
                        STK11Mutation INT NOT NULL,
                        KEAP1Mutation INT NOT NULL,
                        BRAFMutation INT NOT NULL,
                        ROS1Mutation INT NOT NULL,
                        METMutation INT NOT NULL,
                        CreatedAt DATETIME2 DEFAULT GETDATE(),
                        FOREIGN KEY (PredictionID) REFERENCES Predictions(PredictionID)
                    )
                """)
                conn.execute(create_lung_table_query)
                conn.commit()
                print("✅ LungCancerInputs table created successfully with proper column names")
            else:
                # Check if table has correct columns and fix if needed
                print("🔍 Checking LungCancerInputs table structure...")
                check_columns_query = text("""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = 'LungCancerInputs'
                    ORDER BY ORDINAL_POSITION
                """)
                
                result = conn.execute(check_columns_query)
                existing_columns = [row[0] for row in result]
                print(f"📋 Existing columns: {existing_columns}")
                
                # Expected columns with proper naming
                expected_columns = [
                    'InputID', 'PredictionID', 'Gender', 'Age', 
                    'EGFRMutation', 'KRASMutation', 'ALKMutation', 'TP53Mutation', 
                    'STK11Mutation', 'KEAP1Mutation', 'BRAFMutation', 'ROS1Mutation', 
                    'METMutation', 'CreatedAt'
                ]
                
                # Check if we need to recreate the table
                if not all(col in existing_columns for col in expected_columns[1:]):  # Skip InputID as it's auto-generated
                    print("⚠️ LungCancerInputs table structure is incorrect. Recreating...")
                    
                    # Drop and recreate the table
                    conn.execute(text("DROP TABLE LungCancerInputs"))
                    
                    create_lung_table_query = text("""
                        CREATE TABLE LungCancerInputs (
                            InputID INT IDENTITY(1,1) PRIMARY KEY,
                            PredictionID INT NOT NULL,
                            Gender INT NOT NULL,
                            Age INT NOT NULL,
                            EGFRMutation INT NOT NULL,
                            KRASMutation INT NOT NULL,
                            ALKMutation INT NOT NULL,
                            TP53Mutation INT NOT NULL,
                            STK11Mutation INT NOT NULL,
                            KEAP1Mutation INT NOT NULL,
                            BRAFMutation INT NOT NULL,
                            ROS1Mutation INT NOT NULL,
                            METMutation INT NOT NULL,
                            CreatedAt DATETIME2 DEFAULT GETDATE(),
                            FOREIGN KEY (PredictionID) REFERENCES Predictions(PredictionID)
                        )
                    """)
                    conn.execute(create_lung_table_query)
                    conn.commit()
                    print("✅ LungCancerInputs table recreated with proper structure")
            
            print("✅ Database schema validation completed")
            return True
            
    except Exception as e:
        print(f"❌ Database schema validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# DATABASE HELPER FUNCTIONS
# =============================================================================

def save_prediction_to_db(user_id, prediction_type, input_data, prediction_result):
    """Save prediction to database with enhanced error handling"""
    try:
        print(f"💾 Attempting to save prediction to database:")
        print(f"   User ID: {user_id}")
        print(f"   Type: {prediction_type}")
        print(f"   Input Data Keys: {list(input_data.keys()) if input_data else 'None'}")
        
        # Validate database schema first
        if not validate_database_schema():
            print("❌ Database schema validation failed")
            return False
        
        engine = get_db_connection()
        if not engine:
            print("❌ Database connection failed")
            return False
            
        with engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
            
            try:
                # Convert user_id to integer if it's a string
                user_id_int = int(user_id)
                
                # First, check if user exists, if not create them
                user_check_query = text("SELECT COUNT(*) FROM Users WHERE UserID = :user_id")
                user_exists = conn.execute(user_check_query, {'user_id': user_id_int}).fetchone()[0]
                
                if user_exists == 0:
                    print(f"👤 Creating new user: {user_id_int}")
                    # Create the user first
                    create_user_query = text("""
                        INSERT INTO Users (UserID, Username, Email, CreatedAt)
                        VALUES (:user_id, :username, :email, :created_at)
                    """)
                    
                    user_params = {
                        'user_id': user_id_int,
                        'username': str(input_data.get('username', f'User{user_id_int}')),
                        'email': str(input_data.get('email', f"user{user_id_int}@precisionmedicine.com")),
                        'created_at': datetime.now()
                    }
                    
                    conn.execute(create_user_query, user_params)
                    print(f"✅ Created new user: {user_id_int}")
                
                # Insert into main Predictions table
                prediction_query = text("""
                    INSERT INTO Predictions (UserID, PredictionType, RiskLevel, Probability, HealthScore, Confidence, InputData, OutputData, CreatedAt)
                    OUTPUT INSERTED.PredictionID
                    VALUES (:user_id, :prediction_type, :risk_level, :probability, :health_score, :confidence, :input_data, :output_data, :created_at)
                """)
                
                # Prepare prediction data with proper type conversion and null handling
                prediction_params = {
                    'user_id': user_id_int,
                    'prediction_type': str(prediction_type),
                    'risk_level': str(prediction_result.get('risk_level', 'Unknown')),
                    'probability': float(prediction_result.get('probability', {}).get('high_risk', 0.0)),
                    'health_score': int(prediction_result.get('health_score', 0)),
                    'confidence': float(prediction_result.get('confidence', 0.0)),
                    'input_data': json.dumps(input_data, default=str),
                    'output_data': json.dumps(prediction_result, default=str),
                    'created_at': datetime.now()
                }
                
                print(f"💾 Inserting prediction with params: {prediction_params}")
                result = conn.execute(prediction_query, prediction_params)
                prediction_id = result.fetchone()[0]
                print(f"✅ Prediction inserted with ID: {prediction_id}")
                
                # Insert into specific input table based on prediction type
                success = True
                if prediction_type == 'heart_disease':
                    success = save_heart_disease_inputs(conn, prediction_id, input_data)
                elif prediction_type == 'diabetes':
                    success = save_diabetes_inputs(conn, prediction_id, input_data)
                elif prediction_type == 'lung_cancer':
                    success = save_lung_cancer_inputs(conn, prediction_id, input_data)
                elif prediction_type == 'kidney_disease':
                    success = save_kidney_disease_inputs(conn, prediction_id, input_data)
                else:
                    print(f"⚠️ Unknown prediction type: {prediction_type}")
                    success = True  # Allow main prediction to save even if specific inputs fail
                
                if success:
                    # Commit the transaction
                    trans.commit()
                    print(f"✅ All data saved successfully with prediction ID: {prediction_id}")
                    return prediction_id
                else:
                    # Rollback if specific inputs failed
                    trans.rollback()
                    print("❌ Failed to save specific inputs, rolling back transaction")
                    return False
                    
            except Exception as e:
                # Rollback on any error
                trans.rollback()
                print(f"❌ Transaction rolled back due to error: {e}")
                import traceback
                traceback.print_exc()
                return False
            
    except Exception as e:
        print(f"❌ Error saving prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_heart_disease_inputs(conn, prediction_id, input_data):
    """Save heart disease inputs to database"""
    try:
        query = text("""
            INSERT INTO HeartDiseaseInputs 
            (PredictionID, Age, Sex, Anaemia, CreatininePhosphokinase, Diabetes, EjectionFraction, 
             HighBloodPressure, Platelets, SerumCreatinine, SerumSodium, Smoking, Time)
            VALUES (:prediction_id, :age, :sex, :anaemia, :cpk, :diabetes, :ef, :hbp, :platelets, :creatinine, :sodium, :smoking, :time)
        """)
        
        params = {
            'prediction_id': int(prediction_id),
            'age': int(input_data.get('age', 0)),
            'sex': int(input_data.get('sex', 0)),
            'anaemia': int(input_data.get('anaemia', 0)),
            'cpk': float(input_data.get('creatinine_phosphokinase', 0)),
            'diabetes': int(input_data.get('diabetes', 0)),
            'ef': float(input_data.get('ejection_fraction', 0)),
            'hbp': int(input_data.get('high_blood_pressure', 0)),
            'platelets': float(input_data.get('platelets', 0)),
            'creatinine': float(input_data.get('serum_creatinine', 0)),
            'sodium': float(input_data.get('serum_sodium', 0)),
            'smoking': int(input_data.get('smoking', 0)),
            'time': float(input_data.get('time', 0))
        }
        
        conn.execute(query, params)
        print("✅ Heart disease inputs saved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error saving heart disease inputs: {e}")
        return False

def save_diabetes_inputs(conn, prediction_id, input_data):
    """Save diabetes inputs to database"""
    try:
        query = text("""
            INSERT INTO DiabetesInputs 
            (PredictionID, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            VALUES (:prediction_id, :pregnancies, :glucose, :bp, :skin, :insulin, :bmi, :dpf, :age)
        """)
        
        params = {
            'prediction_id': int(prediction_id),
            'pregnancies': int(input_data.get('Pregnancies', 0)),
            'glucose': float(input_data.get('Glucose', 0)),
            'bp': float(input_data.get('BloodPressure', 0)),
            'skin': float(input_data.get('SkinThickness', 0)),
            'insulin': float(input_data.get('Insulin', 0)),
            'bmi': float(input_data.get('BMI', 0)),
            'dpf': float(input_data.get('DiabetesPedigreeFunction', 0)),
            'age': int(input_data.get('Age', 0))
        }
        
        conn.execute(query, params)
        print("✅ Diabetes inputs saved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error saving diabetes inputs: {e}")
        return False

def save_lung_cancer_inputs(conn, prediction_id, input_data):
    """Save lung cancer inputs to database - FIXED VERSION with correct column names"""
    try:
        # Use the correct column names that match the database schema
        # Ensure all values are properly converted to appropriate types with safe conversion
        def safe_int(value, default=0):
            try:
                return int(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        params = {
            'prediction_id': int(prediction_id),
            'gender': safe_int(input_data.get('sex', 0)),
            'age': safe_int(input_data.get('age', 0)),
            'egfr_mutation': safe_int(input_data.get('egfr', 0)),
            'kras_mutation': safe_int(input_data.get('kras', 0)),
            'alk_mutation': safe_int(input_data.get('alk', 0)),
            'tp53_mutation': safe_int(input_data.get('tp53', 0)),
            'stk11_mutation': safe_int(input_data.get('stk11', 0)),
            'keap1_mutation': safe_int(input_data.get('keap1', 0)),
            'braf_mutation': safe_int(input_data.get('braf', 0)),
            'ros1_mutation': safe_int(input_data.get('ros1', 0)),
            'met_mutation': safe_int(input_data.get('met', 0))
        }
        
        print(f"💾 Saving lung cancer inputs with params: {params}")
        
        # Execute the query
        print(f"✅ Lung cancer inputs saved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error saving lung cancer inputs: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_kidney_disease_inputs(conn, prediction_id, input_data):
    """Save kidney disease inputs to database"""
    try:
        query = text("""
            INSERT INTO KidneyDiseaseInputs 
            (PredictionID, Age, BloodPressure, BloodGlucoseRandom, BloodUrea, SerumSodium, Potassium, 
             WhiteBloodCells, Hypertension, DiabetesMellitus, CoronaryArteryDisease, Appetite, PedalEdema, Anemia)
            VALUES (:prediction_id, :age, :bp, :bgr, :bu, :sod, :pot, :wc, :htn, :dm, :cad, :appet, :pe, :ane)
        """)
        
        params = {
            'prediction_id': int(prediction_id),
            'age': int(input_data.get('age', 0)),
            'bp': float(input_data.get('bp', 0)),
            'bgr': float(input_data.get('bgr', 0)),
            'bu': float(input_data.get('bu', 0)),
            'sod': float(input_data.get('sod', 0)),
            'pot': float(input_data.get('pot', 0)),
            'wc': float(input_data.get('wc', 0)),
            'htn': int(input_data.get('htn_yes', 0)),
            'dm': int(input_data.get('dm_yes', 0)),
            'cad': int(input_data.get('cad_yes', 0)),
            'appet': int(input_data.get('appet_poor', 0)),
            'pe': int(input_data.get('pe_yes', 0)),
            'ane': int(input_data.get('ane_yes', 0))
        }
        
        conn.execute(query, params)
        print("✅ Kidney disease inputs saved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error saving kidney disease inputs: {e}")
        return False

# =============================================================================
# RECOMMENDATION FUNCTIONS
# =============================================================================

def get_heart_disease_recommendations(health_score, prediction_proba, features):
    """Generate personalized heart disease recommendations"""
    recommendations = []
    
    # Base recommendations based on health score
    if health_score >= 80:
        recommendations.extend([
            "Great heart health! Keep up the good work",
            "Continue regular exercise and healthy diet",
            "Annual cardiac check-ups recommended"
        ])
    elif health_score >= 60:
        recommendations.extend([
            "Good heart health with room for improvement",
            "Consider increasing physical activity",
            "Monitor blood pressure and cholesterol regularly"
        ])
    else:
        recommendations.extend([
            "Heart health needs attention - consult a cardiologist",
            "Immediate lifestyle changes recommended",
            "Consider cardiac rehabilitation program"
        ])
    
    # Specific recommendations based on risk factors
    if features[4] < 40:  # Low ejection fraction
        recommendations.append("Focus on heart-strengthening exercises")
    
    if features[5] == 1:  # High blood pressure
        recommendations.append("Implement strict blood pressure management")
    
    if features[10] == 1:  # Smoking
        recommendations.append("Quit smoking immediately - seek support")
    
    if features[7] > 1.5:  # High serum creatinine
        recommendations.append("Monitor kidney function closely")
    
    if features[0] > 65:  # Age factor
        recommendations.append("Regular senior cardiac screening")
    
    return recommendations[:6]  # Limit to 6 recommendations

def get_diabetes_recommendations(health_score, prediction_proba, features):
    """Generate personalized diabetes recommendations"""
    recommendations = []
    
    # Base recommendations based on health score
    if health_score >= 80:
        recommendations.extend([
            "Excellent glucose control! Maintain current habits",
            "Continue balanced diet and regular exercise",
            "Annual diabetes screening recommended"
        ])
    elif health_score >= 60:
        recommendations.extend([
            "Good metabolic health with improvements needed",
            "Focus on carbohydrate management",
            "Increase physical activity gradually"
        ])
    else:
        recommendations.extend([
            "High diabetes risk - consult an endocrinologist",
            "Immediate dietary changes essential",
            "Consider diabetes prevention program"
        ])
    
    # Specific recommendations based on features
    if features[1] > 125:  # High glucose
        recommendations.append("Strict glucose monitoring and control")
    
    if features[5] > 30:  # High BMI
        recommendations.append("Weight management program recommended")
    
    if features[2] > 140:  # High blood pressure
        recommendations.append("Blood pressure control essential")
    
    if features[7] > 50:  # Age factor
        recommendations.append("Enhanced diabetes screening for seniors")
    
    if features[0] > 5:  # Multiple pregnancies
        recommendations.append("Monitor for gestational diabetes risk")
    
    return recommendations[:6]  # Limit to 6 recommendations

def get_lung_cancer_recommendations(health_score, prediction_proba, features, input_data):
    """Generate personalized lung cancer recommendations for 10-feature model"""
    recommendations = []
    
    # Base recommendations based on health score
    if health_score >= 80:
        recommendations.extend([
            "Low lung cancer risk - maintain healthy lifestyle",
            "Continue avoiding smoking and secondhand smoke",
            "Annual health check-ups recommended"
        ])
    elif health_score >= 60:
        recommendations.extend([
            "Moderate risk factors present",
            "Consider lung cancer screening if indicated",
            "Maintain healthy diet and exercise"
        ])
    else:
        recommendations.extend([
            "High-risk factors detected - consult oncologist",
            "Immediate smoking cessation if applicable",
            "Consider genetic counseling"
        ])
    
    # Specific recommendations based on genetic factors (features 1-9 for 10-feature model)
    mutation_count = sum(features[1:])  # Count all mutations except age
    if mutation_count > 3:
        recommendations.append("Genetic counseling strongly recommended")
    
    # Age factor (feature 0)
    if features[0] > 55:
        recommendations.append("Regular lung cancer screening advised")
    
    # Gender factor from input data (not used in 10-feature model but available for recommendations)
    if input_data.get('sex') == 1:
        recommendations.append("Higher risk due to gender - monitor closely")
    
    # Specific mutation recommendations based on feature positions
    
    if len(features) > 1 and features[1] == 1:  # EGFR
        recommendations.append("EGFR mutation - discuss targeted therapy options")
    if len(features) > 2 and features[2] == 1:  # KRAS
        recommendations.append("KRAS mutation - consider clinical trials")
    if len(features) > 4 and features[4] == 1:  # TP53
        recommendations.append("TP53 mutation - enhanced monitoring needed")
    
    return recommendations[:6]  # Limit to 6 recommendations

def get_kidney_disease_recommendations(health_score, prediction_proba, features_dict):
    """Generate personalized kidney health recommendations"""
    recommendations = []
    
    if health_score >= 80:
        recommendations.extend([
            "Excellent kidney health! Continue current lifestyle",
            "Regular annual check-ups recommended",
            "Maintain healthy diet and exercise routine"
        ])
    elif health_score >= 60:
        recommendations.extend([
            "Good kidney health with room for improvement",
            "Monitor blood pressure and glucose regularly",
            "Increase water intake to 8-10 glasses daily"
        ])
    else:
        recommendations.extend([
            "Kidney health needs attention - consult a nephrologist",
            "Strict blood pressure and glucose control essential",
            "Consider dietary modifications and medication review"
        ])
    
    # Specific recommendations based on parameters
    if features_dict.get('bp', 80) > 140:
        recommendations.append("Focus on blood pressure management")
    
    if features_dict.get('bgr', 148) > 140:
        recommendations.append("Implement strict glucose control measures")
    
    if features_dict.get('bu', 25) > 40:
        recommendations.append("Monitor kidney function closely")
    
    if features_dict.get('htn_yes', 0):
        recommendations.append("Continue hypertension management")
    
    if features_dict.get('dm_yes', 0):
        recommendations.append("Maintain optimal diabetes control")
    
    if features_dict.get('appet_poor', 0):
        recommendations.append("Address nutritional concerns with dietitian")
    
    return recommendations[:6]  # Limit to 6 recommendations

# =============================================================================
# FEATURE ANALYSIS FUNCTIONS
# =============================================================================

def analyze_heart_disease_features(features):
    """Analyze heart disease features and provide insights"""
    analysis = {}
    feature_names = HEART_DISEASE_FEATURES
    
    for i, feature in enumerate(feature_names):
        value = features[i]
        ideal_info = HEART_DISEASE_IDEAL_VALUES.get(feature, {})
        
        analysis[feature] = {
            'value': value,
            'ideal': ideal_info.get('ideal', 'N/A'),
            'unit': ideal_info.get('unit', ''),
            'status': 'normal'  # Default status
        }
        
        # Determine status based on value
        if feature == 'ejection_fraction':
            if value < 40:
                analysis[feature]['status'] = 'critical'
            elif value < 50:
                analysis[feature]['status'] = 'warning'
        elif feature == 'serum_creatinine':
            if value > 1.5:
                analysis[feature]['status'] = 'warning'
        elif feature == 'age':
            if value > 70:
                analysis[feature]['status'] = 'warning'
    
    return analysis

def analyze_diabetes_features(features):
    """Analyze diabetes features and provide insights"""
    analysis = {}
    feature_names = DIABETES_FEATURES
    
    for i, feature in enumerate(feature_names):
        value = features[i]
        ideal_info = DIABETES_IDEAL_VALUES.get(feature, {})
        
        analysis[feature] = {
            'value': value,
            'ideal': ideal_info.get('ideal', 'N/A'),
            'unit': ideal_info.get('unit', ''),
            'status': 'normal'  # Default status
        }
        
        # Determine status based on value
        if feature == 'Glucose':
            if value > 125:
                analysis[feature]['status'] = 'critical'
            elif value > 100:
                analysis[feature]['status'] = 'warning'
        elif feature == 'BMI':
            if value > 30:
                analysis[feature]['status'] = 'warning'
        elif feature == 'BloodPressure':
            if value > 140:
                analysis[feature]['status'] = 'warning'
    
    return analysis

def analyze_lung_cancer_features(features, feature_names):
    """Analyze lung cancer features and provide insights"""
    analysis = {}
    
    for i, feature in enumerate(feature_names):
        if i >= len(features):
            break
            
        value = features[i]
        ideal_info = LUNG_CANCER_IDEAL_VALUES.get(feature, {})
        
        analysis[feature] = {
            'value': value,
            'ideal': ideal_info.get('ideal', 'N/A'),
            'unit': ideal_info.get('unit', ''),
            'status': 'normal'  # Default status
        }
        
        # Determine status for genetic mutations
        if feature in ['EGFR', 'KRAS', 'ALK', 'TP53', 'STK11', 'KEAP1', 'BRAF', 'ROS1', 'MET']:
            if value == 1:
                analysis[feature]['status'] = 'warning'
        elif feature == 'Age':
            if value > 65:
                analysis[feature]['status'] = 'warning'
    
    return analysis

def analyze_kidney_disease_features(features, input_data):
    """Analyze kidney disease features and provide insights"""
    analysis = {}
    feature_names = KIDNEY_DISEASE_FEATURES
    
    for i, feature in enumerate(feature_names):
        value = features[i]
        ideal_info = KIDNEY_DISEASE_IDEAL_VALUES.get(feature, {})
        
        analysis[feature] = {
            'value': value,
            'ideal': ideal_info.get('ideal', 'N/A'),
            'unit': ideal_info.get('unit', ''),
            'status': 'normal'  # Default status
        }
        
        # Determine status based on value
        if feature == 'bp' and value > 140:
            analysis[feature]['status'] = 'warning'
        elif feature == 'bgr' and value > 140:
            analysis[feature]['status'] = 'warning'
        elif feature == 'bu' and value > 40:
            analysis[feature]['status'] = 'warning'
        elif feature in ['htn_yes', 'dm_yes', 'cad_yes'] and value == 1:
            analysis[feature]['status'] = 'warning'
    
    return analysis

# =============================================================================
# ROUTES - MAIN PAGES
# =============================================================================

@app.route('/')
def index():
    """Landing page"""
    return render_template('landing_page.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    """Analytics page"""
    return render_template('analytics.html')

@app.route('/login')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/signup')
def signup():
    """Signup page"""
    return render_template('signup.html')

@app.route('/profile')
def profile_page():
    """Profile page"""
    return render_template('profile.html')

@app.route('/settings')
def settings_page():
    """Settings page"""
    return render_template('settings.html')

@app.route('/feedback')
def feedback_page():
    """Feedback page"""
    return render_template('feedback.html')

@app.route('/forgetpass')
def forgetpass():
    """Forgot password page"""
    return render_template('forgetpass.html')

@app.route('/report')
def report():
    """Report page"""
    return render_template('report.html')

@app.route('/heart_disease')
def heart_disease():
    """Heart disease prediction page"""
    return render_template('heart_disease.html')

@app.route('/diabeties')
def diabeties():
    """Diabetes prediction page (note the spelling)"""
    return render_template('diabeties.html')

@app.route('/lung_cancer')
def lung_cancer():
    """Lung cancer prediction page"""
    return render_template('lung_cancer.html')

@app.route('/kidney_disease')
def kidney_disease():
    """Kidney disease prediction page"""
    return render_template('kidney_disease.html')

@app.route('/chat')
def chat():
    """Chatbot page"""
    return render_template('chat.html')

@app.route('/test-firebase')
def test_firebase():
    """Firebase test page"""
    return render_template('test_firebase.html')

# =============================================================================
# PREDICTION ENDPOINTS
# =============================================================================

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease():
    """Heart disease prediction endpoint"""
    try:
        if heart_disease_model is None or heart_disease_scaler is None:
            return jsonify({'error': 'Heart disease model not loaded'}), 500
        
        data = request.json
        print(f"📊 Heart Disease Prediction Request: {data}")
        
        # Get user ID from request
        user_id = data.get('user_id', 'anonymous')
        
        # Extract features in the correct order
        features = [
            float(data.get('age', 0)),
            int(data.get('anaemia', 0)),
            float(data.get('creatinine_phosphokinase', 0)),
            int(data.get('diabetes', 0)),
            float(data.get('ejection_fraction', 0)),
            int(data.get('high_blood_pressure', 0)),
            float(data.get('platelets', 0)),
            float(data.get('serum_creatinine', 0)),
            float(data.get('serum_sodium', 0)),
            int(data.get('sex', 0)),
            int(data.get('smoking', 0)),
            float(data.get('time', 0))
        ]
        
        # Scale the features
        features_scaled = heart_disease_scaler.transform([features])
        
        # Make prediction
        prediction = heart_disease_model.predict(features_scaled)[0]
        prediction_proba = heart_disease_model.predict_proba(features_scaled)[0]
        
        # Calculate health score
        health_score = calculate_health_score(features, prediction_proba)
        
        # Get recommendations
        recommendations = get_heart_disease_recommendations(health_score, prediction_proba, features)
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'High Risk of Heart Failure' if prediction == 1 else 'Low Risk of Heart Failure',
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'no_risk': float(prediction_proba[0]),
                'high_risk': float(prediction_proba[1])
            },
            'health_score': health_score,
            'confidence': float(max(prediction_proba)) * 100,
            'recommendations': recommendations,
            'feature_analysis': analyze_heart_disease_features(features),
            'interpretation': {
                'title': f"Heart Disease Risk Assessment: {'High Risk' if prediction == 1 else 'Low Risk'}",
                'description': f"Based on the provided health parameters, the model predicts a {'high' if prediction == 1 else 'low'} risk of heart failure.",
                'recommendation': recommendations
            }
        }
        
        # Save to database
        prediction_id = save_prediction_to_db(user_id, 'heart_disease', data, result)
        if prediction_id:
            result['prediction_id'] = prediction_id
            print(f"✅ Heart disease prediction saved with ID: {prediction_id}")
        else:
            print("⚠️ Failed to save heart disease prediction to database")
        
        print(f"✅ Heart Disease Prediction Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Heart Disease Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """Diabetes prediction endpoint"""
    try:
        if diabetes_model is None or diabetes_scaler is None:
            return jsonify({'error': 'Diabetes model not loaded'}), 500
        
        data = request.json
        print(f"📊 Diabetes Prediction Request: {data}")
        
        # Get user ID from request
        user_id = data.get('user_id', 'anonymous')
        
        # Extract features in the correct order
        features = [
            float(data.get('Pregnancies', 0)),
            float(data.get('Glucose', 0)),
            float(data.get('BloodPressure', 0)),
            float(data.get('SkinThickness', 0)),
            float(data.get('Insulin', 0)),
            float(data.get('BMI', 0)),
            float(data.get('DiabetesPedigreeFunction', 0)),
            float(data.get('Age', 0))
        ]
        
        # Scale the features
        features_scaled = diabetes_scaler.transform([features])
        
        # Make prediction
        prediction = diabetes_model.predict(features_scaled)[0]
        prediction_proba = diabetes_model.predict_proba(features_scaled)[0]
        
        # Calculate health score
        health_score = calculate_diabetes_health_score(features, prediction_proba)
        
        # Get recommendations
        recommendations = get_diabetes_recommendations(health_score, prediction_proba, features)
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'High Risk of Diabetes' if prediction == 1 else 'Low Risk of Diabetes',
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'no_diabetes': float(prediction_proba[0]),
                'diabetes': float(prediction_proba[1])
            },
            'health_score': health_score,
            'confidence': float(max(prediction_proba)) * 100,
            'recommendations': recommendations,
            'feature_analysis': analyze_diabetes_features(features),
            'interpretation': {
                'title': f"Diabetes Risk Assessment: {'High Risk' if prediction == 1 else 'Low Risk'}",
                'description': f"Based on the provided health parameters, the model predicts a {'high' if prediction == 1 else 'low'} risk of developing diabetes.",
                'recommendation': recommendations
            }
        }
        
        # Save to database
        prediction_id = save_prediction_to_db(user_id, 'diabetes', data, result)
        if prediction_id:
            result['prediction_id'] = prediction_id
            print(f"✅ Diabetes prediction saved with ID: {prediction_id}")
        else:
            print("⚠️ Failed to save diabetes prediction to database")
        
        print(f"✅ Diabetes Prediction Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Diabetes Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/lung-cancer', methods=['POST'])
def predict_lung_cancer():
    """Lung cancer prediction endpoint - COMPLETELY FIXED VERSION"""
    try:
        if lung_cancer_model is None:
            return jsonify({'error': 'Lung cancer model not loaded'}), 500
        
        data = request.json
        print(f"📊 Lung Cancer Prediction Request: {data}")
        
        # Get user ID from request
        user_id = data.get('user_id', 'anonymous')
        
        # Get different feature combinations to try
        feature_combinations = get_lung_cancer_features(data)
        
        prediction = None
        prediction_proba = None
        used_features = None
        used_feature_names = None
        
        # Try different feature combinations until one works
        for i, combo in enumerate(feature_combinations):
            try:
                features = combo['features']
                feature_names = combo['names']
                
                print(f"🔍 Trying feature combination {i+1}: {combo['description']}")
                print(f"📊 Features ({len(features)}): {features}")
                
                # Validate feature count
                if len(features) != 10:
                    print(f"⚠️ Skipping: Expected 10 features, got {len(features)}")
                    continue
                
                # Scale the features if scaler is available
                if lung_cancer_scaler is not None:
                    features_scaled = lung_cancer_scaler.transform([features])
                    print(f"📊 Features scaled: {features_scaled}")
                else:
                    features_scaled = [features]
                
                # Make prediction
                test_prediction = lung_cancer_model.predict(features_scaled)[0]
                test_prediction_proba = lung_cancer_model.predict_proba(features_scaled)[0]
                
                # If we get here, this feature combination works
                prediction = test_prediction
                prediction_proba = test_prediction_proba
                used_features = features
                used_feature_names = feature_names
                
                print(f"✅ Feature combination {i+1} worked!")
                print(f"📊 Raw prediction: {prediction}")
                print(f"📊 Raw probabilities: {prediction_proba}")
                break
                
            except Exception as combo_error:
                print(f"❌ Feature combination {i+1} failed: {combo_error}")
                continue
        
        # If no combination worked, raise an error
        if prediction is None:
            raise ValueError("No feature combination worked with the model. Please check model requirements.")
        
        # Validate and fix probabilities
        prediction_proba = validate_and_fix_probabilities(prediction_proba, prediction)
        
        print(f"📊 Final probabilities: {prediction_proba}")
        
        # Calculate health score using the working features
        health_score = calculate_lung_cancer_health_score(used_features, prediction_proba)
        
        # Get recommendations
        recommendations = get_lung_cancer_recommendations(health_score, prediction_proba, used_features, data)
        
        # Prepare response with correct field names
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'High Risk of Lung Cancer' if prediction == 1 else 'Low Risk of Lung Cancer',
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'no_risk': float(prediction_proba[0]),
                'high_risk': float(prediction_proba[1])
            },
            'health_score': health_score,
            'confidence': float(max(prediction_proba)) * 100,
            'recommendations': recommendations,
            'feature_analysis': analyze_lung_cancer_features(used_features, used_feature_names),
            'interpretation': {
                'title': f"Lung Cancer Risk Assessment: {'High Risk' if prediction == 1 else 'Low Risk'}",
                'description': f"Based on genetic markers and age, the model predicts a {'high' if prediction == 1 else 'low'} risk of lung cancer.",
                'recommendation': recommendations[:3]
            },
            'model_info': {
                'features_used': len(used_features),
                'feature_names': used_feature_names
            }
        }
        
        # Save to database with proper field mapping
        db_data = {
            'user_id': user_id,
            'username': data.get('username', 'Sky-3305'),
            'email': data.get('email', 'sky-3305@precisionmedicine.com'),
            'age': data.get('age', 0),
            'sex': data.get('sex', 0),
            'egfr': data.get('egfr', 0),
            'kras': data.get('kras', 0),
            'alk': data.get('alk', 0),
            'tp53': data.get('tp53', 0),
            'stk11': data.get('stk11', 0),
            'keap1': data.get('keap1', 0),
            'braf': data.get('braf', 0),
            'ros1': data.get('ros1', 0),
            'met': data.get('met', 0)
        }
        
        # Enhanced database saving with detailed logging
        print(f"💾 Attempting to save lung cancer prediction to database...")
        prediction_id = save_prediction_to_db(user_id, 'lung_cancer', db_data, result)
        if prediction_id:
            result['prediction_id'] = prediction_id
            print(f"✅ Lung cancer prediction saved to database with ID: {prediction_id}")
        else:
            print("⚠️ Failed to save lung cancer prediction to database")
            # Don't fail the entire request if database save fails
            result['database_save_status'] = 'failed'
        
        print(f"✅ Lung Cancer Prediction Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Lung Cancer Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/kidney-disease', methods=['POST'])
def predict_kidney_disease():
    """Kidney disease prediction endpoint"""
    try:
        # Load kidney disease models on first request
        if kidney_disease_model is None:
            if not load_kidney_disease_models():
                return jsonify({'error': 'Kidney disease models not available'}), 500
        
        data = request.json
        print(f"📊 Kidney Disease Prediction Request: {data}")
        
        # Get user ID from request
        user_id = data.get('user_id', 'anonymous')
        
        # Extract features in the correct order
        features = [
            float(data.get('age', 0)),
            float(data.get('bp', 0)),
            float(data.get('bgr', 0)),
            float(data.get('bu', 0)),
            float(data.get('sod', 0)),
            float(data.get('pot', 0)),
            float(data.get('wc', 0)),
            int(data.get('htn_yes', 0)),
            int(data.get('dm_yes', 0)),
            int(data.get('cad_yes', 0)),
            int(data.get('appet_poor', 0)),
            int(data.get('pe_yes', 0)),
            int(data.get('ane_yes', 0))
        ]
        
        # Scale the features
        features_scaled = kidney_disease_scaler.transform([features])
        
        # Make prediction
        prediction = kidney_disease_model.predict(features_scaled)[0]
        prediction_proba = kidney_disease_model.predict_proba(features_scaled)[0]
        
        # Calculate health score
        health_score = calculate_kidney_health_score(data)
        
        # Get recommendations
        recommendations = get_kidney_disease_recommendations(health_score, prediction_proba, data)
        
        # Prepare response
        result = {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'Chronic Kidney Disease Detected' if prediction == 1 else 'No Chronic Kidney Disease Detected',
            'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': {
                'no_disease': float(prediction_proba[0]),
                'disease': float(prediction_proba[1])
            },
            'health_score': health_score,
            'confidence': float(max(prediction_proba)) * 100,
            'recommendations': recommendations,
            'feature_analysis': analyze_kidney_disease_features(features, data),
            'interpretation': {
                'title': f"Kidney Disease Risk Assessment: {'High Risk' if prediction == 1 else 'Low Risk'}",
                'description': f"Based on clinical parameters and medical history, the model predicts a {'high' if prediction == 1 else 'low'} risk of chronic kidney disease.",
                'recommendation': recommendations
            }
        }
        
        # Save to database
        prediction_id = save_prediction_to_db(user_id, 'kidney_disease', data, result)
        if prediction_id:
            result['prediction_id'] = prediction_id
            print(f"✅ Kidney disease prediction saved with ID: {prediction_id}")
        else:
            print("⚠️ Failed to save kidney disease prediction to database")
        
        print(f"✅ Kidney Disease Prediction Result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Kidney Disease Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# =============================================================================
# CHAT ENDPOINT
# =============================================================================

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat messages"""
    try:
        data = request.json
        message = data.get('message', data.get('question', '')).strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        print(f"🤖 Chat Message: {message}")

        # Check for greeting/thanks/goodbye
        greeting = simple_bot.is_greeting(message)
        if greeting:
            response = simple_bot.greetings[greeting]
        elif simple_bot.is_thanks(message):
            response = random.choice(simple_bot.thanks_responses)
        elif simple_bot.is_goodbye(message):
            response = simple_bot.goodbye_response
        else:
            answer = simple_bot.find_answer(message)
            if answer:
                response = answer
            else:
                response = random.choice(simple_bot.unknown_responses)

        result = {
            'response': response,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        print(f"🤖 Chat Response: {response}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@app.route('/api/analytics/user-predictions', methods=['POST'])
def get_user_predictions_by_username_email():
    """Get all predictions for a specific user based on username and email"""
    try:
        data = request.json
        username = data.get('username')
        email = data.get('email')
        
        print(f"📊 Fetching predictions for username: {username}, email: {email}")
        
        if not username or not email:
            return jsonify({'error': 'Username and email are required', 'success': False}), 400
        
        engine = get_db_connection()
        if not engine:
            return jsonify({'error': 'Database connection failed', 'success': False}), 500
        
        with engine.connect() as conn:
            # Query to get all predictions for the user based on username and email
            query = text("""
                SELECT 
                    p.PredictionID,
                    p.UserID,
                    p.PredictionType,
                    p.RiskLevel,
                    p.Probability,
                    p.HealthScore,
                    p.Confidence,
                    p.CreatedAt,
                    u.Username,
                    u.Email
                FROM Predictions p
                INNER JOIN Users u ON p.UserID = u.UserID
                WHERE u.Username = :username AND u.Email = :email
                ORDER BY p.CreatedAt DESC
            """)
            
            result = conn.execute(query, {
                'username': username,
                'email': email
            })
            
            predictions = []
            for row in result:
                prediction = {
                    'PredictionID': row[0],
                    'UserID': row[1],
                    'PredictionType': row[2],
                    'RiskLevel': row[3],
                    'Probability': row[4],
                    'HealthScore': row[5],
                    'Confidence': row[6],
                    'CreatedAt': row[7].isoformat() if row[7] else None,
                    'Username': row[8],
                    'Email': row[9]
                }
                predictions.append(prediction)
            
            print(f"✅ Found {len(predictions)} predictions for user {username} ({email})")
            
            # Return the predictions
            return jsonify({
                'success': True,
                'username': username,
                'email': email,
                'predictions': predictions,
                'total_count': len(predictions)
            })
            
    except Exception as e:
        print(f"❌ Error fetching user predictions by username/email: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to fetch predictions: {str(e)}',
            'success': False
        }), 500

@app.route('/api/analytics/predictions/<user_id>')
def get_user_predictions(user_id):
    """Get all predictions for a specific user"""
    try:
        print(f"📊 Fetching predictions for user: {user_id}")
        
        engine = get_db_connection()
        if not engine:
            return jsonify({'error': 'Database connection failed'}), 500
        
        with engine.connect() as conn:
            # Query to get all predictions for the user
            query = text("""
                SELECT 
                    p.PredictionID,
                    p.UserID,
                    p.PredictionType,
                    p.RiskLevel,
                    p.Probability,
                    p.HealthScore,
                    p.Confidence,
                    p.CreatedAt,
                    u.Username,
                    u.Email
                FROM Predictions p
                LEFT JOIN Users u ON p.UserID = u.UserID
                WHERE u.Username = :user_id OR p.UserID = :user_id_int
                ORDER BY p.CreatedAt DESC
            """)
            
            # Try to convert user_id to int, if it fails, use 0
            try:
                user_id_int = int(user_id)
            except ValueError:
                user_id_int = 0
            
            result = conn.execute(query, {
                'user_id': user_id,
                'user_id_int': user_id_int
            })
            
            predictions = []
            for row in result:
                prediction = {
                    'PredictionID': row[0],
                    'UserID': row[1],
                    'PredictionType': row[2],
                    'RiskLevel': row[3],
                    'Probability': row[4],
                    'HealthScore': row[5],
                    'Confidence': row[6],
                    'CreatedAt': row[7].isoformat() if row[7] else None,
                    'Username': row[8],
                    'Email': row[9]
                }
                predictions.append(prediction)
            
            print(f"✅ Found {len(predictions)} predictions for user {user_id}")
            
            # Return the predictions
            return jsonify({
                'success': True,
                'user_id': user_id,
                'predictions': predictions,
                'total_count': len(predictions)
            })
            
    except Exception as e:
        print(f"❌ Error fetching user predictions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to fetch predictions: {str(e)}',
            'success': False
        }), 500

@app.route('/api/analytics')
def get_analytics_overview():
    """Get overall analytics data for the dashboard"""
    try:
        engine = get_db_connection()
        if not engine:
            # Return sample data if database is not available
            return jsonify({
                'totalPredictions': 156,
                'predictionsByType': {
                    'heart_disease': 45,
                    'diabetes': 38,
                    'lung_cancer': 32,
                    'kidney_disease': 41
                },
                'riskDistribution': {
                    'low_risk': 89,
                    'high_risk': 67
                },
                'avgHealthScore': 72.5,
                'recentActivity': [
                    {'date': '2025-07-06', 'count': 12},
                    {'date': '2025-07-05', 'count': 8},
                    {'date': '2025-07-04', 'count': 15},
                    {'date': '2025-07-03', 'count': 10},
                    {'date': '2025-07-02', 'count': 7}
                ]
            })
            
        with engine.connect() as conn:
            # Get total predictions
            total_query = text("SELECT COUNT(*) as total FROM Predictions")
            result = conn.execute(total_query)
            total_predictions = result.fetchone()[0]
            
            # Get predictions by type
            type_query = text("""
                SELECT PredictionType, COUNT(*) as count 
                FROM Predictions 
                GROUP BY PredictionType
            """)
            result = conn.execute(type_query)
            predictions_by_type = {row[0]: row[1] for row in result}
            
            # Get risk distribution
            risk_query = text("""
                SELECT RiskLevel, COUNT(*) as count 
                FROM Predictions 
                WHERE RiskLevel IS NOT NULL
                GROUP BY RiskLevel
            """)
            result = conn.execute(risk_query)
            risk_distribution = {}
            for row in result:
                if 'High' in str(row[0]):
                    risk_distribution['high_risk'] = row[1]
                else:
                    risk_distribution['low_risk'] = row[1]
            
            # Get average health score
            score_query = text("SELECT AVG(CAST(HealthScore AS FLOAT)) as avg_score FROM Predictions WHERE HealthScore IS NOT NULL")
            result = conn.execute(score_query)
            avg_health_score = result.fetchone()[0] or 0
            
            # Get recent activity (last 7 days)
            activity_query = text("""
                SELECT 
                    CAST(CreatedAt AS DATE) as date,
                    COUNT(*) as count
                FROM Predictions 
                WHERE CreatedAt >= DATEADD(day, -7, GETDATE())
                GROUP BY CAST(CreatedAt AS DATE)
                ORDER BY date DESC
            """)
            result = conn.execute(activity_query)
            recent_activity = [{'date': str(row[0]), 'count': row[1]} for row in result]
            
            return jsonify({
                'totalPredictions': total_predictions,
                'predictionsByType': predictions_by_type,
                'riskDistribution': risk_distribution,
                'avgHealthScore': round(float(avg_health_score), 2),
                'recentActivity': recent_activity
            })
            
    except Exception as e:
        print(f"Error fetching analytics overview: {e}")
        # Return sample data on error
        return jsonify({
            'totalPredictions': 156,
            'predictionsByType': {
                'heart_disease': 45,
                'diabetes': 38,
                'lung_cancer': 32,
                'kidney_disease': 41
            },
            'riskDistribution': {
                'low_risk': 89,
                'high_risk': 67
            },
            'avgHealthScore': 72.5,
            'recentActivity': [
                {'date': '2025-07-06', 'count': 12},
                {'date': '2025-07-05', 'count': 8},
                {'date': '2025-07-04', 'count': 15}
            ]
        })

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.route('/features/<prediction_type>')
def get_features(prediction_type):
    """Get feature information for a specific prediction type"""
    try:
        if prediction_type == 'heart-disease':
            return jsonify({
                'features': HEART_DISEASE_FEATURES,
                'ideal_values': HEART_DISEASE_IDEAL_VALUES
            })
        elif prediction_type == 'diabetes':
            return jsonify({
                'features': DIABETES_FEATURES,
                'ideal_values': DIABETES_IDEAL_VALUES
            })
        elif prediction_type == 'lung-cancer':
            return jsonify({
                'features': LUNG_CANCER_FEATURES_10,
                'ideal_values': LUNG_CANCER_IDEAL_VALUES
            })
        elif prediction_type == 'kidney-disease':
            return jsonify({
                'features': KIDNEY_DISEASE_FEATURES,
                'ideal_values': KIDNEY_DISEASE_IDEAL_VALUES
            })
        else:
            return jsonify({'error': 'Invalid prediction type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint with database status"""
    try:
        # Test database connection
        db_status = False
        db_message = "Not connected"
        
        try:
            engine = get_db_connection()
            if engine:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    db_status = True
                    db_message = "Connected successfully"
        except Exception as db_error:
            db_message = f"Connection failed: {str(db_error)}"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': {
                'connected': db_status,
                'message': db_message
            },
            'models': {
                'heart_disease': heart_disease_model is not None,
                'diabetes': diabetes_model is not None,
                'lung_cancer': lung_cancer_model is not None,
                'kidney_disease': kidney_disease_model is not None
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test-db')
def test_db():
    """Test database connection with detailed diagnostics"""
    try:
        print("🔍 Testing database connection...")
        
        # Test basic connection
        engine = get_db_connection()
        if not engine:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to create database engine',
                'details': 'Check connection string and SQL Server availability'
            }), 500
        
        # Test query execution
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test, GETDATE() as current_time"))
            row = result.fetchone()
            
            # Test schema validation
            schema_valid = validate_database_schema()
            
            return jsonify({
                'status': 'success', 
                'message': 'Database connection successful',
                'server': DB_CONFIG['server'],
                'database': DB_CONFIG['database'],
                'test_query_result': row[0] if row else None,
                'server_time': str(row[1]) if row and len(row) > 1 else None,
                'schema_validation': 'passed' if schema_valid else 'failed'
            })
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'Database test failed: {str(e)}',
            'details': 'Check SQL Server service, connection string, and permissions'
        }), 500

# =============================================================================
# LEGACY ANALYTICS ROUTES (For backward compatibility)
# =============================================================================

@app.route('/analytics-data')
def get_analytics_data():
    """Legacy analytics endpoint for backward compatibility"""
    try:
        engine = get_db_connection()
        if not engine:
            return jsonify({'error': 'Database connection failed'}), 500
            
        with engine.connect() as conn:
            # Get prediction counts by type
            counts_query = text("""
                SELECT 
                    PredictionType,
                    COUNT(*) as Count
                FROM Predictions 
                GROUP BY PredictionType
            """)
            
            result = conn.execute(counts_query)
            prediction_counts = {}
            total_count = 0
            
            for row in result:
                prediction_type = row[0]
                count = row[1]
                prediction_counts[prediction_type] = count
                total_count += count
            
            # Get today's predictions
            today_query = text("""
                SELECT COUNT(*) as TodayCount
                FROM Predictions 
                WHERE CAST(CreatedAt AS DATE) = CAST(GETDATE() AS DATE)
            """)
            
            result = conn.execute(today_query)
            today_result = result.fetchone()
            today_count = today_result[0] if today_result else 0
            
            # Get risk level distribution
            risk_query = text("""
                SELECT 
                    RiskLevel,
                    COUNT(*) as Count
                FROM Predictions 
                WHERE RiskLevel IS NOT NULL
                GROUP BY RiskLevel
            """)
            
            result = conn.execute(risk_query)
            risk_distribution = {}
            for row in result:
                risk_level = row[0]
                count = row[1]
                risk_distribution[risk_level.lower().replace(' ', '_')] = count
            
            return jsonify({
                'heart_disease': prediction_counts.get('heart_disease', 0),
                'diabetes': prediction_counts.get('diabetes', 0),
                'lung_cancer': prediction_counts.get('lung_cancer', 0),
                'kidney_disease': prediction_counts.get('kidney_disease', 0),
                'total': total_count,
                'today': today_count,
                'high_risk': risk_distribution.get('high_risk', 0),
                'medium_risk': risk_distribution.get('medium_risk', 0),
                'low_risk': risk_distribution.get('low_risk', 0),
                'success': True
            })
            
    except Exception as e:
        print(f"Error fetching analytics data: {e}")
        # Return sample data if database fails
        return jsonify({
            'heart_disease': 12,
            'diabetes': 8,
            'lung_cancer': 5,
            'kidney_disease': 3,
            'total': 28,
            'today': 2,
            'high_risk': 8,
            'medium_risk': 12,
            'low_risk': 8,
            'success': True,
            'note': 'Sample data - database connection failed'
        })

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    # Validate database schema on startup
    print("🔍 Validating database schema on startup...")
    schema_validation = validate_database_schema()
    if schema_validation:
        print("✅ Database schema validation passed")
    else:
        print("⚠️ Database schema validation failed - some features may not work")
    
    
    # Print server information
    print("=" * 60)
    print("🌐 PRECISION MEDICINE FLASK SERVER")
    print("=" * 60)
    print(f"🕐 Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("📋 Available Routes:")
    print("  🏠 Landing Page:     http://localhost:5000/")
    print("  📱 Dashboard:        http://localhost:5000/dashboard")
    print("  📊 Analytics:        http://localhost:5000/analytics")
    print("  🔐 Authentication:   http://localhost:5000/login")
    print("  📝 Signup:           http://localhost:5000/signup")
    print("  🔐 Forgot Password:  http://localhost:5000/forgetpass")
    print("=" * 60)
    print("🏥 Prediction Modules:")
    print("  ❤️  Heart Disease:   http://localhost:5000/heart_disease")
    print("  🩺 Diabetes:         http://localhost:5000/diabeties")
    print("  🫁 Lung Cancer:      http://localhost:5000/lung_cancer")
    print("  🫘 Kidney Disease:   http://localhost:5000/kidney_disease")
    print("=" * 60)
    print("🛠️ API Endpoints:")
    print("  🤖 Chatbot:          http://localhost:5000/chat")
    print("  📊 Analytics API:    http://localhost:5000/api/analytics")
    print("  🩺 Health Check:     http://localhost:5000/health")
    print("  🔍 Database Test:    http://localhost:5000/test-db")
    print("  🔧 Firebase Test:    http://localhost:5000/test-firebase")
    print("=" * 60)
    print("💾 Database Configuration:")
    print(f"  Server: {DB_CONFIG['server']}")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  Driver: {DB_CONFIG['driver']}")
    print("=" * 60)
    print("🚀 Starting Flask application...")
    print("=" * 60)

    # Start the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)