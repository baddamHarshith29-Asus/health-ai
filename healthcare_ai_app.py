# Complete Healthcare AI System for Diabetes Prediction and Care
# Implements all 4 expected solutions with specified technologies

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import pickle
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .solution-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class HealthcareAISystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.patient_profiles = {}
        self.medical_knowledge_base = self.load_medical_knowledge()
        
    def load_medical_knowledge(self):
        """Medical knowledge base for patient support"""
        return {
            "diabetes_info": {
                "description": "Diabetes is a group of metabolic disorders characterized by high blood sugar levels.",
                "risk_factors": ["Family history", "Obesity", "Age > 45", "Sedentary lifestyle", "High blood pressure"],
                "symptoms": ["Frequent urination", "Excessive thirst", "Unexplained weight loss", "Fatigue", "Blurred vision"],
                "prevention": ["Maintain healthy weight", "Regular exercise", "Balanced diet", "Regular checkups"],
                "management": ["Blood sugar monitoring", "Medication adherence", "Diet control", "Regular exercise"]
            },
            "lifestyle_recommendations": {
                "diet": ["Choose whole grains", "Eat plenty of vegetables", "Limit processed foods", "Control portion sizes"],
                "exercise": ["150 minutes moderate activity per week", "Strength training 2x per week", "Daily walking"],
                "monitoring": ["Check blood sugar regularly", "Monitor blood pressure", "Regular eye exams", "Foot care"]
            }
        }

    def load_sample_data(self):
        """Load Pima diabetes dataset from file"""
        try:
            # Update the path to look for pima.csv in the data/ directory
            possible_paths = [
                'data/pima.csv',  # Look in the data directory
                'pima.csv',  # Fallback to the root directory
                '../pima.csv',  # Parent directory
                '../../pima.csv',  # Two levels up
                '../data/pima.csv',  # Data folder in parent directory
                '../../data/pima.csv',  # Data folder two levels up
                '../dataset/pima.csv',  # Dataset folder in parent directory
                '../../dataset/pima.csv',  # Dataset folder two levels up
                './data/pima.csv',  # Data folder in current directory
                './dataset/pima.csv',  # Dataset folder in current directory
            ]
            
            # Try each path until we find the file
            for path in possible_paths:
                try:
                    df = pd.read_csv(path)
                    st.success(f"✅ Dataset loaded successfully from: {path}")
                    
                    # Validate dataset structure
                    expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
                    
                    if all(col in df.columns for col in expected_columns):
                        st.info(f"📊 Dataset contains {len(df)} records with {len(df.columns)} features")
                        print("DataFrame columns:", df.columns)
                        return df
                    else:
                        st.warning(f"Dataset found but missing some expected columns. Expected: {expected_columns}")
                        st.info(f"Available columns: {list(df.columns)}")
                        return df  # Return anyway, might still work
                        
                except FileNotFoundError:
                    continue
                except Exception as e:
                    st.warning(f"Error reading {path}: {str(e)}")
                    continue
            
            # If no file found, show error and provide fallback
            st.error("❌ Could not find pima.csv in any of the expected locations:")
            for path in possible_paths:
                st.write(f"  • {path}")
            
            st.info("🔧 **File Path Setup Instructions:**")
            st.write("1. Copy your `pima.csv` file to the same folder as this program, OR")
            st.write("2. Update the file path in the code below:")
            
            # Let user specify custom path
            custom_path = st.text_input("Enter the full path to your pima.csv file:", 
                                       placeholder="e.g., C:/Users/YourName/Documents/data/pima.csv")
            
            if custom_path:
                try:
                    df = pd.read_csv(custom_path)
                    st.success(f"✅ Dataset loaded from custom path: {custom_path}")
                    return df
                except Exception as e:
                    st.error(f"Error loading from custom path: {str(e)}")
            
            # Fallback to sample data
            st.warning("📝 Using sample data instead. For real results, please provide your pima.csv file.")
            return self.create_sample_data()
            
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample Pima diabetes dataset structure as fallback"""
        np.random.seed(42)
        n_samples = 768
        
        data = {
            'Pregnancies': np.random.poisson(3, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
            'BloodPressure': np.random.normal(70, 12, n_samples).clip(0, 120),
            'SkinThickness': np.random.normal(20, 15, n_samples).clip(0, 100),
            'Insulin': np.random.normal(80, 115, n_samples).clip(0, 850),
            'BMI': np.random.normal(32, 7, n_samples).clip(0, 70),
            'DiabetesPedigreeFunction': np.random.gamma(0.5, 1, n_samples).clip(0, 2.5),
            'Age': np.random.poisson(33, n_samples).clip(21, 81),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic diabetes outcome based on risk factors
        risk_score = (
            (df['Glucose'] > 140) * 2 +
            (df['BMI'] > 30) * 1.5 +
            (df['Age'] > 45) * 1 +
            (df['BloodPressure'] > 80) * 0.5 +
            (df['DiabetesPedigreeFunction'] > 0.5) * 1
        )
        
        df['Outcome'] = (risk_score + np.random.normal(0, 1, n_samples) > 2.5).astype(int)
        
        return df

    # SOLUTION 1: Disease Diagnosis and Prediction
    def train_diagnosis_models(self, df):
        """Train multiple ML models for diabetes diagnosis"""
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['diagnosis'] = scaler
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        model_results = {}
        
        for name, model in models.items():
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            self.models[f'diagnosis_{name.lower().replace(" ", "_")}'] = model
        
        # Train Neural Network
        nn_model = self.train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test)
        self.models['diagnosis_neural_network'] = nn_model
        
        return model_results, X_test, y_test

    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train TensorFlow neural network"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        return model

    # SOLUTION 2: Personalized Care Recommendations
    def create_patient_profiles(self, df):
        """Create patient profiles using clustering for personalized care"""
        # Prepare features for clustering
        features_for_clustering = ['Age', 'BMI', 'Glucose', 'BloodPressure', 'Insulin']
        X_cluster = df[features_for_clustering].copy()
        
        # Scale features
        scaler_cluster = StandardScaler()
        X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
        self.scalers['clustering'] = scaler_cluster
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['PatientProfile'] = kmeans.fit_predict(X_cluster_scaled)
        self.models['patient_clustering'] = kmeans
        
        # Define care recommendations for each profile
        care_profiles = {
            0: {
                'profile_name': 'Low Risk - Young & Healthy',
                'recommendations': [
                    'Maintain current healthy lifestyle',
                    'Annual diabetes screening',
                    'Regular exercise routine',
                    'Balanced diet with portion control'
                ],
                'monitoring_frequency': 'Annual',
                'lifestyle_focus': 'Prevention'
            },
            1: {
                'profile_name': 'Moderate Risk - Pre-diabetic',
                'recommendations': [
                    'Intensive lifestyle modification',
                    'Weight loss program if BMI > 25',
                    'Bi-annual glucose monitoring',
                    'Nutritionist consultation'
                ],
                'monitoring_frequency': 'Every 6 months',
                'lifestyle_focus': 'Risk Reduction'
            },
            2: {
                'profile_name': 'High Risk - Early Diabetes',
                'recommendations': [
                    'Medication management',
                    'Daily glucose monitoring',
                    'Diabetes education program',
                    'Regular endocrinologist visits'
                ],
                'monitoring_frequency': 'Monthly',
                'lifestyle_focus': 'Disease Management'
            },
            3: {
                'profile_name': 'Complex Case - Multiple Factors',
                'recommendations': [
                    'Comprehensive care team approach',
                    'Intensive monitoring',
                    'Medication optimization',
                    'Complication screening'
                ],
                'monitoring_frequency': 'Weekly',
                'lifestyle_focus': 'Complication Prevention'
            }
        }
        
        self.patient_profiles = care_profiles
        return df, care_profiles

    # SOLUTION 3: Patient Support Chatbot
    def generate_patient_response(self, query, patient_data=None):
        """Simulate AI-powered patient support responses"""
        query_lower = query.lower()
        
        # Common queries and responses
        responses = {
            'diabetes': self.get_diabetes_info(),
            'diet': self.get_diet_recommendations(),
            'exercise': self.get_exercise_recommendations(),
            'symptoms': self.get_symptom_info(),
            'medication': self.get_medication_info(),
            'monitoring': self.get_monitoring_info(),
            'emergency': self.get_emergency_info()
        }
        
        # Find relevant response
        for key, response in responses.items():
            if key in query_lower:
                if patient_data:
                    return self.personalize_response(response, patient_data)
                return response
        
        return "I understand you have questions about your health. Could you please be more specific about diabetes, diet, exercise, symptoms, medication, or monitoring?"

    def get_diabetes_info(self):
        return """
        **About Diabetes:**
        Diabetes is a condition where your blood sugar levels are too high. There are two main types:
        
        - **Type 1**: Your body doesn't make insulin
        - **Type 2**: Your body doesn't use insulin properly
        
        **Key Management Steps:**
        1. Monitor blood sugar regularly
        2. Take medications as prescribed
        3. Maintain a healthy diet
        4. Exercise regularly
        5. Attend regular check-ups
        
        Would you like specific information about diet, exercise, or monitoring?
        """

    def get_diet_recommendations(self):
        return """
        **Diabetes-Friendly Diet Tips:**
        
        **Foods to Include:**
        - Whole grains (brown rice, quinoa, oats)
        - Lean proteins (chicken, fish, beans)
        - Non-starchy vegetables (broccoli, spinach, peppers)
        - Healthy fats (avocado, nuts, olive oil)
        
        **Foods to Limit:**
        - Sugary drinks and snacks
        - Refined carbohydrates
        - Processed foods
        - High-sodium foods
        
        **Meal Planning Tips:**
        - Use the plate method (1/2 vegetables, 1/4 protein, 1/4 carbs)
        - Eat at regular times
        - Control portion sizes
        """

    def get_exercise_recommendations(self):
        return """
        **Exercise for Diabetes Management:**
        
        **Recommended Activities:**
        - 150 minutes of moderate aerobic activity per week
        - Strength training 2-3 times per week
        - Daily walking after meals
        
        **Benefits:**
        - Lowers blood sugar levels
        - Improves insulin sensitivity
        - Helps with weight management
        - Reduces cardiovascular risk
        
        **Safety Tips:**
        - Check blood sugar before and after exercise
        - Stay hydrated
        - Wear proper footwear
        - Start slowly and gradually increase intensity
        """

    # SOLUTION 4: Healthcare Provider Efficiency Tools
    def generate_provider_dashboard_data(self, df):
        """Generate efficiency metrics for healthcare providers"""
        total_patients = len(df)
        diabetes_cases = df['Outcome'].sum()
        high_risk_patients = len(df[df['Glucose'] > 140])
        
        # Risk stratification
        df['RiskLevel'] = pd.cut(
            df['Glucose'], 
            bins=[0, 100, 140, 200], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Generate insights
        insights = {
            'total_patients': total_patients,
            'diabetes_prevalence': f"{(diabetes_cases/total_patients)*100:.1f}%",
            'high_risk_count': high_risk_patients,
            'avg_age': df['Age'].mean(),
            'avg_bmi': df['BMI'].mean(),
            'risk_distribution': df['RiskLevel'].value_counts().to_dict()
        }
        
        return insights

    def predict_patient_risk(self, patient_features):
        """Predict diabetes risk for a single patient"""
        # Use the best performing model (Random Forest)
        model = self.models.get('diagnosis_random_forest')
        scaler = self.scalers.get('diagnosis')
        
        if model and scaler:
            # For tree-based models, no scaling needed
            risk_prob = model.predict_proba([patient_features])[0][1]
            return risk_prob
        return 0.5

def main():
    st.markdown('<h1 class="main-header">🏥 Healthcare AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize the system
    if 'healthcare_system' not in st.session_state:
        st.session_state.healthcare_system = HealthcareAISystem()
        
    ai_system = st.session_state.healthcare_system
    
    # Sidebar navigation
    st.sidebar.title("🔬 AI Solutions")
    solution = st.sidebar.selectbox(
        "Select Healthcare Solution:",
        ["🎯 Disease Diagnosis & Prediction", 
         "👤 Personalized Care Plans", 
         "💬 Patient Support Chat", 
         "📊 Provider Efficiency Dashboard",
         "🧪 Model Training & Evaluation"]
    )
    
    # Load data
    df = ai_system.load_sample_data()
    
    if solution == "🎯 Disease Diagnosis & Prediction":
        st.markdown('<div class="solution-card"><h2>Solution 1: Advanced Disease Diagnosis & Prediction</h2></div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Patient Information Input")
            
            # Input form for patient data
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
            bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
            skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, value=80)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            
            if st.button("🔍 Analyze Patient Risk", type="primary"):
                patient_features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
                
                # Train models if not already trained
                if 'diagnosis_random_forest' not in ai_system.models:
                    with st.spinner("Training AI models..."):
                        model_results, X_test, y_test = ai_system.train_diagnosis_models(df)
                
                risk_prob = ai_system.predict_patient_risk(patient_features)
                
                st.session_state.current_risk = risk_prob
                st.session_state.current_patient = patient_features
        
        with col2:
            st.subheader("🎯 Diagnosis Results")
            
            if 'current_risk' in st.session_state:
                risk_prob = st.session_state.current_risk
                
                # Risk visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk %"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70}}))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk interpretation
                if risk_prob < 0.3:
                    st.success("✅ Low Risk: Maintain healthy lifestyle")
                elif risk_prob < 0.7:
                    st.warning("⚠️ Moderate Risk: Consider lifestyle changes")
                else:
                    st.error("🚨 High Risk: Consult healthcare provider immediately")
    
    elif solution == "👤 Personalized Care Plans":
        st.markdown('<div class="solution-card"><h2>Solution 2: Personalized Patient Care Plans</h2></div>', 
                   unsafe_allow_html=True)
        
        # Create patient profiles
        df_with_profiles, care_profiles = ai_system.create_patient_profiles(df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("👥 Patient Profile Analysis")
            
            # Display profile distribution
            profile_counts = df_with_profiles['PatientProfile'].value_counts().sort_index()
            fig = px.pie(
                values=profile_counts.values, 
                names=[care_profiles[i]['profile_name'] for i in profile_counts.index],
                title="Patient Profile Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Select Patient Profile")
            selected_profile = st.selectbox(
                "Choose a patient profile:",
                options=list(care_profiles.keys()),
                format_func=lambda x: care_profiles[x]['profile_name']
            )
            
            profile_info = care_profiles[selected_profile]
            
            st.markdown(f"### {profile_info['profile_name']}")
            st.markdown(f"**Monitoring Frequency:** {profile_info['monitoring_frequency']}")
            st.markdown(f"**Lifestyle Focus:** {profile_info['lifestyle_focus']}")
            
            st.markdown("**Personalized Recommendations:**")
            for rec in profile_info['recommendations']:
                st.markdown(f"• {rec}")
    
    elif solution == "💬 Patient Support Chat":
        st.markdown('<div class="solution-card"><h2>Solution 3: 24/7 Patient Support Chatbot</h2></div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💬 Chat with AI Health Assistant")
            
            # Chat interface
            if 'messages' not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! I'm your AI health assistant. I can help you with questions about diabetes, diet, exercise, medications, and general health monitoring. How can I help you today?"}
                ]
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about your health..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                response = ai_system.generate_patient_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        
        with col2:
            st.subheader("🔗 Quick Help Topics")
            topics = [
                "What is diabetes?",
                "Diet recommendations",
                "Exercise guidelines", 
                "Medication reminders",
                "Blood sugar monitoring",
                "Emergency symptoms"
            ]
            
            for topic in topics:
                if st.button(topic, key=f"topic_{topic}"):
                    response = ai_system.generate_patient_response(topic)
                    st.session_state.messages.append({"role": "user", "content": topic})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    elif solution == "📊 Provider Efficiency Dashboard":
        st.markdown('<div class="solution-card"><h2>Solution 4: Healthcare Provider Efficiency Dashboard</h2></div>', 
                   unsafe_allow_html=True)
        
        # Generate dashboard data
        insights = ai_system.generate_provider_dashboard_data(df)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Patients", insights['total_patients'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Diabetes Prevalence", insights['diabetes_prevalence'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("High Risk Patients", insights['high_risk_count'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Age", f"{insights['avg_age']:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            risk_data = pd.DataFrame(list(insights['risk_distribution'].items()), 
                                   columns=['Risk Level', 'Count'])
            fig = px.bar(risk_data, x='Risk Level', y='Count', 
                        title="Patient Risk Distribution",
                        color='Risk Level',
                        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age vs BMI scatter
            fig = px.scatter(df, x='Age', y='BMI', color='Outcome',
                           title="Age vs BMI Analysis",
                           color_discrete_map={0: 'blue', 1: 'red'},
                           labels={'Outcome': 'Diabetes'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Patient prioritization
        st.subheader("🎯 Patient Prioritization")
        high_priority = df[(df['Glucose'] > 140) | (df['BMI'] > 35) | (df['Age'] > 65)].head()
        st.dataframe(high_priority, use_container_width=True)
    
    elif solution == "🧪 Model Training & Evaluation":
        st.markdown('<div class="solution-card"><h2>Model Training & Performance Evaluation</h2></div>', 
                   unsafe_allow_html=True)
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training multiple AI models..."):
                model_results, X_test, y_test = ai_system.train_diagnosis_models(df)
            
            st.success("✅ All models trained successfully!")
            
            # Model comparison
            st.subheader("📊 Model Performance Comparison")
            
            performance_data = []
            for name, results in model_results.items():
                performance_data.append({
                    'Model': name,
                    'Accuracy': results['accuracy'],
                    'AUC Score': results['auc']
                })
            
            perf_df = pd.DataFrame(performance_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(perf_df, x='Model', y='Accuracy', 
                           title="Model Accuracy Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(perf_df, x='Model', y='AUC Score', 
                           title="Model AUC Score Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model details
            best_model = perf_df.loc[perf_df['Accuracy'].idxmax()]
            st.success(f"🏆 Best Performing Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.3f})")
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.write("Dataset Statistics:")
            st.dataframe(df.describe(), use_container_width=True)

if __name__ == "__main__":
    main()