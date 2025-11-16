# ------------------------ STRESS DETECTOR (Flask Version) -----------------------
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ------------------ MODEL TRAINING ------------------
print("Loading dataset...")
df = pd.read_csv("SaYoPillow.csv")
df.columns = [
    'snoring', 'respiration_rate', 'body_temp', 'limb_movement',
    'blodd_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate', 'stress_level'
]
df['stressed'] = df['stress_level'].apply(lambda x: 1 if x > 0 else 0)

# Stage 1 - Binary Classification
X = df[['snoring', 'respiration_rate', 'body_temp', 'limb_movement',
        'blodd_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate']]
y = df['stressed']
scaler1 = StandardScaler()
X_scaled = scaler1.fit_transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Stage 2 - Multiclass Classification
df_stressed = df[df['stressed'] == 1]
X2 = df_stressed[['snoring', 'respiration_rate', 'body_temp', 'limb_movement',
                  'blodd_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate']]
y2 = df_stressed['stress_level']
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)

# Train all models for Stage 1 (Binary)
print("Training Stage 1 models...")
models_stage1 = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

trained_models_stage1 = {}
accuracies_stage1 = {}

for name, model in models_stage1.items():
    model.fit(X_train1, y_train1)
    y_pred = model.predict(X_test1)
    acc = accuracy_score(y_test1, y_pred)
    trained_models_stage1[name] = model
    accuracies_stage1[name] = acc
    print(f"{name} (Stage 1) - Accuracy: {acc:.4f}")

# Train all models for Stage 2 (Multiclass)
print("Training Stage 2 models...")
models_stage2 = {
    'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

trained_models_stage2 = {}
accuracies_stage2 = {}

for name, model in models_stage2.items():
    model.fit(X_train2, y_train2)
    y_pred = model.predict(X_test2)
    acc = accuracy_score(y_test2, y_pred)
    trained_models_stage2[name] = model
    accuracies_stage2[name] = acc
    print(f"{name} (Stage 2) - Accuracy: {acc:.4f}")

print("All models trained successfully!")

# ------------------ PREDICTION FUNCTION ------------------
def predict_stress_all_models(values):
    """
    Predict stress using all models and return results with accuracy scores
    """
    values = np.array(values).reshape(1, -1)
    scaled_1 = scaler1.transform(values)
    scaled_2 = scaler2.transform(values)
    
    results = []
    
    for model_name in models_stage1.keys():
        # Stage 1 prediction
        model_stage1 = trained_models_stage1[model_name]
        stress_pred = model_stage1.predict(scaled_1)[0]
        acc_stage1 = accuracies_stage1[model_name]
        
        if stress_pred == 0:
            result_text = "NOT stressed 😌"
            stress_level = 0
            acc_stage2 = None
        else:
            # Stage 2 prediction
            model_stage2 = trained_models_stage2[model_name]
            level = model_stage2.predict(scaled_2)[0]
            result_text = f"STRESSED 😩 | Level: {level}"
            stress_level = int(level)
            acc_stage2 = accuracies_stage2[model_name]
        
        results.append({
            'model': model_name,
            'result': result_text,
            'level': stress_level,
            'accuracy_stage1': acc_stage1,
            'accuracy_stage2': acc_stage2
        })
    
    return results

# ------------------ FLASK ROUTES ------------------
@app.route('/')
def home():
    return render_template('index.html', results=None, accuracies_stage1=accuracies_stage1, accuracies_stage2=accuracies_stage2)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['snoring']),
            float(request.form['respiration_rate']),
            float(request.form['body_temp']),
            float(request.form['limb_movement']),
            float(request.form['blood_oxygen']),
            float(request.form['eye_movement']),
            float(request.form['sleep_hours']),
            float(request.form['heart_rate'])
        ]
        results = predict_stress_all_models(features)
        return render_template('index.html', 
                             results=results, 
                             accuracies_stage1=accuracies_stage1, 
                             accuracies_stage2=accuracies_stage2)
    except Exception as e:
        return render_template('index.html', 
                             results=None, 
                             error=str(e),
                             accuracies_stage1=accuracies_stage1, 
                             accuracies_stage2=accuracies_stage2)

if __name__ == '__main__':
    app.run(debug=True)
