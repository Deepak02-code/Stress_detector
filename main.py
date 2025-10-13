# ------------------------ STRESS DETECTOR (Flask Version) -----------------------
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ------------------ MODEL TRAINING ------------------
df = pd.read_csv("SaYoPillow.csv")
df.columns = [
    'snoring', 'respiration_rate', 'body_temp', 'limb_movement',
    'blodd_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate', 'stress_level'
]
df['stressed'] = df['stress_level'].apply(lambda x: 1 if x > 0 else 0)

# Stage 1 - Binary
X = df[['snoring', 'respiration_rate', 'body_temp', 'limb_movement',
        'blodd_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate']]
y = df['stressed']
scaler1 = StandardScaler()
X_scaled = scaler1.fit_transform(X)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_stage1 = LogisticRegression()
model_stage1.fit(X_train1, y_train1)

# Stage 2 - Multiclass
df_stressed = df[df['stressed'] == 1]
X2 = df_stressed[['snoring', 'respiration_rate', 'body_temp', 'limb_movement',
                  'blodd_oxygen', 'eye_movement', 'sleep_hours', 'heart_rate']]
y2 = df_stressed['stress_level']
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2_scaled, y2, test_size=0.2, random_state=42)
model_stage2 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_stage2.fit(X_train2, y_train2)

# ------------------ PREDICTION FUNCTION ------------------
def predict_stress(values):
    values = np.array(values).reshape(1, -1)
    scaled_1 = scaler1.transform(values)
    stress_pred = model_stage1.predict(scaled_1)[0]

    if stress_pred == 0:
        return "You are NOT stressed 😌", 0
    else:
        scaled_2 = scaler2.transform(values)
        level = model_stage2.predict(scaled_2)[0]
        return f"You are STRESSED 😩 | Stress Level: {level}", level

# ------------------ FLASK ROUTES ------------------
@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['snoring']),
            float(request.form['respiration_rate']),
            float(request.form['body_temp']),
            float(request.form['limb_movement']),
            float(request.form['blodd_oxygen']),
            float(request.form['eye_movement']),
            float(request.form['sleep_hours']),
            float(request.form['heart_rate'])
        ]
        result_text, stress_level = predict_stress(features)
        return render_template('index.html', result=result_text, level=stress_level)
    except Exception as e:
        return render_template('index.html', result=f"Error: {e}", level=None)

if __name__ == '__main__':
    app.run(debug=True)
