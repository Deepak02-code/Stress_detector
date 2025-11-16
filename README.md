# 🧠 Smart Stress Detector

A web-based machine learning application that predicts stress levels based on sleep and physiological data. The application uses a two-stage logistic regression model to first determine if a person is stressed, and then classify the stress level if stress is detected.

## 📋 Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license) 

## ✨ Features

- **Two-Stage Prediction System**: 
  - Stage 1: Binary classification (stressed vs. not stressed)
  - Stage 2: Multiclass classification (stress levels 1-4) for stressed individuals
- **User-Friendly Web Interface**: Modern, responsive design with Bootstrap 5
- **Real-Time Predictions**: Instant stress level predictions based on user input
- **Visual Feedback**: Animated results with emoji indicators
- **Educational Information**: Built-in guide explaining stress levels

## 🛠 Technology Stack

### Backend
- **Python 3.x**
- **Flask**: Web framework for building the application
- **scikit-learn**: Machine learning library for model training and prediction
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Frontend
- **HTML5**: Structure
- **Bootstrap 5.3.0**: Responsive UI framework
- **CSS3**: Custom styling with animations
- **AOS (Animate On Scroll)**: Scroll animations

## 📁 Project Structure

```
Stress_Detector/
│
├── main.py                 # Main Flask application and ML model
├── SaYoPillow.csv          # Training dataset
├── README.md               # Project documentation
│
├── templates/
│   └── index.html          # Main HTML template
│
└── static/
    └── style.css           # Custom CSS styles
```

## 🔬 How It Works

The application uses a **two-stage machine learning approach**:

1. **Stage 1 - Binary Classification**:
   - Determines if the person is stressed (1) or not stressed (0)
   - Uses Logistic Regression with standardized features
   - Trained on the entire dataset

2. **Stage 2 - Multiclass Classification**:
   - Only activated if Stage 1 predicts stress
   - Classifies stress level from 1 to 4
   - Uses Multinomial Logistic Regression
   - Trained only on stressed samples from the dataset

### Input Features

The model analyzes 8 physiological and sleep-related features:

1. **Snoring** - Snoring rate/level
2. **Respiration Rate** - Breathing rate per minute
3. **Body Temperature** - Body temperature in Fahrenheit
4. **Limb Movement** - Frequency of limb movements
5. **Blood Oxygen** - Blood oxygen saturation level
6. **Eye Movement** - REM (Rapid Eye Movement) activity
7. **Sleep Hours** - Total hours of sleep
8. **Heart Rate** - Heart rate in beats per minute

### Stress Levels

- **0**: Low / Normal (No stress)
- **1**: Medium Low
- **2**: Medium
- **3**: Medium High
- **4**: High Stress

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Stress_Detector
```

### Step 2: Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

Install required packages:

```bash
pip install flask pandas numpy scikit-learn
```

Or create a `requirements.txt` file with:

```
Flask==2.3.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
```

Then install:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Application

1. Ensure you're in the project directory
2. Make sure `SaYoPillow.csv` is in the root directory
3. Run the Flask application:

```bash
python main.py
```

4. Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

### Making Predictions

1. Fill in all 8 input fields with your sleep and physiological data
2. Click the "Predict Stress Level" button
3. View your stress prediction result in the right panel

### Example Input Values

- Snoring: 50-95
- Respiration Rate: 15-25
- Body Temperature: 90-100
- Limb Movement: 5-20
- Blood Oxygen: 85-100
- Eye Movement: 60-100
- Sleep Hours: 1-9
- Heart Rate: 50-80

## 🧮 Model Architecture

### Data Preprocessing

- **Feature Scaling**: StandardScaler is used to normalize features for both stages
- **Train-Test Split**: 80-20 split with random_state=42 for reproducibility

### Model Details

**Stage 1 Model:**
- Algorithm: Logistic Regression
- Type: Binary Classification
- Features: 8 standardized features
- Output: 0 (not stressed) or 1 (stressed)

**Stage 2 Model:**
- Algorithm: Multinomial Logistic Regression
- Type: Multiclass Classification
- Solver: L-BFGS
- Max Iterations: 1000
- Features: 8 standardized features
- Output: Stress level (1, 2, 3, or 4)

## 📊 Dataset

The application uses the **SaYoPillow.csv** dataset, which contains sleep and physiological data with corresponding stress levels. The dataset includes:

- Multiple sleep-related features
- Stress level labels (0-4)
- Sufficient samples for training both binary and multiclass models

## 🌐 API Endpoints

### GET `/`
- **Description**: Renders the main prediction form
- **Response**: HTML page with input form

### POST `/predict`
- **Description**: Processes form data and returns stress prediction
- **Request Body**: Form data with 8 features
  - `snoring` (float)
  - `respiration_rate` (float)
  - `body_temp` (float)
  - `limb_movement` (float)
  - `blood_oxygen` (float)
  - `eye_movement` (float)
  - `sleep_hours` (float)
  - `heart_rate` (float)
- **Response**: HTML page with prediction result

## 🎨 Screenshots

The application features:
- Modern gradient background (blue to purple)
- Responsive card-based layout
- Smooth animations and transitions
- Clear result display with emoji indicators
- Informational sidebar explaining stress levels

## ⚠️ Important Notes

1. **Typo in Code**: There's a typo in the codebase where `blodd_oxygen` is used in some places instead of `blood_oxygen`. The form field correctly uses `blood_oxygen`, but the column name in the dataset processing uses `blodd_oxygen`. This works as long as the CSV column is also named with the typo, but it's recommended to fix this for consistency.

2. **Model Training**: The model is trained every time the application starts. For production use, consider saving trained models using `joblib` or `pickle` to improve startup time.

3. **Debug Mode**: The application runs in debug mode (`debug=True`), which is suitable for development but should be disabled in production.

## 🔧 Future Improvements

- [ ] Save trained models to disk to avoid retraining on each startup
- [ ] Add model evaluation metrics (accuracy, precision, recall)
- [ ] Implement data validation and input range checking
- [ ] Add user authentication and prediction history
- [ ] Create API endpoints for programmatic access
- [ ] Add data visualization for predictions
- [ ] Implement model versioning
- [ ] Add unit tests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a stress detection tool using machine learning and web technologies.

## 🙏 Acknowledgments

- Dataset: SaYoPillow.csv
- Flask community for the excellent web framework
- scikit-learn team for the machine learning library

---

**Note**: This application is for educational and research purposes. For medical stress assessment, please consult healthcare professionals.

