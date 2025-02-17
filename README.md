# 💉 Medical Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)

![Ekran görüntüsü 2025-02-18 015444](https://github.com/user-attachments/assets/bbb4e28a-b629-45ee-bc14-24f80fc5b93a)

This project is a machine learning-based web application that predicts medical insurance costs based on user input. Built with Gradient Boosting and Streamlit, it provides accurate predictions based on real insurance data.

🔗 [Live Demo: Medical Insurance Cost Predictor](https://01-medical-insurance-cost-prediction.streamlit.app/)

## 📊 Overview

A robust machine learning system that analyzes various personal factors to provide instant and accurate insurance cost predictions. The application offers a user-friendly interface for easy interaction.

## ⭐ Features

- Real-time cost predictions based on personal data
- User-friendly web interface
- Analysis of personal health factors
- Detailed result reporting
- Advanced ML model (Gradient Boosting)
- Instant cost calculation
- Comprehensive factor analysis

## 💻 Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web framework for interactive UI
- **Scikit-learn** - Model training & preprocessing
- **Gradient Boosting** - Advanced regression model
- **Pandas & NumPy** - Data processing & transformation
- **Pickle** - Model & feature storage

## 🛠️ Installation & Setup

### Download Required Model Files

The following files must be present in the `models/` directory:

```bash
models/
├── final_model.pkl    # Trained ML Model
├── feature_columns.pkl    # Feature columns used during training
└── scaler.pkl    # Scaler for normalizing input data
```

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/yourusername/medical_insurance_cost_prediction.git
```

### Navigate to project directory
cd medical_insurance_cost_prediction

### Install required packages

pip install -r requirements.txt

## Run the application
streamlit run app.py

📁 Project Structure
```bash
Medical Insurance Cost Prediction
├── models/
│   ├── final_model.pkl          # Trained ML model
│   ├── feature_columns.pkl      # Feature set used for training
│   └── scaler.pkl              # Scaler for numerical features
├── app.py                      # Streamlit web application
├── insurance.csv              # Original dataset
├── requirements.txt           # Required Python packages
└── README.md                 # Documentation
```
## 🎯 How to Use

- Enter personal details (age, BMI, smoking status, region, etc.)
- Click the "Predict Insurance Cost" button
- View the predicted insurance cost instantly! 💰

## 📊 Dataset
The model is trained on the Medical Insurance Dataset, featuring:

1,338 insurance records
Comprehensive personal information
Real market data

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing & Contact
Interested in contributing? Follow these steps:

Fork the repository
Create a new branch (feature-addition)
Make your changes and commit them
Submit a Pull Request (PR) for review

## Fork the repository
Create a new branch (feature-addition) Make your changes and commit them Submit a Pull Request (PR) for review



















