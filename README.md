# SugarSense – Diabetes Risk Screening

## Overview

SugarSense is a machine learning project that demonstrates diabetes risk screening using health-related input features. It includes a Streamlit web application that displays a model-estimated probability and a screening result.

**Disclaimer:** This project is for educational purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

## Features

* Interactive Streamlit interface
* Machine learning classification pipeline
* Estimated probability and configurable classification threshold
* Feature engineering using glucose, BMI, and age
* Model evaluation using classification metrics
* ROC and Precision–Recall curves

## Dataset

The project uses `diabetes_screening_data.csv`, containing 768 rows and 9 columns, including the `Outcome` target column.

## Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* Streamlit
* Matplotlib

## Project Structure

```text
SugarSense/
├── app.py
├── requirements.txt
├── README.md
├── diabetes_screening_data.csv
├── SugarSense_Diabetes_Risk_Screening.ipynb
└── artifacts/
    └── sugarsense_model.joblib
```

## Run Locally

1. Install Python.

2. Open a terminal in the project directory.

3. Install the dependencies:

   `python -m pip install -r requirements.txt`

4. Start the application:

   `python -m streamlit run app.py`

5. Open `http://localhost:8501` in your browser.

## Limitations

The model's evaluation results are preliminary and depend on the data split, preprocessing, and threshold-selection process. Further validation is required before drawing conclusions about its performance in real-world populations.

## Author

Sai Charanya Koneti
