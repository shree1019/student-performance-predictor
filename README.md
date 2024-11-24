
# Student Performance Predictor

This project predicts a student's future grade and trend based on their past academic performance. It uses a machine learning model trained on student performance data and provides an easy-to-use interface through a Flask API and a Streamlit frontend.

---

## Table of Contents

- [Introduction](#introduction)
- [Approach](#approach)
  - [Objective](#objective)
  - [Steps Followed](#steps-followed)
    - [Data Preparation](#data-preparation)
    - [Model Training](#model-training)
    - [API Development](#api-development)
    - [Frontend Development](#frontend-development)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [How to Run the Application](#how-to-run-the-application)
  - [Run the Flask API](#run-the-flask-api)
  - [Run the Streamlit Frontend](#run-the-streamlit-frontend)
- [Input and Output](#input-and-output)
  - [API Input Format](#api-input-format)
  - [API Output Format](#api-output-format)
  - [Streamlit Frontend](#streamlit-frontend)
- [File Structure](#file-structure)
- [Future Enhancements](#future-enhancements)

---

## Introduction

The **Student Performance Predictor** leverages machine learning to predict a student's future grade (G3) and assess their performance trend (improving, declining, or steady) based on past academic data. It offers an intuitive interface for users through a Streamlit app and exposes an API for programmatic interaction.

---

## Approach

### Objective

To predict a student's final grade (G3) based on:
- Past grades (G1 and G2)
- Study time
- Number of failures
- Total absences

Additionally, the model evaluates the performance trend based on grade progression.

### Steps Followed

#### Data Preparation
1. Used the `student-mat.csv` dataset from the UCI Machine Learning Repository.
2. Added a new feature `trend`, calculated as the rate of change between G1 and G2.
3. Cleaned and normalized the dataset using a `StandardScaler`.

#### Model Training
1. Split the dataset into training and testing sets (80%-20%).
2. Trained a **Random Forest Regressor** to predict G3.
3. Achieved a low **Mean Squared Error (MSE)** on the test set.

#### API Development
1. Built a Flask API to serve the trained model.
2. Implemented a POST endpoint (`/predict`) to accept input data in JSON format and return predictions (grade and trend).

#### Frontend Development
1. Created a Streamlit app to provide a user-friendly interface for interacting with the API.
2. Users can input data, send it to the API, and view the predicted grade and trend.

---

## Setup

### Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  ```bash
  pip install flask streamlit requests numpy pandas scikit-learn joblib
  ```

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/shree1019/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. Download the `student-mat.csv` dataset from the UCI ML Repository and place it in the project directory.  
   [Dataset link: UCI ML Repository - Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

3. Train the machine learning model:
   ```bash
   python train_model.py
   ```
   This will generate `student_performance_model.pkl` and `scaler.pkl`.

---

## How to Run the Application

### Run the Flask API

Start the API server:
```bash
python app.py
```
The API will be accessible at `http://127.0.0.1:5000`.

### Run the Streamlit Frontend

In a new terminal, start the Streamlit app:
```bash
streamlit run streamlit_app.py
```
The Streamlit app will open in your browser at `http://localhost:8501`.

---

## Input and Output

### API Input Format

The API accepts a POST request at `/predict` with the following JSON structure:

```json
{
  "G1": 10,
  "G2": 12,
  "studytime": 3,
  "failures": 0,
  "absences": 5
}
```

| Parameter   | Description                                 | Example Value |
|-------------|---------------------------------------------|---------------|
| `G1`        | Grade for the first period (0-20)          | 10            |
| `G2`        | Grade for the second period (0-20)         | 12            |
| `studytime` | Weekly study time (1: <2h, 4: >10h)        | 3             |
| `failures`  | Number of past class failures (0-5)        | 0             |
| `absences`  | Total number of absences                   | 5             |

### API Output Format

The API responds with the predicted grade and performance trend:

```json
{
  "predicted_grade": 14.75,
  "trend": "improving"
}
```

| Field             | Description                                       | Example Value |
|-------------------|---------------------------------------------------|---------------|
| `predicted_grade` | Predicted final grade (G3)                        | 14.75         |
| `trend`           | Performance trend (improving, declining, or steady) | improving     |

### Streamlit Frontend

- **Input Fields**: Users provide values for:
  - G1, G2, Study Time, Failures, and Absences.
- **Output**:
  - Predicted Grade
  - Performance Trend

#### Example

**Input:**
```yaml
G1: 10
G2: 12
Study Time: 3
Failures: 0
Absences: 5
```

**Output:**
```yaml
Predicted Grade: 14.75
Trend: Improving
```

---

## File Structure

```bash
.
├── app.py                  # Flask API
├── train_model.py          # Model training script
├── streamlit_app.py        # Streamlit frontend
├── student-mat.csv         # Dataset file
├── student_performance_model.pkl # Trained model
├── scaler.pkl              # Saved scaler
└── README.md               # Project documentation
```

---

