# Uniblox_assignment :Assignmment for uniblox assesment

# Employee Insurance Enrollment Prediction

This project is designed to predict whether an employee will opt-in for a new voluntary insurance product based on their demographic and employment-related data. It consists of a machine learning pipeline for data preprocessing, model training, and deployment using a Streamlit web application.

## 📌 **Project Structure**

```
project_folder/
│
├── employee_data.csv                 # Employee demographic and employment data
├── data_preprocessing.py             # Data preprocessing pipeline
├── train_model.py                    # Model training script
├── predict.py                        # Prediction script
├── employee_enrollment_model.pkl     # Trained model
├── app.py                            # Streamlit application
└── README.md                         # Project documentation
```

## 🚀 **Modules Explanation**

### 1️⃣ **data\_preprocessing.py**

This module handles the following:

* Loads the employee data from a CSV file
* Splits the data into training and test sets
* Applies transformations:

  * Standard scaling for numerical features
  * One-hot encoding for categorical features

### 2️⃣ **train\_model.py**

This module performs:

* Model training using RandomForestClassifier
* Hyperparameter tuning using GridSearchCV
* Saves the trained model as `employee_enrollment_model.pkl`

### 3️⃣ **predict.py**

This module is responsible for:

* Loading the trained model
* Accepting a sample input for prediction
* Returning the enrollment prediction (0 for No, 1 for Yes)

### 4️⃣ **app.py**

Streamlit application that provides:

* An interactive UI for user input
* Real-time prediction of employee enrollment
* Displays the prediction result interactively

---

## ⚙️ **Model Training and Prediction**

1. Run `train_model.py` to train the model:

```bash
python train_model.py
```

2. Use `predict.py` for testing a single input:

```bash
python predict.py
```

---

## 🌐 **Streamlit Application Usage**

To launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`, where you can:

* Enter employee demographic and employment details
* Get real-time predictions for insurance enrollment

---

## 📦 **Dependencies and Setup**

Install dependencies:

```bash
pip install -r requirements.txt
```

### **requirements.txt** should include:

* pandas
* scikit-learn
* joblib
* streamlit

---

## 🔄 **Future Enhancements**

* Add feature importance visualization to the Streamlit app
* Include history tracking for predictions
* Improve hyperparameter tuning with Bayesian Optimization

---

## 📞 **Contact Information**

Developed by **Ajayan Saroj**. Feel free to reach out for collaborations or queries.

LinkedIn: \[https://www.linkedin.com/in/ajayan-saroj-7b0200133/]
Email: \[ajayansaroj@gmail.com]
