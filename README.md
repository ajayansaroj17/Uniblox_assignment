# Employee Insurance Enrollment Prediction

This project aims to predict whether an employee will opt-in to a voluntary insurance product based on their demographic and employment-related data using **XGBoost**.

---

## 📁 **Project Structure**

```
project_folder/
│
├── employee_data.csv                 # Dataset for training and evaluation
├── data_preprocessing.py             # Data loading and preprocessing logic
├── train_model.py                    # Training logic using XGBoost
├── predict.py                        # Prediction logic
├── xgb_employee_enrollment_model.pkl     # Trained model (generated after training)
└── README.md                         # Project documentation (You are here)
```

---

## ⚙️ **Modules Overview**

### 1️⃣ **data\_preprocessing.py**

* Loads the dataset.
* Splits the data into **training** and **testing** sets.
* Identifies categorical and numerical features.
* Applies preprocessing steps:

  * Numerical data → Standard Scaling
  * Categorical data → One-Hot Encoding

### 2️⃣ **train\_model.py**

* Imports preprocessed data.
* Trains an **XGBoost Classifier** with default settings.
* Evaluates using:

  * **Accuracy Score**
  * **ROC AUC Score**
  * **Classification Report**
  * **Confusion Matrix**
* Saves the trained model as `xgb_employee_enrollment_model.pkl`.

### 3️⃣ **predict.py**

* Loads the saved model.
* Accepts user input and processes it using the trained pipeline.
* Returns the **prediction**

---

## 🚀 **How to Run the Application**

1️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

2️⃣ **Train the Model**

```bash
python train_model.py
```

## 📊 **Evaluation Metrics**

* **Accuracy Score**: Measures the overall correctness of the model.
* **ROC AUC Score**: Evaluates the model's capability to distinguish between classes.
* **Classification Report**: Detailed precision, recall, and F1-score.
* **Confusion Matrix**: Visualization of prediction vs. actual labels.
---

## 🤝 **Contributions**

Feel free to fork this repository, create issues, or submit pull requests for improvements!

---

## 📝 **Author**

**Ajayan Saroj**
ajayansaroj@gmail.com

For any queries, feel free to reach out!

---
