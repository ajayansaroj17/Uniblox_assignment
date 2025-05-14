### Using Random forest

# # train_model.py
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from data_preprocessing import load_and_preprocess_data
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# def train_model():
#     # Load and preprocess data
#     X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("employee_data.csv")

#     # Define the model pipeline
#     model = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', RandomForestClassifier(random_state=42))
#     ])

#     # Train the model
#     model.fit(X_train, y_train)

#     # Predictions and evaluation
#     y_pred = model.predict(X_test)
#     print("Classification Report:\n", classification_report(y_test, y_pred))
#     print("Accuracy Score: ", accuracy_score(y_test, y_pred))

#     # Save the model
#     joblib.dump(model, 'randomforest_employee_enrollment_model.pkl')
#     print("Model saved as employee_enrollment_model.pkl")






### Using xgboost model
# train_model.py
from data_preprocessing import load_and_preprocess_data
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import joblib

def train_model():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('employee_data.csv')

    # Define the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'xgb_employee_enrollment_model.pkl')
    print("Model saved as 'employee_enrollment_model.pkl'")
