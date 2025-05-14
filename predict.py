import joblib
import pandas as pd

# Load the trained model
model = joblib.load('xgb_employee_enrollment_model.pkl')
print("Model loaded successfully.")

# function for single prediction
def predict_enrollment(data):
    """
    Predicts if the employee will enroll in the voluntary insurance product.

    Parameters:
    data (dict): A dictionary containing the employee's data.

    Returns:
    int: Prediction (0 or 1).
    """
    # Convert to DataFrame (single row)
    df = pd.DataFrame([data])
    
    # Predict using the loaded model
    prediction = model.predict(df)
    
    # Display the result
    print(f"Predicted Enrollment: {'Yes' if prediction[0] == 1 else 'No'}")
    return prediction[0]