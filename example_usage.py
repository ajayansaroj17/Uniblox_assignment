from predict import predict_enrollment

# Sample input
sample_input = {
    "age": 45,
    "gender": "Female",
    "marital_status": "Single",
    "salary": 70292.40,
    "employment_type": "Full-time",
    "region": "South",
    "has_dependents": "Yes",
    "tenure_years": 6.5
}

# Make a prediction
predict_enrollment(sample_input)
