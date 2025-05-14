import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from data_preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, accuracy_score
import wandb

def tune_hyperparameters():
    # Initialize W&B experiment tracking
    wandb.init(project="Uniblox_assignment", entity="ajayansaroj")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data("employee_data.csv")

    # Define the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10]
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    print("Best parameters:", grid_search.best_params_)

    # Evaluate the tuned model
    y_pred_tuned = grid_search.best_estimator_.predict(X_test)
    print(classification_report(y_test, y_pred_tuned))
    print("Accuracy Score: ", accuracy_score(y_test, y_pred_tuned))

    # Log the results to W&B
    wandb.log({"best_params": grid_search.best_params_})
    wandb.log({"classification_report": classification_report(y_test, y_pred_tuned, output_dict=True)})

    wandb.finish()
