# main.py
from train_model import train_model
from hyperparameter_tuning import tune_hyperparameters

def run_pipeline():
    print("Training Model...")
    train_model()
    print("\nHyperparameter Tuning...")
    tune_hyperparameters()

if __name__ == "__main__":
    run_pipeline()
