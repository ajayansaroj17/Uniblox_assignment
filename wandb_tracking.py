# wandb_tracking.py
import wandb
from sklearn.metrics import classification_report

def init_wandb():
    wandb.init(project="Uniblox_assignment", entity="ajayansaroj")

def log_metrics(roc_auc, classification_report):
    wandb.log({"roc_auc": roc_auc})
    wandb.log({"classification_report": classification_report})

def finish_wandb():
    wandb.finish()
