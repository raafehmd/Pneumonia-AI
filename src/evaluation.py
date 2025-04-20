from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(true_classes, predicted_probs):
    predicted_classes = np.round(predicted_probs).astype(int)
    report = classification_report(true_classes, predicted_classes, target_names=['NORMAL', 'PNEUMONIA'])
    matrix = confusion_matrix(true_classes, predicted_classes)
    return report, matrix