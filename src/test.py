from keras._tf_keras.keras.models import load_model
from preprocessing import load_data
from evaluation import evaluate_model

# Load model
model = load_model("saved_models\pneumonia_model.h5")

# Load test data
_, _, test_data = load_data("D:\OneDrive - Canadian University Dubai\CUD\SPRING 2024\Programming Paradigms\Pneumonia-AI\data\chest_xray")

# Evaluate
loss, accuracy = model.evaluate(test_data)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predictions
predicted_probs = model.predict(test_data)
true_classes = test_data.classes

report, matrix = evaluate_model(true_classes, predicted_probs)

print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)
