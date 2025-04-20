from preprocessing import load_data
from model import PneumoniaClassifier
from evaluation import evaluate_model

DATASET_PATH = 'D:\OneDrive - Canadian University Dubai\CUD\SPRING 2024\Programming Paradigms\Pneumonia-AI\data\chest_xray'
EPOCHS = 10

# Load data
train_data, val_data, test_data = load_data(DATASET_PATH)

# Initialize and train model
classifier = PneumoniaClassifier()
history = classifier.train(train_data, val_data, EPOCHS)
# classifier.model.save("pneumonia_model.h5")

# Evaluate model
loss, accuracy = classifier.evaluate(test_data)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predictions and report
predicted_probs = classifier.predict(test_data)
true_classes = test_data.classes
report, matrix = evaluate_model(true_classes, predicted_probs)

print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)