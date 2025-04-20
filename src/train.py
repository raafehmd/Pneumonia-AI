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

# save model
classifier.model.save("pneumonia_model.h5")