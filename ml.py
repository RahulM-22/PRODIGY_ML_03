import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load and preprocess data from a single folder
def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if 'cat' in filename:  # Assign label 0 for cats
            label = 0
        elif 'dog' in filename:  # Assign label 1 for dogs
            label = 1
        else:
            continue  # Skip files that don't match 'cat' or 'dog'

        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img.flatten())  # Flatten to 1D
            labels.append(label)
    return images, labels

# Specify path for the dataset folder containing both cat and dog images
dataset_path = 'C:/Users/rahul mhapankar/Desktop/ML 03/train'

# Load the images and labels
images, labels = load_images_from_folder(dataset_path)

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Perform PCA for dimensionality reduction (optional but recommended)
pca = PCA(n_components=150)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. Train the SVM model with hyperparameter tuning
# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Initialize the SVM model
svm = SVC()

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_pca, y_train)

# Print the best parameters found by GridSearchCV
print(f"Best parameters found: {grid_search.best_params_}")

# Train the best model
best_svm = grid_search.best_estimator_

# 5. Evaluate the model
y_pred = best_svm.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# 6. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
