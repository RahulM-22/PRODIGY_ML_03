import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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

# 4. Train the SVM model
svm = SVC(kernel='rbf', C=1, gamma='scale')  # RBF kernel is effective for image data
svm.fit(X_train_pca, y_train)

# 5. Evaluate the model
y_pred = svm.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
