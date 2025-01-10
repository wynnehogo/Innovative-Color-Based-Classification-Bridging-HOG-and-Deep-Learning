import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Paths
data_dir = "D:/C/wyns/0. m Machine Learning/archive (1)/Animals"
image_size = (128, 128)  # Resize all images to 128x128
n_colors = 3  # Number of clusters for dominant color extraction

# Load pre-trained VGG16 model without the top classification layer
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
vgg_model.trainable = False  # Set VGG16 layers to non-trainable

# Helper functions
def extract_color_features(image, n_colors):
    """Extract dominant color features using KMeans clustering."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.flatten()

def categorize_by_color(image, n_colors=3):
    """Categorize images by dominant color using KMeans and HSV clustering."""
    color_features = extract_color_features(image, n_colors)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[..., 0]

    if np.mean(hue) < 30:  # Redish
        return 'reddish', color_features
    elif 30 <= np.mean(hue) < 90:  # Greenish
        return 'greenish', color_features
    else:  # Bluish
        return 'bluish', color_features

def extract_vgg_features(image):
    """Extract features from the VGG16 model."""
    img = cv2.resize(image, image_size)
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg_model.predict(img)
    return features.flatten()

def preprocess_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return image

# Data loading and feature extraction
X = []
y = []
color_groups = {'reddish': [], 'greenish': [], 'bluish': []}
grouped_images = {'reddish': [], 'greenish': [], 'bluish': []}
y_labels = {'reddish': [], 'greenish': [], 'bluish': []}

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                image = preprocess_image(image_path)
                vgg_features = extract_vgg_features(image)
                color_group, color_features = categorize_by_color(image, n_colors)
                combined_features = np.concatenate([vgg_features, color_features])
                color_groups[color_group].append(combined_features)
                grouped_images[color_group].append(combined_features)
                y_labels[color_group].append(class_name)
                X.append(combined_features)
                y.append(class_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# PCA for dimensionality reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Hyperparameter grids
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}

# Function for hyperparameter tuning
def perform_hyperparameter_tuning(X_train, y_train, classifier_name):
    if classifier_name == "Random Forest":
        clf = RandomForestClassifier(random_state=42)
        param_grid = rf_param_grid
    elif classifier_name == "SVM":
        clf = SVC(random_state=42)
        param_grid = svm_param_grid
    elif classifier_name == "k-NN":
        clf = KNeighborsClassifier()
        param_grid = knn_param_grid
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")
    grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Train and evaluate with hyperparameter tuning
def train_and_evaluate_with_tuning(grouped_images, y_labels, label_encoder):
    results = {}
    for color_group in grouped_images:
        print(f"Training and tuning for {color_group} group...")
        X_group = np.array(grouped_images[color_group])
        y_group = np.array(y_labels[color_group])
        y_group_encoded = label_encoder.fit_transform(y_group)
        X_train, X_test, y_train, y_test = train_test_split(X_group, y_group_encoded, test_size=0.2, random_state=42)
        classifiers = ["Random Forest", "SVM", "k-NN"]
        results[color_group] = {}
        for clf_name in classifiers:
            print(f"Tuning hyperparameters for {clf_name}...")
            best_model, best_params = perform_hyperparameter_tuning(X_train, y_train, clf_name)
            print(f"Best parameters for {clf_name}: {best_params}")
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[color_group][clf_name] = {
                "classification_report": classification_report(y_test, y_pred, target_names=label_encoder.classes_),
                "accuracy": accuracy,
                "best_params": best_params
            }
    return results

# Run training and evaluation
results_with_tuning = train_and_evaluate_with_tuning(grouped_images, y_labels, label_encoder)

# Display results
for color_group, classifiers_report in results_with_tuning.items():
    print(f"\nResults for {color_group} group:")
    for classifier_name, report in classifiers_report.items():
        print(f"\n{classifier_name} Classification Report:")
        print(report["classification_report"])
        print(f"{classifier_name} Accuracy: {report['accuracy']:.4f}")
        print(f"{classifier_name} Best Parameters: {report['best_params']}")
        
# Visualize PCA
def visualize_pca(X, y, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(np.unique(y)):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=labels.classes_[i])
    plt.title("PCA of Features")
    plt.legend()
    plt.show()

visualize_pca(X_pca, y_encoded, label_encoder)
