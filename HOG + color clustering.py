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
from skimage.feature import hog

# Paths
data_dir = "D:/C/wyns/0. m Machine Learning/archive (1)/Animals"
# Parameters
image_size = (128, 128)  # Resize all images to 128x128
n_colors = 3  # Number of clusters for dominant color extraction

# Helper functions
def extract_color_features(image, n_colors):
    """Extract dominant color features using KMeans clustering."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.flatten()

def extract_hog_features(image):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features, _ = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def preprocess_image(image_path):
    """Load and preprocess an image with bilateral filtering for noise reduction."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply bilateral filter to reduce noise while preserving edges
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)  
    
    image = cv2.resize(image, image_size)
    return image

def categorize_by_color(image, n_colors=3):
    """Categorize images by dominant color using KMeans and HSV clustering."""
    # Extract color features
    color_features = extract_color_features(image, n_colors)

    # Convert to HSV for better color distinction
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[..., 0]

    # Define color thresholds for classification (Example: Red, Green, Blue based on hue range)
    if np.mean(hue) < 30:  # Reddish
        return 'reddish', color_features
    elif 30 <= np.mean(hue) < 90:  # Greenish
        return 'greenish', color_features
    else:  # Bluish
        return 'bluish', color_features

def visualize_color_groups_hsv(color_groups):
    """Visualize the grouped colors for each class."""
    group_colors = {
        'reddish': (1, 0, 0),   # Red
        'greenish': (0, 1, 0),  # Green
        'bluish': (0, 0, 1)     # Blue
    }

    fig, axes = plt.subplots(len(color_groups), 1, figsize=(8, 2 * len(color_groups)))
    if len(color_groups) == 1:
        axes = [axes]

    for ax, (group_name, group_colors_data) in zip(axes, color_groups.items()):
        avg_color = np.mean(group_colors_data, axis=0)
        ax.imshow([avg_color / 255.0])
        ax.set_title(group_name)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

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
                color_group, color_features = categorize_by_color(image)
                
                hog_features = extract_hog_features(image)
                combined_features = np.concatenate([color_features, hog_features])
                
                color_groups[color_group].append(combined_features)
                grouped_images[color_group].append(combined_features)
                y_labels[color_group].append(class_name)
                
                X.append(combined_features)
                y.append(f"{class_name}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Visualize the average color in each group
visualize_color_groups_hsv(color_groups)

# Apply PCA for dimensionality reduction on the entire dataset
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# Hyperparameter tuning
def train_and_evaluate_with_gridsearch(grouped_images, y_labels, label_encoder):
    results = {}
    param_grids = {
        "Random Forest": {
            "max_depth": [10, 20, 30],
            "n_estimators": [100, 200, 300],
        },
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf"],
        },
        "k-NN": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        }
    }
    
    for color_group in grouped_images:
        print(f"\nHyperparameter tuning for {color_group} group...")
        
        X_group = np.array(grouped_images[color_group])
        y_group = np.array(y_labels[color_group])
        y_group_encoded = label_encoder.fit_transform(y_group)
        
        X_train, X_test, y_train, y_test = train_test_split(X_group, y_group_encoded, test_size=0.2, random_state=42)
        
        classifiers = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(random_state=42),
            "k-NN": KNeighborsClassifier(),
        }

        results[color_group] = {}
        
        for name, clf in classifiers.items():
            print(f"Tuning {name}...")
            
            grid_search = GridSearchCV(clf, param_grids[name], scoring="accuracy", cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_clf = grid_search.best_estimator_
            y_pred = best_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[color_group][name] = {
                "best_params": grid_search.best_params_,
                "classification_report": classification_report(y_test, y_pred, target_names=label_encoder.classes_),
                "accuracy": accuracy
            }
    
    return results

# Train and evaluate classifiers with hyperparameter tuning
results_with_gridsearch = train_and_evaluate_with_gridsearch(grouped_images, y_labels, label_encoder)

# Print results for each group
for color_group, classifiers_report in results_with_gridsearch.items():
    print(f"\nResults for {color_group} group with hyperparameter tuning:")
    for classifier_name, report in classifiers_report.items():
        print(f"\n{classifier_name} Best Parameters: {report['best_params']}")
        print(f"{classifier_name} Classification Report:")
        print(report["classification_report"])
        print(f"{classifier_name} Accuracy: {report['accuracy']:.4f}")

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
