# KNN-based parasite image classifier
#
# Approach: Use a simple KNN classifier on flattened image pixels. This works
# because parasite images within the same class share similar visual patterns
# (color, shape, texture). KNN finds the k most similar training images and
# uses majority voting to classify new samples.
#
# Pipeline:
# 1. Extract ROIs using bounding box annotations
# 2. Resize to uniform dimensions (required for consistent feature vectors)
# 3. Flatten pixels into 1D feature vectors
# 4. Normalize features (important: pixel values can vary widely)
# 5. Apply PCA to reduce dimensionality (speeds up KNN, reduces noise)
# 6. Train KNN with cross-validation to find optimal k

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_DIR = Path(__file__).parent / "data"
# 64x64 balances detail preservation with computational efficiency
# Larger sizes increase feature count (64*64*3 = 12288 features)
IMAGE_SIZE = (64, 64)
# Filter out rare classes - KNN needs sufficient neighbors per class
MIN_SAMPLES_PER_CLASS = 10


def load_dataset():
    """Load and parse the ParasitoBank JSON dataset."""
    with open(DATA_DIR / "ParasitoBank.json") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    categories_by_id = {cat["id"]: cat["name"] for cat in data["categories"]}

    return data["annotations"], images_by_id, categories_by_id


def crop_and_resize(image_path: Path, bbox: list[float], size: tuple[int, int]) -> np.ndarray:
    """Crop bounding box region from image and resize to uniform dimensions."""
    # Convert to RGB to ensure consistent 3-channel format (some images may be grayscale)
    img = Image.open(image_path).convert("RGB")

    # Extract the region of interest containing the parasite
    # bbox format: [x, y, width, height] - standard COCO annotation format
    x, y, w, h = bbox
    cropped = img.crop((x, y, x + w, y + h))
    # Resize to uniform dimensions because bounding boxes vary in size
    # KNN requires fixed-length feature vectors - all samples must have identical dimensions
    # LANCZOS provides high-quality resampling by using a windowed sinc filter
    resized = cropped.resize(size, Image.Resampling.LANCZOS)

    # Flatten 3D array (H, W, C) into 1D vector for sklearn compatibility
    return np.array(resized).flatten()


def save_cropped_images(annotations, images_by_id, categories_by_id):
    """Save cropped and resized images to processed folder for inspection."""
    processed_dir = DATA_DIR.parent / "processed"
    processed_dir.mkdir(exist_ok=True)

    for cat_name in categories_by_id.values():
        (processed_dir / cat_name).mkdir(exist_ok=True)

    saved = 0
    for ann in annotations:
        img_info = images_by_id[ann["image_id"]]
        img_path = DATA_DIR / "images" / img_info["file_name"]

        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            x, y, w, h = ann["bbox"]
            cropped = img.crop((x, y, x + w, y + h))
            resized = cropped.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            cat_name = categories_by_id[ann["category_id"]]
            out_path = processed_dir / cat_name / f"ann_{ann['id']}.jpg"
            resized.save(out_path)
            saved += 1
        except Exception as e:
            print(f"Error saving {img_path.name}: {e}")

    print(f"Saved {saved} cropped images to {processed_dir}")


def prepare_features(annotations, images_by_id, categories_by_id):
    """Load images, crop ROIs, and prepare feature vectors."""
    from collections import Counter

    # Count samples per class to filter out underrepresented categories
    # Classes with too few samples won't have enough neighbors for reliable KNN
    # and can cause issues with stratified train/test splitting
    category_counts = Counter(a["category_id"] for a in annotations)
    valid_categories = {
        cat_id for cat_id, count in category_counts.items()
        if count >= MIN_SAMPLES_PER_CLASS
    }

    excluded = [
        categories_by_id[cat_id]
        for cat_id in sorted(set(categories_by_id.keys()) - valid_categories)
    ]
    if excluded:
        print(f"Excluding categories with <{MIN_SAMPLES_PER_CLASS} samples: {excluded}")

    X, y = [], []
    skipped = 0

    for ann in annotations:
        if ann["category_id"] not in valid_categories:
            continue

        img_info = images_by_id[ann["image_id"]]
        img_path = DATA_DIR / "images" / img_info["file_name"]

        if not img_path.exists():
            skipped += 1
            continue

        try:
            features = crop_and_resize(img_path, ann["bbox"], IMAGE_SIZE)
            X.append(features)
            y.append(ann["category_id"])
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} annotations due to missing/invalid images")

    return np.array(X), np.array(y), valid_categories


def train_knn(X_train, y_train, n_components=100):
    """Train KNN classifier with PCA and grid search for optimal k."""
    # Standardize features to zero mean and unit variance
    # Critical for KNN: features with larger scales would dominate distance calculations
    # e.g., without scaling, a pixel value difference of 100 would overshadow all others
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA reduces dimensionality while preserving variance
    # Benefits: 1) Faster KNN (fewer dimensions to compute distances over)
    #           2) Noise reduction (discards low-variance components)
    #           3) Mitigates curse of dimensionality (KNN struggles in high-D spaces)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA: {n_components} components, {explained_var:.1f}% variance explained")

    # Grid search to find optimal k (number of neighbors)
    # Small k: sensitive to noise, may overfit
    # Large k: smoother boundaries, but may underfit and blur class distinctions
    # Odd values preferred to avoid ties in binary/multi-class voting
    param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier()
    # 5-fold CV: train on 4 folds, validate on 1, repeat 5 times for robust estimate
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_pca, y_train)

    print(f"Best k: {grid_search.best_params_['n_neighbors']}")
    print(f"Cross-validation accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, scaler, pca


def evaluate(model, scaler, pca, X_test, y_test, categories_by_id):
    """Evaluate model and print metrics."""
    # Apply same transformations as training (but don't refit - use learned params)
    X_scaled = scaler.transform(X_test)
    X_pca = pca.transform(X_scaled)
    y_pred = model.predict(X_pca)

    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    unique_labels = sorted(set(y_test) | set(y_pred))
    target_names = [categories_by_id[label] for label in unique_labels]

    # Classification report shows per-class precision, recall, F1
    # Precision: of predicted positives, how many are correct?
    # Recall: of actual positives, how many did we find?
    # F1: harmonic mean of precision and recall
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion matrix shows where misclassifications occur
    # Rows = actual class, Columns = predicted class
    # Useful for identifying which parasite species get confused with each other
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    print("Loading dataset...")
    annotations, images_by_id, categories_by_id = load_dataset()
    print(f"Found {len(annotations)} annotations across {len(images_by_id)} images")

    print("\nSaving cropped images for inspection...")
    save_cropped_images(annotations, images_by_id, categories_by_id)

    print("\nPreparing features...")
    X, y, valid_categories = prepare_features(annotations, images_by_id, categories_by_id)
    print(f"Prepared {len(X)} samples with {X.shape[1]} features each")

    # Stratified split ensures each class has proportional representation in train/test
    # This prevents issues where rare classes might end up entirely in one split
    # random_state=42 for reproducibility
    print("\nSplitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # n_components=30 is a reasonable starting point for 64x64 images
    # Lower than default 100 because we have limited training samples
    # Rule of thumb: n_components should be << n_samples to avoid overfitting
    print("\nTraining with optimal config...")
    model, scaler, pca = train_knn(X_train, y_train, n_components=30)

    print("\nEvaluating on test set...")
    evaluate(model, scaler, pca, X_test, y_test, categories_by_id)


if __name__ == "__main__":
    main()
