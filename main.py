"""
Main script for running soil fertility classification and clustering analysis
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_dataset, train_test_split
from src.preprocessing import preprocess_dataset
from src.classifiers import DtClassifier
from src.clustering import K_MEANS, silhouette_score_custom
from src.metrics import classification_metrics


def run_classification_pipeline(dataset_path='datasets/Dataset1.csv'):
    """
    Run complete classification pipeline
    """
    print("=" * 80)
    print("SOIL FERTILITY CLASSIFICATION PIPELINE")
    print("=" * 80)
    
    print("\n1. Loading dataset...")
    dataset, header = load_dataset(dataset_path)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {header}")
    
    print("\n2. Preprocessing...")
    dataset = preprocess_dataset(
        dataset,
        missing_method=1,
        outlier_method=0,
        correlation_threshold=0.75,
        normalize_method=1
    )
    print(f"Preprocessed shape: {dataset.shape}")
    
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[:, :-1], 
        dataset[:, -1], 
        test_size=0.2, 
        random_state=42
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    print("\n4. Training Decision Tree Classifier...")
    dt_clf = DtClassifier(min_samples_split=2, max_depth=5, info_gain_method="Gini")
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    print("\n--- Decision Tree Results ---")
    classification_metrics(y_test, y_pred_dt)
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION PIPELINE COMPLETED")
    print("=" * 80)
    
    return dataset


def run_clustering_pipeline(dataset_path='datasets/Dataset2.csv'):
    """
    Run complete clustering pipeline
    """
    print("\n" + "=" * 80)
    print("SOIL FERTILITY CLUSTERING PIPELINE")
    print("=" * 80)
    
    print("\n1. Loading dataset...")
    dataset, header = load_dataset(dataset_path)
    print(f"Dataset shape: {dataset.shape}")
    
    print("\n2. Preprocessing...")
    dataset = preprocess_dataset(
        dataset,
        missing_method=1,
        outlier_method=0,
        correlation_threshold=0.75,
        normalize_method=1
    )
    print(f"Preprocessed shape: {dataset.shape}")
    
    X = dataset[:, :-1]
    
    print("\n3. Running K-Means Clustering (k=3)...")
    kmeans = K_MEANS(k=3, distance_method=1, centroid_method=1, max_iterations=10000)
    kmeans.fit(X)
    labeled_data = kmeans.cluster()
    labels = kmeans.get_labels()
    
    print(f"Number of clusters: {len(np.unique(labels))}")
    for i in range(int(max(labels)) + 1):
        cluster_size = np.sum(labels == i)
        print(f"Cluster {i}: {cluster_size} samples")
    
    silhouette_avg, intra_dist, inter_dist = silhouette_score_custom(X, labels, metric=0)
    print(f"\nSilhouette Score: {silhouette_avg:.4f}")
    print(f"Intra-cluster Distance: {intra_dist:.4f}")
    print(f"Inter-cluster Distance: {inter_dist:.4f}")
    
    print("\n" + "=" * 80)
    print("CLUSTERING PIPELINE COMPLETED")
    print("=" * 80)


def main():
    """
    Main function to run all pipelines
    """
    print("\n" + "=" * 80)
    print("DATA MINING PROJECT: SOIL FERTILITY ANALYSIS")
    print("=" * 80)
    
    try:
        dataset = run_classification_pipeline()
        run_clustering_pipeline()
        
        print("\n\n" + "=" * 80)
        print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()