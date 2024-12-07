import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier  # Importing MLPClassifier for Neural Network
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Function to load the dataset from .txt files
def load_data(train_data_file, train_label_file, test_data_file):
    train_data = pd.read_csv(train_data_file, sep=r'[,\s;]+', header=None, engine='python')
    train_labels = pd.read_csv(train_label_file, sep=r'[,\s;]+', header=None, engine='python').values.ravel()
    test_data = pd.read_csv(test_data_file, sep=r'[,\s;]+', header=None, engine='python')
    return train_data, train_labels, test_data

# Imputation function (Mean/Median and KNN)
def impute_missing_values(train_data, test_data, method="mean"):
    train_data.replace(1.00000000000000e+99, np.nan, inplace=True)
    test_data.replace(1.00000000000000e+99, np.nan, inplace=True)

    if method == "mean" or method == "median":
        imputer = SimpleImputer(strategy=method)
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)

    imputed_train_data = imputer.fit_transform(train_data)
    imputed_test_data = imputer.transform(test_data)

    return imputed_train_data, imputed_test_data

# Train Classifiers and Evaluate with Cross-Validation
def train_classifier_with_cv(train_data, train_labels, model_type="random_forest", cv_folds=5):
    if model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "svm":
        classifier = SVC(kernel='linear', random_state=42)
    elif model_type == "neural_network":
        classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)  # Neural Network

    stratified_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, train_data, train_labels, cv=stratified_cv, scoring='accuracy')

    print(f"{model_type.capitalize()} - CV Accuracy Scores for {cv_folds} folds: {cv_scores}")
    print(f"{model_type.capitalize()} - Mean CV Accuracy: {np.mean(cv_scores):.4f}")
    
    classifier.fit(train_data, train_labels)
    return classifier

# Make Predictions and Save Results
def predict_and_save(classifier, test_data, output_file, probability_file=None):
    # Use predict() to get class labels (not probabilities)
    predictions = classifier.predict(test_data)
    np.savetxt(output_file, predictions, fmt='%d')  # Save predictions as integers (class labels)

    # If probability_file is provided, save the predicted probabilities as well
    if probability_file:
        probabilities = classifier.predict_proba(test_data)  # Predicted probabilities
        np.savetxt(probability_file, probabilities, fmt='%.6f')  # Save probabilities with more precision

# Main execution for each dataset
def handle_classification_task(train_data_file, train_label_file, test_data_file, result_file, missing_result_file, model_type="random_forest"):
    train_data, train_labels, test_data = load_data(train_data_file, train_label_file, test_data_file)
    
    if train_data is None or test_data is None:
        return  # Exit if files were not loaded successfully
    
    # Impute missing values (Mean/Median or KNN)
    imputed_train_data, imputed_test_data = impute_missing_values(train_data, test_data, method="knn")  # Using KNN for this example

    # Save the imputed data as the missing result file
    np.savetxt(missing_result_file, imputed_test_data, fmt='%.6f')

    # Train the classifier and make predictions
    classifier = train_classifier_with_cv(imputed_train_data, train_labels, model_type=model_type)
    predict_and_save(classifier, imputed_test_data, result_file)  # Save the class labels as result

# Run for each dataset (Example for Dataset 1, you can repeat for others)
if __name__ == '__main__':
    handle_classification_task(
        '48506850Project/TrainData1.txt', 
        '48506850Project/TrainLabel1.txt', 
        '48506850Project/TestData1.txt', 
        'output/SamanoResult1.txt', 
        'output/SamanoMissingResult1.txt',
        model_type="neural_network"
    )
    handle_classification_task(
        '48506850Project/TrainData2.txt', 
        '48506850Project/TrainLabel2.txt', 
        '48506850Project/TestData2.txt', 
        'output/SamanoResult2.txt', 
        'output/SamanoMissingResult2.txt',
        model_type="svm"
    )
    handle_classification_task(
        '48506850Project/TrainData3.txt', 
        '48506850Project/TrainLabel3.txt', 
        '48506850Project/TestData3.txt', 
        'output/SamanoResult3.txt', 
        'output/SamanoMissingResult3.txt',
        model_type="random_forest"
    )
    handle_classification_task(
        '48506850Project/TrainData4.txt', 
        '48506850Project/TrainLabel4.txt', 
        '48506850Project/TestData4.txt', 
        'output/SamanoResult4.txt', 
        'output/SamanoMissingResult4.txt',
        model_type="random_forest"
    )
    handle_classification_task(
        '48506850Project/TrainData5.txt', 
        '48506850Project/TrainLabel5.txt', 
        '48506850Project/TestData5.txt', 
        'output/SamanoResult5.txt', 
        'output/SamanoMissingResult5.txt',
        model_type="svm"
    )