import pandas as pd
import numpy as np
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

def preprocess_context(context):
    """Replaces sequences of redaction blocks with '<redacted>'."""
    return re.sub(r'█+', '<redacted>', str(context))

def read_and_preprocess_data():
    """Reads the data and preprocesses the context."""
    # Read the data with proper parsing parameters
    data = pd.read_csv(
        'unredactor.tsv',
        sep='\t',
        engine='python',
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        encoding='utf-8',
        names=['split', 'name', 'context'],
    )

    # Preprocess the context
    data['processed_context'] = data['context'].apply(preprocess_context)

    # Drop any rows with missing values after preprocessing
    data.dropna(subset=['processed_context', 'name', 'split'], inplace=True)
    return data

def split_data(data):
    """Splits the data into training and validation sets."""
    train_data = data[data['split'] == 'training']
    val_data = data[data['split'] == 'validation']
    return train_data, val_data

def balance_training_data(train_data):
    """Balances the training dataset using undersampling."""
    name_counts = train_data['name'].value_counts()
    min_count = name_counts.min()
    balanced_train_data = pd.DataFrame()

    for name in name_counts.index:
        name_data = train_data[train_data['name'] == name]
        if len(name_data) > min_count:
            name_data = name_data.sample(min_count, random_state=42)
        balanced_train_data = pd.concat([balanced_train_data, name_data])

    # Shuffle the balanced training data
    balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_train_data

def encode_labels(balanced_train_data, val_data):
    """Encodes the names and filters validation data."""
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(balanced_train_data['name'])

    # Filter validation data to only include names present in training data
    val_data = val_data[val_data['name'].isin(label_encoder.classes_)]
    val_labels = label_encoder.transform(val_data['name'])
    return label_encoder, train_labels, val_data, val_labels

def vectorize_texts(train_texts, val_texts):
    """Vectorizes the contexts using TF-IDF."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=1,
        max_df=1.0
    )
    train_vectors = vectorizer.fit_transform(train_texts)
    val_vectors = vectorizer.transform(val_texts)
    return vectorizer, train_vectors, val_vectors

def train_classifier(train_vectors, train_labels):
    """Trains the Logistic Regression classifier."""
    clf = LogisticRegression(
        penalty='l2',
        solver='saga', 
        max_iter=1000,
        random_state=42
    )
    clf.fit(train_vectors, train_labels)
    return clf

def evaluate_model(clf, val_vectors, val_labels, label_encoder):
    """Evaluates the model and prints precision, recall, and F1-score."""
    val_probs = clf.predict_proba(val_vectors)

    # Evaluate the predictions
    top_k = 5
    top_k_preds = np.argsort(val_probs, axis=1)[:, -top_k:][:, ::-1]

    # Convert class indices to names
    predicted_names_val = [
        label_encoder.inverse_transform(preds).tolist() for preds in top_k_preds
    ]

    # For evaluation
    true_names = label_encoder.inverse_transform(val_labels)
    y_true = [[name] for name in true_names]
    y_pred = predicted_names_val

    # Get a list of all unique names
    all_names = list(label_encoder.classes_)
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    def binarize(names_list):
        binarized = np.zeros(len(all_names))
        for name in names_list:
            binarized[name_to_idx[name]] = 1
        return binarized

    y_true_binarized = np.array([binarize(names) for names in y_true])
    y_pred_binarized = np.array([binarize(names) for names in y_pred])

    # Compute precision, recall, and F1-score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true_binarized,
        y_pred_binarized,
        average='micro',
        zero_division=0
    )

    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1_score))

def read_and_preprocess_test_data():
    """Reads and preprocesses the test data."""
    test_data = pd.read_csv(
        'test.tsv',
        sep='\t',
        engine='python',
        quoting=csv.QUOTE_NONE,
        encoding='utf-8',
        names=['id', 'context'],
    )
    test_data['processed_context'] = test_data['context'].apply(preprocess_context)
    return test_data

def create_submission_file(clf, vectorizer, test_data, label_encoder):
    """Creates the submission file with predictions."""
    test_vectors = vectorizer.transform(test_data['processed_context'])
    test_probs = clf.predict_proba(test_vectors)
    top_preds = np.argmax(test_probs, axis=1)
    predicted_names = label_encoder.inverse_transform(top_preds)

    submission = pd.DataFrame({
        'id': test_data['id'],
        'name': predicted_names
    })

    submission.to_csv('submission.tsv', sep='\t', index=False)
    print("\nSubmission file 'submission.tsv' has been created.")

def unredactor():
    """Main function to run the unredaction process."""
    # Step 1: Read and preprocess data
    data = read_and_preprocess_data()

    # Step 2: Split data into training and validation sets
    train_data, val_data = split_data(data)

    # Step 3: Balance the training dataset
    balanced_train_data = balance_training_data(train_data)

    # Step 4: Encode labels
    label_encoder, train_labels, val_data, val_labels = encode_labels(balanced_train_data, val_data)

    # Step 5: Vectorize texts
    vectorizer, train_vectors, val_vectors = vectorize_texts(
        balanced_train_data['processed_context'],
        val_data['processed_context']
    )

    # Step 6: Train the classifier
    clf = train_classifier(train_vectors, train_labels)

    # Step 7: Evaluate the model
    evaluate_model(clf, val_vectors, val_labels, label_encoder)

    # Step 8: Read and preprocess test data
    test_data = read_and_preprocess_test_data()

    # Step 9: Create submission file
    create_submission_file(clf, vectorizer, test_data, label_encoder)

if __name__ == "__main__":
    unredactor()
