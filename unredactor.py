import pandas as pd
import numpy as np
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

def unredactor():
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

    # Preprocess the context: replace redaction blocks with '<redacted>'
    def preprocess_context(context):
        # Replace sequences of █ characters with '<redacted>'
        return re.sub(r'█+', '<redacted>', str(context))

    data['processed_context'] = data['context'].apply(preprocess_context)

    # Drop any rows with missing values after preprocessing
    data.dropna(subset=['processed_context', 'name', 'split'], inplace=True)

    # Split the data into training and validation sets
    train_data = data[data['split'] == 'training']
    val_data = data[data['split'] == 'validation']

    # Analyze the distribution of names in the training data
    name_counts = train_data['name'].value_counts()
    print("Name distribution in training data:")
    print(name_counts)

    # Balance the training dataset using undersampling
    min_count = name_counts.min()
    balanced_train_data = pd.DataFrame()

    for name in name_counts.index:
        name_data = train_data[train_data['name'] == name]
        if len(name_data) > min_count:
            name_data = name_data.sample(min_count, random_state=42)
        balanced_train_data = pd.concat([balanced_train_data, name_data])

    # Shuffle the balanced training data
    balanced_train_data = balanced_train_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Encode the names
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(balanced_train_data['name'])

    # Filter validation data to only include names present in training data
    val_data = val_data[val_data['name'].isin(label_encoder.classes_)]
    val_labels = label_encoder.transform(val_data['name'])

    # Vectorize the contexts using TF-IDF with adjusted parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        min_df=1,
        max_df=1.0
    )

    # Fit the vectorizer on the balanced training data
    train_vectors = vectorizer.fit_transform(balanced_train_data['processed_context'])
    val_vectors = vectorizer.transform(val_data['processed_context'])
    # Reduce dimensionality
    # svd = TruncatedSVD(n_components=1000, random_state=42)
    # train_vectors_reduced = svd.fit_transform(train_vectors)
    # val_vectors_reduced = svd.transform(val_vectors)

    clf = LogisticRegression(
    penalty='l2',
    solver='saga', 
    max_iter=1000,
    random_state=42
    )
    clf.fit(train_vectors, train_labels)
    # # Train the Multinomial Naive Bayes classifier
    # clf = MultinomialNB()
    # clf.fit(train_vectors, train_labels)

    # Predict on validation data
    val_preds = clf.predict(val_vectors)
    val_probs = clf.predict_proba(val_vectors)

    labels = label_encoder.transform(label_encoder.classes_)

    # Evaluate the predictions
    top_k = 5
    top_k_preds = np.argsort(val_probs, axis=1)[:, -top_k:][:, ::-1]

    # Convert class indices to names
    predicted_names_val = []
    for preds in top_k_preds:
        names = label_encoder.inverse_transform(preds)
        predicted_names_val.append(names.tolist())

    # For evaluation
    true_names = val_data['name'].tolist()
    y_true = [[name] for name in true_names]
    y_pred = predicted_names_val

    # Get a list of all unique names
    all_names = list(label_encoder.classes_)
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    def binarize(names_list):
        binarized = np.zeros(len(all_names))
        for name in names_list:
            if name in name_to_idx:
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

    # Read the test data
    test_data = pd.read_csv(
        'test.tsv',
        sep='\t',
        engine='python',
        quoting=csv.QUOTE_NONE,
        encoding='utf-8',
        names=['id', 'context'],
    )

    # Preprocess the test context
    test_data['processed_context'] = test_data['context'].apply(preprocess_context)

    # Transform the test data using the fitted vectorizer
    test_vectors = vectorizer.transform(test_data['processed_context'])

    # Predict probabilities for the test data
    test_probs = clf.predict_proba(test_vectors)

    # Get top prediction
    top_preds = np.argmax(test_probs, axis=1)

    # Convert class indices to names
    predicted_names = label_encoder.inverse_transform(top_preds)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_data['id'],
        'name': predicted_names
    })

    # Save to TSV file
    submission.to_csv('submission.tsv', sep='\t', index=False)
    print("\nSubmission file 'submission.tsv' has been created.")

if __name__ == "__main__":
    unredactor()
