# cis6930fa24-project2

Name: Ishtmeet Singh Arora

## Project Description
The "Unredactor" project aims to develop a machine learning-based solution for identifying redacted names in textual data. In many cases, sensitive information, such as names, is redacted (e.g., replaced with blacked-out characters like "████") in textual documents. The goal of this project is to "unredact" these names by leveraging the contextual information surrounding the redaction. This is achieved through a multi-step pipeline involving text preprocessing, feature extraction, model training, and evaluation.

The project involves:

Text Preprocessing: Redacted blocks in the text are identified and replaced with the placeholder <redacted>.

Balanced Training Data: The training dataset is balanced through undersampling to handle class imbalance, ensuring the model learns fairly across all classes

Context Vectorization: Contexts are transformed into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) with n-gram support, capturing word patterns and relationships.

Model Training: A logistic regression model is trained to predict the names from the processed contexts. The model is chosen for its efficiency and ability to handle large feature spaces.

Evaluation: The model's performance is evaluated using precision, recall, and F1-score metrics. A "top-k" evaluation strategy is used to assess the accuracy of the model's predictions.

Test Predictions: The trained model predicts names for unseen test data, and the predictions are saved in a submission file.

## How to install

```
pipenv install -e .
```

## How to run
Run the following command on terminal

``` bash
pipenv run python unredactor.py
```

## Video Overview

https://github.com/user-attachments/assets/34f1a6fa-f256-44a7-aa5d-9e6652887182



## unredactor.py overview ( Pipeline )

### Overview

The task is to predict names that have been redacted in text (represented by sequences of █ characters). The solution includes:

-> Preprocessing the input data.

-> Balancing the training dataset.

-> Vectorizing the text data into numerical features.

-> Training a logistic regression model to predict names based on the surrounding context.

-> Evaluating the model's performance.

-> Generating predictions for a test dataset.

## Preprocessing

### preprocess_context(context) 

    Purpose: Replace sequences of █ characters (redacted text) in the input context with the 
    placeholder <redacted>.

    Implementation: return re.sub(r'█+', '<redacted>', str(context))

    Regex Pattern r'█+': Matches one or more consecutive █ characters.

    Replacement: These matched sequences are replaced with <redacted>.


### read_and_preprocess_data()

    Purpose: Reads the unredactor.tsv dataset and preprocesses the text contexts.

    Steps:
        Reads the data using pandas.read_csv with custom parsing parameters.

        Applies preprocess_context to replace redacted blocks in the context column.

        Drops rows where any of processed_context, name, or split is missing.

    Result: A cleaned and preprocessed DataFrame.


## Data Splitting

### split_data(data) 

    Purpose: Splits the input data into two subsets:

    Training Data: Used to train the model (split == 'training').

    Validation Data: Used to evaluate the model (split == 'validation').
    
    Implementation: Uses pandas filtering to create separate DataFrames for training 
    and validation.


## Balancing the Training Data

### balance_training_data(train_data)

    Purpose: Ensures the training data is balanced across all classes (names). Balancing avoids 
    bias in the model towards more frequent names.

    Steps:
        Count Occurrences: Counts the number of samples for each name.

        Undersampling: For each name, keeps only min_count samples (the smallest class size) by 
        random sampling.

        Shuffle: Shuffles the balanced dataset to avoid ordering bias.


## Label Encoding

### encode_labels(balanced_train_data, val_data)

    Purpose:
        Converts names (categorical labels) into numerical labels using LabelEncoder.

        Ensures that the validation dataset only contains names present in the training 
        dataset.

    Steps:
        Fits a LabelEncoder on the training names to encode them as integers.

        Filters the validation data to include only known names.

        Transforms the validation names using the same encoder.


## Feature Vectorization

### vectorize_texts(train_texts, val_texts)

    Purpose: Converts the processed context text into numerical feature vectors using 
    the TF-IDF (Term Frequency-Inverse Document Frequency) method.

    Key Parameters:
        ngram_range=(1, 2): Captures single words (unigrams) and consecutive 
        word pairs (bigrams).

        max_features=50000: Limits the number of features to 50,000.

        min_df=1: A term must appear in at least one document to be included.

        max_df=1.0: A term can appear in all documents.


## Model Training

### train_classifier(train_vectors, train_labels)

    Purpose: Trains a Logistic Regression model on the training data.

    Model Parameters:
        penalty='l2': L2 regularization prevents overfitting by penalizing 
        large weights.

        solver='saga': Optimizer suitable for large datasets and sparse data.

        max_iter=1000: Allows up to 1000 iterations to converge.


## Model Evaluation

### evaluate_model(clf, val_vectors, val_labels, label_encoder)

    Purpose: Evaluates the model's performance on the validation set.

    Steps:
        Top-K Predictions: For each validation sample, retrieves the top-5 most probable 
        predictions.

        Binarized Labels: Converts both true labels and predictions into binary vectors 
        for multi-class evaluation.

        Metrics Computation: Uses precision_recall_fscore_support with average='micro' 
        to compute precision, recall, and F1-score.


## Test Data Handling

### read_and_preprocess_test_data()

    Purpose: Reads the test dataset from test.tsv and preprocesses the context 
    column using preprocess_context.

    Result: Returns a DataFrame with processed test data.


### create_submission_file(clf, vectorizer, test_data, label_encoder)

    Purpose: Uses the trained model to predict names for the test dataset.
    Saves the predictions to a submission.tsv file.

    Steps:
        Transforms the test data using the same TF-IDF vectorizer.

        Predicts the most likely name for each test context.

        Writes the predictions (test IDs and names) to a TSV file.


## Orchestration

### unredactor()

    Purpose: Orchestrates the entire pipeline by calling all helper functions in 
    sequence.

    Steps:
        Reads and preprocesses the training data.

        Splits the data into training and validation sets.

        Balances the training data.

        Encodes the labels.

        Vectorizes the text data.

        Trains a Logistic Regression classifier.

        Evaluates the model on the validation set.

        Reads and preprocesses the test data.

        Generates predictions for the test set and saves them in a submission.tsv file.


## Examples of Usage

-> Predict redacted names from the text context in declassified intelligence documents.

-> Predict names from redacted sections to aid investigations.

-> Predict patient names from context when approved by institutional policies.

## Bugs and Assumptions

### Assumptions

-> Works on unseen test data by leveraging context and uses sparse matrices and efficient algorithms for large datasets.

-> Logistic Regression works efficiently with sparse data and large feature spaces. It uses linear functions that are computationally less expensive compared to tree-based methods like Random Forest.

-> Logistic Regression models linear decision boundaries, which are often sufficient for text classification tasks. It fits a weighted sum of features to predict class probabilities.

-> The L2 regularization used in Logistic Regression is sufficient to prevent overfitting.

-> TF-IDF provides sufficient feature representation for the classification task.


## Test Cases

### test_preprocess_context_basic.py

test_basic_redaction: Tests the function's ability to replace a single redacted block (█) 
with <redacted> in a typical sentence containing multiple redactions.

test_multiple_redactions: Validates the function's ability to handle sentences with multiple 
redacted blocks of varying lengths.

test_no_redactions: Ensures the function does not alter input text when no redacted blocks are present.

### test_read_and_preprocess_data_basic.py

test_read_and_preprocess_data: Verifies that the function Reads a properly formatted TSV file, Applies the preprocess_context function to the context column, Returns a DataFrame with all required columns and preprocessed content.

### test_split_data_only_validation.py

test_split_data_only_validation: Verifies that the function correctly handles a dataset containing only validation samples.

























