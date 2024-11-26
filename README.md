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

## unredactor.py overview ( Pipeline )

### Overview

The task is to predict names that have been redacted in text (represented by sequences of █ characters). The solution includes:

-> Preprocessing the input data.

-> Balancing the training dataset.

-> Vectorizing the text data into numerical features.

-> Training a logistic regression model to predict names based on the surrounding context.

-> Evaluating the model's performance.

-> Generating predictions for a test dataset.

### Preprocessing ( preprocess_context(context) )

```
    Purpose: Replace sequences of █ characters (redacted text) in the input context with the placeholder <redacted>.

    Implementation:
        return re.sub(r'█+', '<redacted>', str(context))

    Regex Pattern r'█+': Matches one or more consecutive █ characters.
    Replacement: These matched sequences are replaced with <redacted>.

```


### initialize_spacy_nlp()

```
def initialize_spacy_nlp():

    Initializes a SpaCy NLP pipeline with custom entity recognition patterns for redacting names, dates, phone numbers, and addresses. Uses lazy loading to load the model only once.
        
    Returns:
        A SpaCy NLP pipeline configured with custom patterns.

```

### initialize_hf_pipeline()

```
def initialize_hf_pipeline():

    Initializes a Hugging Face NER pipeline using the dslim/bert-base-NER model. This pipeline is used for entity detection in the redaction process.
        
    Returns:
        A Hugging Face pipeline object for Named Entity Recognition (NER).

```
### merge_overlapping_spans(spans)

```
def merge_overlapping_spans(spans):

    Merges overlapping or adjacent character spans to ensure there are no redundant redactions in overlapping areas.
        
    Args:
        spans (list of tuples): List of character index ranges to merge.

    Returns:
        A list of merged spans.
        
```
### identify_concept_sentences(text, concepts)

```
def identify_concept_sentences(text, concepts):

    Identifies sentences containing specified concepts (e.g., "confidential") for redaction.

    Args:
        text (str): The text to analyze.

        concepts (list of str): List of keywords or phrases to identify.
        
    Returns:
        List of character index ranges for sentences containing any specified concepts.

```

### redact_entities_hf(text, targets, stats)

```
def redact_entities_spacy(text, targets, stats):

    Uses Hugging Face’s NER pipeline to identify and redact specified entities.

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entities to redact (e.g., ['names', 'addresses']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### redact_entities_spacy(text, targets, stats)

```
def redact_entities_spacy(text, targets, stats):

    Uses the SpaCy pipeline to identify and redact specific entities in the text based on target categories (e.g., names, dates).

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entities to redact (e.g., ['names', 'dates']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### redact_email_headers(text, targets, stats)

```
def redact_email_headers(text, targets, stats):

    Redacts names found in email headers (such as 'From', 'To', 'Cc') within the text.

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entity types to redact (e.g., ['names']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### redact_entities_regex(text, targets, stats)

```
def redact_entities_regex(text, targets, stats):

    Uses regular expressions to identify and redact specified entities, such as phone numbers, dates, and names, based on pattern matching.

    Args:
        text (str): Text to be redacted.

        targets (list of str): List of entities to redact (e.g., ['phones', 'dates']).

        stats (dict): Dictionary tracking redaction counts.
        
    Returns:
        List of character index ranges to redact.

```

### write_stats(stats, destination)

```
def write_stats(stats, destination):

    Outputs the redaction statistics to the specified destination (stderr, stdout, or a file path).

    Args:
        stats (dict): Dictionary containing counts of redacted items by category.

        destination (str): Output destination for statistics.

    This function accumulates counts for all redacted items across multiple files. If a redaction span is identified multiple times by different methods (e.g., SpaCy, Hugging Face, and regex)

```

### process_file(file_path, args, stats)

```
def process_file(file_path, args, stats):

    Processes a single text file, applying redaction based on specified arguments, and saves the redacted content.

    Args:
        file_path (str): Path of the input file.

        args (Namespace): Parsed command-line arguments with options for redaction.

        stats (dict): Dictionary to accumulate redaction statistics.

```

## Bugs and Assumptions

### Assumptions

-> The redact_entities_regex function may not always recognize names accurately, especially in cases of uncommon names or names with special characters. It relies on capitalized words, which could lead to false positives (e.g., capitalized words in sentences being treated as names).

-> The hardcoded patterns for dates, phone numbers, and addresses may not cover all possible formats encountered in real-world data. For instance, international phone number formats are not fully supported.

-> The write_stats function directly increments redaction counts without checking for duplicates. This might cause inaccurate counts, especially if an entity is identified multiple times by different models.


## Test Cases

### test_address.py

test_redact_address_spacy: Tests the redact_entities_spacy function by providing a sample address. Asserts that no address is detected to check if the function works correctly with a non-matching target.

test_redact_address_regex: Tests the redact_entities_regex function by providing an address in text format. Asserts that one address is detected using regex.


### test_concepts.py

test_identify_concept_sentences: Tests the identify_concept_sentences function by providing text with specific concepts. Asserts that sentences containing specified concepts are correctly identified.


### test_dates.py

test_redact_dates_spacy: Tests the redact_entities_spacy function to detect dates in the text. Asserts that one date is detected using SpaCy.

test_redact_dates_regex: Tests the redact_entities_regex function to detect multiple dates in different formats. Asserts that two dates are correctly identified using regex.


### test_identify_concepts.py

test_no_concepts: Verifies that no concepts are detected if the text does not contain specified keywords.

test_single_concept: Checks if a single concept is identified and redacted correctly in the text.

test_multiple_concepts: Tests multiple concept detection within the text and ensures the correct spans are returned.

test_concepts_case_insensitive: Verifies that the function is case-insensitive by detecting concepts written in different cases.

test_concepts_partial_match: Ensures that only exact matches are detected, ignoring partial matches within words.


### test_main.py

test_main_success: Mocks file processing to ensure main function correctly processes multiple files with specified options and calls the appropriate functions.

test_main_no_files_matched: Verifies that an error message is displayed when no files match the input pattern.

test_main_with_single_file: Tests the main function with a single file and verifies that processing functions are called correctly.


### test_merge_spans.py

test_no_overlaps: Tests that non-overlapping spans are returned as-is without modification.

test_with_overlaps: Ensures overlapping spans are merged into a single span.

test_adjacent_spans: Verifies that adjacent spans are merged into one span.

test_nested_spans: Checks if nested spans are merged into a single span correctly.

test_empty_spans: Confirms that an empty list of spans returns an empty result without errors.


### test_names.py

test_redact_person_names_spacy: Tests if SpaCy detects and redacts a person’s name in the text.

test_redact_person_names_hf: Verifies that Hugging Face NER detects and redacts a person’s name in the text.

test_redact_person_names_regex: Tests if regex successfully identifies and redacts multiple person names in the text.


### test_phones.py

test_redact_phone_numbers_spacy: Checks if SpaCy correctly detects and redacts a phone number in the text.

test_redact_phone_numbers_regex: Verifies that regex identifies multiple phone numbers in various formats and redacts them.


### test_process_file.py

test_process_file_success: Tests process_file with mocked functions to verify that redactions are applied correctly, the output file is written, and statistics are updated accurately.

test_process_file_read_error: Verifies that an error message is output to stderr when there is an issue reading the input file.


### test_redact_email_headers.py

test_redact_names_in_headers: Tests redact_email_headers to confirm that names in email headers are correctly detected and redacted, and the statistics are updated accurately.


### test_redact_entities_regex.py

test_no_matches: Checks that no redactions are applied if the text does not contain any of the specified entities.

test_overlapping_matches: Ensures that overlapping entities (e.g., name and phone number) are handled correctly by the regex redaction method, and statistics are updated accurately.


### test_write_stats.py

test_write_stats_to_file: Tests writing redaction statistics to a file by verifying the file content.

test_write_stats_to_stdout: Checks if redaction statistics are correctly output to stdout.

test_write_stats_to_stderr: Ensures redaction statistics are written to stderr when specified.

test_write_stats_to_file_failure: Verifies that an error message is output to stderr if there is a failure in writing statistics to a file.
























