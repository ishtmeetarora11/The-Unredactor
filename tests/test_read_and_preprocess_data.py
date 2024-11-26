import unittest
import pandas as pd
import os
import tempfile
from unredactor import read_and_preprocess_data

class TestReadAndPreprocessDataBasic(unittest.TestCase):

    def setUp(self):
        # Create a temporary unredactor.tsv file with sample data
        self.sample_data = """training\tAlice\tHello, my name is ██████ and I love programming.
training\tBob\tHi there, ████ here. Nice to meet you.
validation\tAlice\tGreetings from ██████! How are you?
validation\tBob\tThis is a message from ████. See you soon."""
        self.temp_file = tempfile.NamedTemporaryFile('w+', delete=False, suffix='.tsv')
        self.temp_file_name = self.temp_file.name
        self.temp_file.write(self.sample_data)
        self.temp_file.flush()
        self.original_read_csv = pd.read_csv

        # Mock read_csv to read from the temporary file
        def mock_read_csv(*args, **kwargs):
            if 'unredactor.tsv' in args[0]:
                args = list(args)
                args[0] = self.temp_file_name
            return self.original_read_csv(*args, **kwargs)
        pd.read_csv = mock_read_csv

    def tearDown(self):
        # Restore original read_csv
        pd.read_csv = self.original_read_csv
        # Remove temporary file
        os.unlink(self.temp_file_name)

    def test_read_and_preprocess_data(self):
        data = read_and_preprocess_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 4)
        self.assertIn('processed_context', data.columns)
        expected_processed_context = [
            "Hello, my name is <redacted> and I love programming.",
            "Hi there, <redacted> here. Nice to meet you.",
            "Greetings from <redacted>! How are you?",
            "This is a message from <redacted>. See you soon."
        ]
        self.assertListEqual(data['processed_context'].tolist(), expected_processed_context)

if __name__ == '__main__':
    unittest.main()
