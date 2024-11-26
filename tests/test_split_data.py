import unittest
import pandas as pd
from unredactor import split_data

class TestSplitDataOnlyValidation(unittest.TestCase):

    def test_split_data_only_validation(self):
        data = pd.DataFrame({
            'split': ['validation', 'validation'],
            'name': ['Alice', 'Bob'],
            'context': ['Context 3', 'Context 4'],
            'processed_context': ['Processed 3', 'Processed 4']
        })
        train_data, val_data = split_data(data)
        self.assertEqual(len(train_data), 0)
        self.assertEqual(len(val_data), 2)
        self.assertTrue(all(val_data['split'] == 'validation'))

if __name__ == '__main__':
    unittest.main()
