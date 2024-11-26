import unittest
from unredactor import preprocess_context

class TestPreprocessContextBasic(unittest.TestCase):

    def test_basic_redaction(self):
        context = "Hello, my name is ████ and I live in ████."
        expected_output = "Hello, my name is <redacted> and I live in <redacted>."
        self.assertEqual(preprocess_context(context), expected_output)

    def test_multiple_redactions(self):
        context = "████ went to the ████ to buy ████."
        expected_output = "<redacted> went to the <redacted> to buy <redacted>."
        self.assertEqual(preprocess_context(context), expected_output)

    def test_no_redactions(self):
        context = "Hello, my name is John and I live in London."
        expected_output = "Hello, my name is John and I live in London."
        self.assertEqual(preprocess_context(context), expected_output)

if __name__ == '__main__':
    unittest.main()
