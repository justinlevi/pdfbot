import pytest
import mock
from app import num_tokens_from_string, calculate_cost, process_pdf, run_qa_chain

# Test num_tokens_from_string
def test_num_tokens_from_string():
    # Given
    string = "This is a test string."

    # When
    result = num_tokens_from_string(string)

    # Then
    assert result == 6  # assuming each word is a token

# Test calculate_cost
def test_calculate_cost():
    # Given
    tokens = 1000

    # When
    result = calculate_cost(tokens)

    # Then
    assert result == 0.02  # based on the perTokenCost defined in your code

# Test process_pdf
@mock.patch('app.PdfReader')
def test_process_pdf(mock_pdf_reader):
    # Given
    mock_pdf_reader.return_value.pages = [
        mock.MagicMock(extract_text=mock.MagicMock(return_value="This is a test."))
    ]
    file_path = 'test.pdf'

    # When
    result = process_pdf(file_path)

    # Then
    assert result == "This is a test."