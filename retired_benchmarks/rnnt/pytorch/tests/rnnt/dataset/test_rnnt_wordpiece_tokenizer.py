import pytest

from rnnt.dataset import RNNTWordpieceTokenizer


@pytest.fixture
def vocab_file():
    return 'wordpieces/bert-base-1000.txt'


@pytest.fixture
def tokenizer(vocab_file):
    charset = "abcdefghijklmnopqrstuvwxyz '"
    return RNNTWordpieceTokenizer(vocab_file, charset)


def test_labels(vocab_file, tokenizer):
    with open(vocab_file) as f:
        labels = f.read().splitlines()

    assert tokenizer.labels == labels + ['<BLANK>']


def test_blank_idex(vocab_file, tokenizer):
    with open(vocab_file) as f:
        lines = sum(1 for line in f)

    assert tokenizer.blank_index == lines


def test_empty_input(tokenizer):
    assert tokenizer('') == []


def detokenize(tokens):
    output = ' '.join(tokens)
    output = output.replace(' ##', '')
    output = output.replace('##', '')
    return output.strip()


def test_expected_input(tokenizer):
    expected_input = 'expected input'
    tokens = tokenizer(expected_input)
    assert detokenize(tokens) == expected_input


def test_unexpected_input(tokenizer):
    unexpected_input = 'unexpected input!!!'
    tokens = tokenizer(unexpected_input)
    assert detokenize(tokens) == 'unexpected <BLANK>'

