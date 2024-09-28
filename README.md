# AI Metrics

AI Metrics is a Python library for basic Natural Language Processing (NLP) metric score implementations. This library provides various metrics commonly used in evaluating NLP models, such as accuracy, BERT score, BLEU score, and ROUGE score.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Metrics](#metrics)
    - [Accuracy Score](#accuracy-score)
    - [BERT Score](#bert-score)
    - [BLEU Score](#bleu-score)
    - [ROUGE Score](#rouge-score)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

You can install the library using pip. First, clone the repository:

```bash
git clone https://github.com/hari-hud/ai-metrics.git
cd ai-metrics
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

Finally, install the library in editable mode:

```bash
pip install -e .
```

## Usage

Here are some examples of how to use the different metrics provided by the library:

### Metrics

#### Accuracy Score

```python
from metrics.nlp.accuracy_score import AccuracyScore

ground_truths = ["label1", "label2", "label3"]
predictions = ["label1", "label2", "label3"]

accuracy = AccuracyScore()
score = accuracy.get_score(ground_truths, predictions)
print(score)  # Output: {'accuracy': 1.0}
```

#### BERT Score

```python
from metrics.nlp.bert_score import BertScore

ground_truths = ["The cat is on the mat."]
predictions = ["The cat is sitting on the mat."]

bert_score = BertScore()
score = bert_score.get_score(ground_truths, predictions)
print(score)  # Output: {'bert_f1': <float_value>}
```

#### BLEU Score

```python
from metrics.nlp.bleu_score import BLEUScore

ground_truths = ["The cat is on the mat."]
predictions = ["The cat is on the mat."]

bleu = BLEUScore()
score = bleu.get_score(ground_truths, predictions)
print(score)  # Output: {'bleu_score': <float_value>}
```

#### ROUGE Score

```python
from metrics.nlp.rouge_score import ROUGEScores

ground_truths = ["The cat sat on the mat."]
predictions = ["The cat sat on the mat."]

rouge = ROUGEScores()
score = rouge.get_score(ground_truths, predictions)
print(score)  # Output: {'rouge_1_score': <float_value>, 'rouge_2_score': <float_value>, ...}
```

## Running Tests

You can run the tests using pytest. To run all tests, execute:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1. Fork the repository
2. Create your feature branch (e.g., `git checkout -b feature/YourFeature`)
3. Commit your changes (e.g., `git commit -m 'Add some feature'`)
4. Push to the branch (e.g., `git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contact

For any questions or suggestions, please contact:

**Hari Hud**  
Email: [hudharibhau@gmail.com](mailto:hudharibhau@gmail.com)  
GitHub: [hari-hud](https://github.com/hari-hud)
