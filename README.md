# Azerbaijani Spelling Correction Model

This repository contains a Transformer-based neural model for correcting spelling errors in Azerbaijani text. The model is designed to handle four different types of errors: transliteration, keyboard mistakes, character insertions, and character deletions.

## Features

- Character-level Transformer architecture for precise spelling correction
- Handles four distinct error types:
  - Transliteration errors (53.7% of dataset)
  - Keyboard errors (35.7% of dataset)
  - Insertion errors (7.7% of dataset)
  - Deletion errors (2.9% of dataset)
- Specialized fine-tuning for deletion errors
- High accuracy (F1 score of 0.9689)
- Fast inference on CPU (~75ms per word)

## Installation

```bash
# Clone the repository
git clone https://github.com/LocalDoc-Azerbaijan/azerbaijani_spell_correction_lstm_model.git
cd azerbaijani_spell_correction_lstm_model

# Install dependencies
pip install -r requirements.txt

# Download the model checkpoint (if not included in the repository)
# [Instructions for downloading model checkpoints if hosted separately]
```

## Usage

### As a Python Module

```python
from spell_corrector import AzerbaijaniSpellCorrector

# Initialize the model
corrector = AzerbaijaniSpellCorrector(checkpoint_path='checkpoints_model/best_model.pt')

# Correct a single word
corrected_word = corrector.correct('qadinlar')
print(f"Corrected: {corrected_word}")  # Output: qadınlar

# Correct multiple words
words = ['qadinlar', 'gelmek', 'ucun', 'insn']
corrected_words = corrector.correct_batch(words)
print(corrected_words)  # Output: ['qadınlar', 'gəlmək', 'üçün', 'insan']
```

### As a Command-Line Tool

```bash
python spell_corrector.py
```

This will start an interactive session where you can enter words to correct:

```
===== Spelling Correction Tool =====
Enter words to correct (type 'exit' or 'quit' to end):

Enter misspelled word: qadinlar
Original:  qadinlar
Corrected: qadınlar
Execution time: 104.47 ms
Changes made: qadinlar → qadınlar
```

## Model Architecture

The spelling correction system employs a character-level Transformer architecture:

- **Embeddings**: 512-dimensional character embeddings
- **Encoder**: 8 layers with 16 attention heads
- **Decoder**: 8 layers with 16 attention heads
- **Feed-forward dimension**: 2048
- **Dropout**: 0.1
- **Training with weighted loss function**: Higher weights for underrepresented error types

## Results

The model achieves the following performance metrics:

- **Overall**:
  - Precision: 0.9689
  - Recall: 0.9689
  - F1 Score: 0.9689
  - Character-level accuracy: 0.9939
  - Average Levenshtein distance: 0.05

- **By error type**:
  - KEYBOARD: 0.9868 (24025/24346)
  - TRANSLITERATION: 0.9728 (35664/36661)
  - INSERTION: 0.9395 (4935/5253)
  - DELETION: 0.7776 (1550/1992) after fine-tuning

## Dataset

The model was trained on a dataset of 682,513 word pairs, each containing a misspelled word and its correction. Each example is annotated with:
- Error type (TRANSLITERATION, KEYBOARD, INSERTION, or DELETION)
- Character-level operations (correct, substitution, deletion, or insertion)

The dataset has the following distribution:
- TRANSLITERATION: 366,603 examples (53.7%)
- KEYBOARD: 243,454 examples (35.7%)
- INSERTION: 52,528 examples (7.7%)
- DELETION: 19,928 examples (2.9%)

## Examples

| Misspelled | Corrected | Error Type | Execution Time |
|------------|-----------|------------|----------------|
| qadinlar   | qadınlar  | TRANSLITERATION | 104.47 ms |
| gelmek     | gəlmək    | TRANSLITERATION | 72.66 ms  |
| ucun       | üçün      | TRANSLITERATION | 53.90 ms  |
| koynek     | köynək    | TRANSLITERATION | 78.01 ms  |
| qacmaq     | qaçmaq    | TRANSLITERATION | 76.99 ms  |
| insn       | insan     | DELETION        | 64.27 ms  |
| mesuliyet  | məsuliyyət| TRANSLITERATION | 124.59 ms |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 LocalDoc-Azerbaijan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
