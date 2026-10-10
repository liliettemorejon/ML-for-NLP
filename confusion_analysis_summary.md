# Confusion analysis: Group 4 character model

**Conditions.** EMNIST ByClass test split, 36 classes (0-9, A-Z), 89,264 test characters the model never saw in training. Raw model reads before any business rule. Augmented model = `models/char_cnn_aug_best.pt` (used by the demo); baseline = `models/char_cnn_best.pt`. Files: `confusion_matrix_augmented.png`, `confusion_matrix_augmented.csv`, `confusion_matrix_baseline.csv`.

## Accuracy

| Model | All characters | Digits (57,918) | Letters (31,346) |
| --- | --- | --- | --- |
| Augmented | 91.03% | 94.98% | 83.72% |
| Baseline | 91.71% | 92.74% | 89.81% |

On clean EMNIST the two models are close overall. The augmented model is better on digits, which are what the mileage boxes hold, and weaker on letters. It was chosen because it was trained on rotated, scaled, sheared, shifted, thick, thin and blurred characters, which is closer to phone photos of real handwriting, not because it scores higher on clean EMNIST.

## Where the errors are

- **87% of the augmented model's 8,010 errors are digit versus letter look-alikes** (6,995 errors). Only 347 digits (0.6%) are read as a different digit.
- Largest confusions (true to read, augmented): **O read as 0** 2,716 times (65% of all O), **I read as 1** 1,005 (49% of all I), **2 read as Z** 534 (9%), **S read as 5** 414 (12%), **1 read as I** 397, **0 read as O** 385, **5 read as S** 261.
- Weakest characters (augmented): O 31% correct, I 49%, S 87%, 0 89%, 2 89%.
- Digit-to-digit errors are rare: 4 read as 9 (35 times), 1 read as 7 (20), 0 read as 6 or 8 (16 each), 9 read as 7 (15).
- The baseline shows the same pairs, with a different balance: O read as 0 is 37% and 0 read as O is 22% (augmented: 65% and 7%).
- **Why:** in EMNIST ByClass, O and 0, I and 1, S and 5, 2 and Z look alike when handwritten, so one image cannot always settle which one it is. This is mostly a limit of the data, not of the model.

## What this means for the mileage log reader

- Mileage and odometer boxes hold only digits, and ID and client boxes have a known type. The pipeline restricts each field to its own type, so most of these 6,995 digit versus letter errors cannot happen on the form. The notebook's field-restriction table gives the measured effect.
- The remaining digit-to-digit errors (4 to 9, 1 to 7, 0 to 6) are small on EMNIST (0 read as 6 is 16 of 5,778 zeros, 0.3%). On real phone photos 0 and 6 were confused more, and that is where the form's own rules help: end odometer minus start odometer must equal the miles, the total must equal the sum of the rows, and a low-confidence digit goes to review instead of being posted.
- Limit of this analysis: it uses clean EMNIST characters. It does not include segmentation errors (a box cut wrongly or empty) or photo-quality problems. Those are measured separately on the synthetic logs and the real forms.
