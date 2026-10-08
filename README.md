# Form ML-7 Mileage Log Reader

ISM 6642, Group 4. Reads a handwritten Weekly Field Mileage Log (Form ML-7) from a phone photo and decides, row by row,
whether to **pay** (`AUTO-POST`) or **send to a clerk** (`REVIEW`).

## Run the demo (one click, nothing to install)

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liliettemorejon/ML-for-NLP/blob/main/MNIST_Final_Group4_DEMO.ipynb)

1. Click the badge above (a Google account is the only requirement).
2. **Runtime > Change runtime type > CPU**. No GPU is needed to read a form.
3. **Runtime > Run all**. The first cell clones this repo, so there is nothing to download by hand.
4. In the last cell, click **Choose files** and pick one or several photos of a filled-in Form ML-7
   (JPG, PNG or iPhone HEIC). Each photo takes a few seconds.

For every photo the demo prints the values it read with a confidence for each row, the rules that failed, and the
decision for each row, followed by a picture of the form (green = OK, orange = fails a rule, red = low confidence,
blue = changed or confirmed by the rules).

## How a photo becomes a decision

1. **Registration:** match SIFT features against the blank form and straighten the photo onto it.
2. **Cell extraction:** erase the printed box lines and cut out each handwritten box.
3. **Normalization:** make each crop look like an EMNIST character (28 x 28, white on black, scaled to [-1, 1]).
4. **Classification:** a small CNN reads each box (36 classes: digits 0-9 and capitals A-Z); digit boxes may only
   hold digits and letter boxes only letters.
5. **Form checks:** the rules below.
6. **Decision per row:** `AUTO-POST` only if none of the row's checks fail, otherwise `REVIEW`.

### The form checks

- Employee ID is in the HR list.
- Week ending is a valid date, a Sunday, not in the future, and within the last 60 days.
- Each trip date falls inside the week ending.
- The client code is on the employee's visit schedule.
- Odometer end is after start, and start is not before the previous row's end.
- Miles equal odometer end minus start; a trip over 300 miles is flagged.
- Total miles equal the sum of the rows.
- Any character read with confidence below 0.50 is flagged.

A bad employee ID or week ending sends the whole log to a clerk. A wrong total is blamed on the rows already flagged,
or on the whole log if none are flagged.

For mileage digits only, a digit the model was unsure of can be kept or swapped for its next guess when the
odometer arithmetic confirms it (blue boxes). Dates and client codes are never changed. Set `AUTO_CORRECT = False`
in the demo's settings cell to turn this off.

## What is in this repo

| Path | What it is |
| --- | --- |
| `MNIST_Final_Group4_DEMO.ipynb` | The live demo: load the saved model, read photos, check the rules |
| `MNIST_Final_Group4_FULL_Colab.ipynb` | The full project: EMNIST download, training, synthetic test logs, evaluation, the real forms |
| `assets/` | Blank Form ML-7, the box positions, and the made-up HR and visit-schedule data |
| `models/` | Trained weights: `char_cnn_aug_best.pt` (used by the demo) and `char_cnn_best.pt` (baseline) |
| `samples/` | Photos of made-up, hand-filled forms to try the demo on |
| `requirements.txt` | Python packages, for running outside Colab |

## The full notebook (training and evaluation)

[![Open the full notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liliettemorejon/ML-for-NLP/blob/main/MNIST_Final_Group4_FULL_Colab.ipynb)

Use a **T4 GPU** runtime for training. In the settings cell, `RETRAIN = False` loads the saved weights from `models/`
and gives the same results as the demo. `RETRAIN = True` trains two new models inside that Colab session (a few minutes
each); the repo is not changed, and its numbers will differ slightly from the saved weights.

## Results of the saved weights

Measured on four forms filled in by hand with made-up data, photographed with an iPhone, one writer (36 rows, 856
characters). These are small-sample numbers.

| Measure | Result |
| --- | --- |
| Characters read correctly, before any rules | 848 / 856 (99.1%) |
| Digits / letters | 737 / 740 (99.6%) and 111 / 116 (95.7%) |
| Rows auto-posted after the rules | 29 / 36, none with an error |

On 60 synthetic logs the straight-through rate is 60.9% clean, 28.9% phone-like and 8.5% rough, with 3 clean-condition
rows auto-posted with an error (none a mileage error). See cells 15, 21, 24 and 36 of the full notebook.

## Known limits

- The whole form must be visible, flat and in reasonable light. A photo that cannot be straightened is rejected.
- Only Form ML-7 is supported.
- Digits the model finds borderline (for example 0 versus 6) can change between Colab sessions, because the photo is
  straightened slightly differently each time. The rules are what catch these.
- The visit schedule is checked per employee, not per date.

## Data

No real logs are used anywhere. All forms are generated programmatically or filled in by hand with made-up data.
