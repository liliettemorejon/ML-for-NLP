# Form ML-7 Mileage Log Reader (ISM 6642, Group 4)

A prototype that reads handwritten Sabal Coast Home Health mileage logs (Form ML-7) from a phone photo, checks every row against the form's rules, and decides **row by row** what can be auto-posted and what goes to a clerk.

**Status:** working prototype. Runs end to end on synthetic logs and on a real handwritten form photographed with a phone.

## How to run it

**One-command demo (any computer, no Colab):**

    pip install -r requirements.txt
    python demo.py samples/your_photo.jpg --save result.png

Prints what was read in each row, the AUTO-POST / REVIEW decision for each row, and the reasons. Works with JPG, PNG and iPhone HEIC photos. `--save` writes the straightened form with each read drawn above its box and the decision at the end of each row.

Before the first run, copy the trained model `char_cnn_aug_best.pt` from Google Drive (`Group4_project`) into `models/`.

Options:
- `--no-reference-checks` for a log from outside our made-up HR data (for example the instructor's): skips the HR list, visit schedule and 60-day checks; every other rule still runs.
- `--as-of YYYY-MM-DD` to set the date used as "today" for the 60-day rule.

**Full notebook (Google Colab):** training, synthetic logs and the evaluation.

1. Open `notebooks/MNIST_Final_Group4.ipynb` in Google Colab and save a copy to your own Drive (File > Save a copy in Drive).
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > **Run all**, and approve the Google Drive pop-up.

The first run takes about 20-30 minutes: it downloads EMNIST, generates 60 test logs, trains both models and runs the evaluation. Everything is saved to `MyDrive/Group4_project`. The photo cell shows an upload button: pick any photo of a filled-in form (JPG, PNG or HEIC). The next cell compares the read against what was written on our test forms.

## Pipeline

1. **Registration:** match visual features (SIFT) between the photo and a blank Form ML-7, then straighten the photo onto the blank form.
2. **Cell extraction:** erase the printed box lines and cut out each box, keeping only ink strokes that reach the middle of the box.
3. **Normalization:** make every crop look like EMNIST (white on black, fit into 20x20, centered by center of mass in 28x28). The same normalization is used in training and at inference.
4. **Recognition:** a small CNN reads each box. Output is restricted by field: digit boxes can only be 0-9, letter boxes only A-Z.
5. **Validation:** check the form's rules (below) and decide AUTO-POST or REVIEW for each row.

## Data

- **EMNIST ByClass**, digits and capital letters only (36 classes). Lowercase is dropped because the form only allows capitals.
- Train 480,594 / validation 53,399 / test 89,264 characters (roughly 2 digits for every letter).
- **Synthetic test logs:** 60 filled Form ML-7 images with ground truth, 20 each under three conditions (clean, phone-like, rough). Handwriting comes only from the EMNIST **test** split, so no training image appears in a test log.
- **Real test form:** Form ML-7 printed, filled in by hand with made-up trips, photographed with an iPhone.
- No real mileage logs or reimbursement documents are used anywhere in this project. Employee IDs and client codes are made up.

## Model

- CNN: Conv(32) > ReLU > MaxPool > Conv(64) > ReLU > MaxPool > Linear(128) > ReLU > Dropout(0.3) > Linear(36) > log_softmax (2 convolutional + 2 fully connected layers, 424,996 parameters)
- Adam, learning rate 0.001, NLLLoss with class weights (letters are rarer than digits), 5 epochs, batch size 128
- Main model is trained with augmentation: random tilt, shift, scale, shear, thicker or thinner strokes, and blur

## Results

Every number states the condition it was measured under.

**Isolated EMNIST test characters (clean, centered, no form), baseline model:**

| | Free choice (36 classes) | Restricted by field |
| --- | --- | --- |
| Digits | 93.3% | 99.3% |
| Letters | 89.9% | 97.8% |

**Synthetic Form ML-7 logs, 20 logs per condition, raw accuracy before any rules:**

| Condition | Model | Digit acc. | Letter acc. | Field acc. | Odometer 6 of 6 right |
| --- | --- | --- | --- | --- | --- |
| Clean | baseline | 98.2% | 93.7% | 89.4% | 87.4% |
| Clean | augmented | 98.7% | 95.7% | 92.4% | 91.4% |
| Phone | baseline | 94.4% | 87.5% | 74.4% | 68.4% |
| Phone | augmented | 96.7% | 91.3% | 83.1% | 80.5% |
| Rough | baseline | 88.0% | 74.7% | 53.5% | 45.8% |
| Rough | augmented | 92.0% | 83.5% | 67.0% | 63.5% |

**Cell extraction (separate from recognition):** 60 of 60 logs registered. Only 2 boxes with handwriting (out of about 10,000 characters) came out empty, both in rough logs.

**Per-row decisions after the rules (augmented model, synthetic logs):**

| Condition | Rows | Auto-posted | Straight-through rate | Auto-posted with an error | Auto-posted with a mileage error | Reviewed but correct |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | 151 | 91 | 60.3% | 1 (1.1%) | 0 | 7 |
| Phone | 128 | 26 | 20.3% | 0 | 0 | 2 |
| Rough | 130 | 6 | 4.6% | 0 | 0 | 0 |

**Real handwriting (1 form, 9 rows, printed Form ML-7, iPhone photo, augmented model):**

- Characters 210/214 (98.1%): digits 185/185 (100%), letters 25/29 (86.2%). Fields fully correct: 45/48.
- Decision: 6 of 9 rows AUTO-POST, 3 to REVIEW. The 3 reviewed rows are exactly the 3 rows with a misread client code (caught by the visit-schedule check). No row was auto-posted with an error.

One form is a small sample; more hand-filled forms are needed for a reliable real-handwriting number.

## Validation rules

- Employee ID exists in the HR master list
- Week ending is a valid date, a Sunday, not in the future, and within the last 60 days
- Trip dates are valid and fall inside the week ending
- Client code is on the employee's visit schedule
- Miles = odometer end - start, and end is after start
- Odometer start is not before the previous row's end
- A trip over 300 miles is flagged as implausible
- Total miles = sum of the rows
- Any character with confidence below 0.50 is flagged

**Per-row policy:** a row is auto-posted only if none of its checks fail. A problem with the employee ID or week ending sends the whole log to review. A wrong total is blamed on the rows already flagged; if no row is flagged, the whole log goes to review.

The HR master list and visit schedule are made-up stand-ins (`assets/reference_data.json`). The system cannot tell a misread from a clinician's own mistake (for example Figure 1, row 5: 42 miles written, 47 by the odometers); both go to review, which is the safe outcome.

## Known limitations

- Real-handwriting results come from one form; handwriting styles not in EMNIST (crossed 7s, some K shapes) are read less reliably.
- Ink that crosses a box line gets partly erased (for example, a 0 losing its bottom).
- Two digits crowding into each other across a box border can produce a broken crop.
- Paper that is not flat can shift the boxes slightly after straightening, mostly at the right edge (MILES column).
- Assumes the whole form is visible in the photo. If it isn't, the reader stops with an error instead of guessing.
- A misread client code that happens to be another client on the same schedule is not caught (the one auto-posted error on clean synthetic logs was not a mileage error).

## Team

| Name | Role |
| --- | --- |
| Liliette Morejon Averhoff | Data |
| Priscila | Lead Product |
| Yasori | QA |

## Sources and AI use

- The synthetic log generator is our own version, adapted from the design of the instructor's `make_log_samples.py`.
- Claude (Anthropic) was used as a coding assistant for the data loader, generator, registration and extraction code, training loop, evaluation cells, validation rules and the demo script. All code was reviewed, run and tested by the team.
- No OCR engine, document model or LLM is used to read the forms. The reader is our own CNN trained on EMNIST.
