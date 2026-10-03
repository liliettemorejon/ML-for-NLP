# Form ML-7 Mileage Log Reader (ISM 6642, Group 4)

A prototype that reads handwritten Sabal Coast Home Health mileage logs (Form ML-7) from a phone photo, checks every row against the form's rules, and decides what can be auto-posted and what goes to a clerk.

**Status:** working prototype, in progress. The full pipeline runs end to end on synthetic logs. Testing on real handwritten forms is the next step.

## How to run it

**Current version (Google Colab):**

1. Open `notebooks/MNIST_Final_Group4.ipynb` in Google Colab and save a copy to your own Drive (File > Save a copy in Drive).
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > **Run all**, and approve the Google Drive pop-up.

The first run takes about 20-30 minutes. It downloads EMNIST, generates 60 test logs, trains the models and runs the evaluation. Everything is saved to `MyDrive/Group4_project`. The last cell reads a photo of a filled-in form: drag the photo into Colab's Files panel and set `PHOTO_PATH` to its name.

**Planned:** a one-command demo, `python demo.py path/to/photo.jpg`, that runs without Colab.

## Pipeline

1. **Registration:** match visual features (SIFT) between the photo and a blank Form ML-7, then straighten the photo onto the blank form.
2. **Cell extraction:** erase the printed box lines and cut out each box, keeping only ink strokes that reach the middle of the box.
3. **Normalization:** make every crop look like EMNIST (white on black, fit into 20x20, centered by center of mass in 28x28). The same normalization is used in training and at inference.
4. **Recognition:** a small CNN reads each box. Output is restricted by field: digit boxes can only be 0-9, letter boxes only A-Z.
5. **Validation:** check the form's rules (below) and decide AUTO-POST or REVIEW.

## Data

- **EMNIST ByClass**, digits and capital letters only (36 classes). Lowercase is dropped because the form only allows capitals.
- Train 480,594 / validation 53,399 / test 89,264 characters (roughly 2 digits for every letter).
- **Synthetic test logs:** 60 filled Form ML-7 images with ground truth, 20 each under three conditions (clean, phone-like, rough). Handwriting comes only from the EMNIST **test** split, so no training image appears in a test log.
- No real mileage logs or reimbursement documents are used anywhere in this project.

## Model

- CNN: Conv(32) > MaxPool > Conv(64) > MaxPool > Linear(128) > Dropout(0.3) > Linear(36) > log_softmax
- Adam, learning rate 0.001, NLLLoss with class weights (letters are rarer than digits), 5 epochs, batch size 128
- Main model is trained with augmentation: random tilt, shift, scale, shear, thicker or thinner strokes, and blur

## Results so far

All numbers state the condition they were measured under.

**Isolated EMNIST test characters (clean, centered, no form), baseline model:**

| | Free choice (36 classes) | Restricted by field |
| --- | --- | --- |
| Digits | 91.8% | 99.3% |
| Letters | 90.5% | 97.9% |

**Synthetic Form ML-7 logs, 20 logs per condition, baseline vs. augmented model:**

| Condition | Model | Digit acc. | Letter acc. | Field acc. | Odometer 6 of 6 right |
| --- | --- | --- | --- | --- | --- |
| Clean | baseline | 98.2% | 93.7% | 89.3% | 88.1% |
| Clean | augmented | 98.8% | 97.4% | 93.6% | 91.1% |
| Phone | baseline | 94.9% | 85.8% | 74.0% | 68.0% |
| Phone | augmented | 96.6% | 92.7% | 83.3% | 78.5% |
| Rough | baseline | 87.9% | 74.7% | 52.7% | 45.0% |
| Rough | augmented | 92.6% | 84.2% | 69.0% | 64.2% |

- **Registration:** 60 of 60 logs registered. Only 2 boxes with handwriting (out of about 10,000 characters) came out empty.
- **Auto-post decisions (augmented model):** 0 logs auto-posted with an error and 0 mileage errors paid out. Only 2 of 20 clean logs auto-posted, though, because most logs contain at least one misread somewhere and get sent to review.

## Validation rules

Implemented:

- Miles = odometer end - start, and end is after start
- Odometer start is not before the previous row's end
- Trip dates are valid and fall inside the week ending date
- Total miles = sum of the rows
- A trip over 300 miles is flagged as implausible
- Any character with confidence below 0.50 is flagged

Not yet implemented:

- Week ending is a Sunday within the last 60 days
- Employee ID exists in the HR master list
- Client code is on the clinician's visit schedule
- Per-row decisions (currently the whole log is posted or reviewed)

## Known limitations

- Not yet tested on real handwriting photographed from a printed form.
- Ink that crosses a box line gets partly erased (for example, a 0 losing its bottom).
- Two digits crowding into each other across a box border can produce a broken crop.
- Assumes the whole form is visible in the photo. If it isn't, the reader stops with an error instead of guessing.
- Client codes have no arithmetic check, so a misread letter can only be caught by low confidence (until the visit-schedule check exists).

## Team

| Name | Role |
| --- | --- |
| Liliette Morejon Averhoff | Pipeline / Model |
| _name_ | _role_ |
| _name_ | _role_ |

## Sources and AI use

- The synthetic log generator is our own version, adapted from the design of the instructor's `make_log_samples.py`.
- Claude (Anthropic) was used as a coding assistant for the data loader, generator, registration and extraction code, training loop and evaluation cells. All code was reviewed, run and tested by the team.
- No OCR engine, document model or LLM is used to read the forms. The reader is our own CNN trained on EMNIST.
