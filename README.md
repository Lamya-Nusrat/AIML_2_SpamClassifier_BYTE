# Spam SMS Classifier

**AVIP 2026 — AI/ML Engineering | Task 2**

## Project Overview

This project develops a machine learning model to classify SMS messages as Spam or Ham (legitimate messages). TF-IDF is used for text feature extraction, and Multinomial Naive Bayes is used for classification.

## Dataset

- **Name:** SMS Spam Collection
- **Source:** UCI Machine Learning Repository
- **Original messages:** 5,572
- **Messages after cleaning:** 5,158
- **Training samples:** 4,126
- **Testing samples:** 1,032
- **Split:** 80% training and 20% testing

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Methodology

1. Load the SMS dataset.
2. Clean the data and remove duplicate messages.
3. Convert text into numerical features using TF-IDF.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate the model on unseen test messages.
6. Save the trained model for future predictions.

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 98.16% |
| Spam Precision | 100% |
| Spam Recall | 85.16% |
| Spam F1-score | 91.98% |

## Evaluation Results

The project includes a confusion matrix, performance visualization, classification report, and 10 sample predictions with confidence scores.

## Tools and Libraries

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Joblib.

## Saved Model

The trained classifier is saved as `outputs/spam_classifier.joblib`.

To load the model:

```python
import joblib

model = joblib.load("outputs/spam_classifier.joblib")
prediction = model.predict(["Congratulations! You won a prize."])
print("Spam" if prediction[0] == 1 else "Ham")
```

Only load model files from trusted sources.

## Project Files

- `TASK2_AVIP.ipynb` — Training and evaluation notebook
- `outputs/spam_classifier.joblib` — Saved model
- `outputs/metrics.json` — Evaluation metrics
- `outputs/classification_report.txt` — Classification report
- `outputs/confusion_matrix.png` — Confusion matrix
- `outputs/performance_metrics.png` — Performance visualization
- `outputs/10_example_predictions.png` — Example predictions
- `outputs/10_example_predictions.csv` — Prediction results
- `outputs/new_message_predictions.csv` — Additional predictions
- `outputs/dataset_sample.csv` — Cleaned dataset sample

## Project Summary

The SMS Spam Classification project uses TF-IDF and Multinomial Naive Bayes to detect spam messages. The dataset contains 5,572 SMS messages, reduced to 5,158 after cleaning. An 80/20 train-test split was used. The model achieved 98.16% test accuracy, 100% spam precision, 85.16% spam recall, and 91.98% spam F1-score. Evaluation results include a confusion matrix, classification report, and sample predictions with confidence scores. The trained classifier was saved using Joblib for future inference.
