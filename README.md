# Word Complexity Prediction

Project for the Machine Learning course at FMI, University of Bucharest: predicting how complex a word is (Kaggle InClass competition, 2025).

It includes feature-based baselines, an MLP, prediction and plotting scripts, evaluation metrics, and the project report (`Documentation.pdf`).

## Data

The competition data (`train.csv`, `test.csv`) belongs to the course organizers and is **not included**. The fine-tuned model (~440 MB) is not included either. Put the competition files in `date/` to run the scripts.

## Files

- `baseline.py`: feature-based baseline
- `MLP.py`: multilayer perceptron model
- `predict.py`: generates predictions
- `plot.py`: plots and error analysis (`prediction_errors.png`, `Distribution.pgf`)
- `evaluation_metrics.csv`: results
