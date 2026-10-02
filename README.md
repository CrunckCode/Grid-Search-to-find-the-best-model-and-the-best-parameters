# Grid Search to Find the Best Model and the Best Parameters

A scikit-learn workflow that trains a kernel SVM to predict whether a social network user buys a product, then uses k-fold cross-validation and grid search to tune it.

## Data
`Social_Network_Ads.csv`: 400 users with Age, Estimated Salary and `Purchased` (0 or 1).

## Method
1. Split 75/25 (random state 0) and standardise the two features on the training set.
2. Train an RBF-kernel SVM and evaluate it with a confusion matrix and accuracy.
3. Apply 10-fold cross-validation on the training set to estimate how well the model generalises.
4. Run `GridSearchCV` (10-fold, scoring accuracy) over a linear kernel with C in {0.25, 0.5, 0.75, 1} and an RBF kernel with the same C values and gamma from 0.1 to 0.9.
5. Plot the decision regions for the training and test sets with matplotlib.

## Results (re-run)
| Step | Result |
|---|---|
| Test accuracy, RBF SVM | 93.0% (confusion matrix [[64, 4], [3, 29]]) |
| 10-fold CV accuracy, training set | 90.33% (standard deviation 6.57%) |
| Best grid search accuracy | 90.67% with `C=0.5`, `gamma=0.6`, RBF kernel |

The cross-validated figure (about 90%) is a more honest estimate than the single-split test accuracy (93%), because it averages over ten folds, and the 6.6% standard deviation shows how much it varies between folds.

## How to run
```
pip install numpy pandas matplotlib scikit-learn
python "Grid Search to find the best model and the best parameters.py"
```

## Limitations
Two features only, a small dataset, and the grid search is run on the same training data used for the CV estimate. For an unbiased final figure, evaluate once on the held-out test set after tuning.
