# StandardScaler Documentation

## Overview

`StandardScaler` is a preprocessing technique from the `scikit-learn` library used to standardize features by removing the mean and scaling to unit variance.

The standard score of a sample `x` is calculated as:

`z = (x - u) / s`

where `u` is the mean of the training samples or zero if `with_mean=False`, and `s` is the standard deviation of the training samples or one if `with_std=False`.

## Usage in SNH-AI Project

In the SNH-AI project, `StandardScaler` is applied to the following numerical features:

*   `age`
*   `annual_income`

This is done to ensure that these features contribute equally to the machine learning model, especially for distance-based algorithms like K-Means clustering, preventing features with larger magnitudes from dominating the results.

## Implementation

The `StandardScaler` is implemented in the `src/scale.py` script.

```python
# Example (conceptual from scale.py)
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Assuming 'data' is a pandas DataFrame with 'age' and 'annual_income' columns
# scaler = StandardScaler()
# numerical_cols = ['age', 'annual_income']
# data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
```

## Parameters

Key parameters of `StandardScaler` include:

*   `with_mean`: boolean, default `True`. If `True`, center the data before scaling.
*   `with_std`: boolean, default `True`. If `True`, scale the data to unit variance.

## Benefits

*   Improves the performance of many machine learning algorithms.
*   Handles features with different units and scales.

## Considerations

*   `StandardScaler` assumes that the data is normally distributed. If this assumption does not hold, other scaling techniques like `MinMaxScaler` or `RobustScaler` might be more appropriate.
*   The scaler should be `fit` only on the training data and then used to `transform` both the training and testing/new data to prevent data leakage.

## References

*   [Scikit-learn StandardScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)