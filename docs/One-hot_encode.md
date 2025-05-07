# One-Hot Encoding in SNH-AI Project

This document provides an overview of One-Hot Encoding as used in the SNH-AI project.

## What is One-Hot Encoding?

One-Hot Encoding is a process of converting categorical data variables into a numerical format that can be fed into machine learning algorithms to do a better job in prediction. It creates new binary (0 or 1) columns for each unique category in the original categorical column.

For example, if a `region` column has categories like "North", "South", "East", "West":

| customer_id | region |
|-------------|--------|
| 1           | North  |
| 2           | South  |
| 3           | East   |
| 4           | North  |

After one-hot encoding, it would look like:

| customer_id | region_North | region_South | region_East | region_West |
|-------------|--------------|--------------|-------------|-------------|
| 1           | 1            | 0            | 0           | 0           |
| 2           | 0            | 1            | 0           | 0           |
| 3           | 0            | 0            | 1           | 0           |
| 4           | 1            | 0            | 0           | 0           |

## Why use One-Hot Encoding?

*   **Compatibility with ML Algorithms**: Most machine learning algorithms require numerical input.
*   **No Ordinal Relationship Assumption**: Unlike label encoding, one-hot encoding does not assume an ordinal relationship between categories (e.g., "North" < "South" is not implied). This is crucial for nominal categorical data.
*   **Improved Model Performance**: For many algorithms, especially linear models and neural networks, one-hot encoding can lead to better performance by representing categorical features appropriately.

## Implementation in this Project

In the SNH-AI project, one-hot encoding is applied to the `region` column of the customer dataset.

*   **Tool Used**: `sklearn.preprocessing.OneHotEncoder` from the scikit-learn library.
*   **Script**: `src/encode.py`
*   **Details**:
    *   The encoder is typically fit on the training data to learn all unique categories.
    *   The fitted encoder is then used to transform the `region` column into multiple binary columns.
    *   Considerations for handling unknown categories in new data (if applicable, e.g., `handle_unknown='ignore'`).
    *   The resulting encoded columns are then concatenated back to the main DataFrame, and the original `region` column is usually dropped.

## Parameters for `OneHotEncoder` (Commonly Used)

*   `categories='auto'`: Determine categories automatically from the training data.
*   `drop=None`: Whether to drop one of the categories to avoid multicollinearity. Common options are `None` (keep all) or `'first'` (drop the first category). The choice might depend on the downstream model.
*   `sparse_output=False` (or `sparse=False` in older versions): By default, `OneHotEncoder` returns a sparse matrix. Setting this to `False` returns a dense NumPy array, which is often easier to work with when concatenating with pandas DataFrames. For very high cardinality features, sparse might be necessary for memory efficiency.
*   `handle_unknown='error'`: How to handle categories encountered during `transform` that were not seen during `fit`. `'error'` will raise an error, `'ignore'` will encode them as all zeros.

## Example Snippet (Conceptual)

```python
# Assuming 'df' is a pandas DataFrame and 'region' is the column to encode
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd

# # Initialize the OneHotEncoder
# # Using sparse_output=False for dense array output
# # Using handle_unknown='ignore' if new data might have unseen regions
# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# # Fit and transform the 'region' column
# # The input to fit_transform should be 2D (e.g., df[['region']])
# encoded_regions = encoder.fit_transform(df[['region']])

# # Create a DataFrame with the new encoded columns
# # feature_names_out_ provides human-readable column names
# encoded_df = pd.DataFrame(encoded_regions, columns=encoder.get_feature_names_out(['region']))

# # Concatenate with the original DataFrame and drop the original 'region' column
# df = pd.concat([df.drop('region', axis=1), encoded_df], axis=1)
```

## Considerations

*   **Curse of Dimensionality**: One-hot encoding can lead to a large number of new features if the categorical variable has many unique categories (high cardinality). This can increase model complexity and training time.
*   **Multicollinearity**: If all one-hot encoded columns are used, they sum up to 1, creating perfect multicollinearity. Some models (like linear regression) are sensitive to this. Dropping one category (`drop='first'` or `drop='if_binary'`) can mitigate this, or the model itself might handle it.
*   **Data Leakage**: Ensure that encoding is done *after* splitting data into training and testing sets, or that the encoder is fit *only* on the training data to prevent data leakage.