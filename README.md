
# Basic Commands for Data Science Libraries

## NumPy
- Importing NumPy:
  ```python
  import numpy as np
  ```
- Creating an array:
  ```python
  array = np.array([1, 2, 3])
  ```
- Array operations:
  ```python
  sum_array = np.sum(array)
  mean_array = np.mean(array)
  ```

## Pandas
- Importing Pandas:
  ```python
  import pandas as pd
  ```
- Creating a DataFrame:
  ```python
  df = pd.DataFrame({'Column1': [1, 2], 'Column2': [3, 4]})
  ```
- Reading a CSV file:
  ```python
  df = pd.read_csv('file.csv')
  ```

## Matplotlib
- Importing Matplotlib:
  ```python
  import matplotlib.pyplot as plt
  ```
- Plotting a simple line graph:
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.show()
  ```

## Seaborn
- Importing Seaborn:
  ```python
  import seaborn as sns
  ```
- Creating a scatter plot:
  ```python
  sns.scatterplot(x='Column1', y='Column2', data=df)
  plt.show()
  ```

## Scikit-learn
- Importing Scikit-learn:
  ```python
  from sklearn.model_selection import train_test_split
  ```
- Splitting data:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  ```

## Basic Python Syntax
- Defining a function:
  ```python
  def my_function(param):
      return param * 2
  ```
- Using a loop:
  ```python
  for i in range(5):
      print(i)
  ```
- Conditional statements:
  ```python
  if condition:
      print("Condition is True")
  else:
      print("Condition is False")
  ```

### This README provides a quick reference for basic commands in popular data science libraries and basic Python syntax.
