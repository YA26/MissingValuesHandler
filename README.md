# Handling missing values with a random forest automatically

Random forest's definition of proximity is a solution that can be used to replace missing values in a dataset.

If you want to know more about how that implementation was done, you can read this article: https://medium.com/analytics-vidhya/replacing-missing-values-in-a-dataset-by-building-a-random-forest-with-python-d82d4ff24223


## Do you want to give it a try?

- Make sure all dependencies are installed. If that’s not the case you can install them with pip install requirements.txt

- Open the *run.py* file

- Import your dataset or use those that are in the repository

- Choose what type of random forest you will use(regressor or classifier)

- Set up the parameters of the random forest you chose. The main difference resides in the criterion: it is gini or entropy for a random forest classifier and mse(mean square error) for a regressor.

- Set up essentials parameters like the number of iterations, the additional trees, the base estimator…
