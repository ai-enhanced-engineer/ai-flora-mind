# MaintainX - Take Home Assessment Applied Machine Learning Engineer

The goal of this take home assessment is to evaluate your machine learning knowledge and your software development skills.

## Task

Your task for this assessment is to train a model and expose it's inference function through an HTTP API. The choice of the model is yours.

The dataset to use is the [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) from scikit-learn.

The focus will be on the depth and considerations taken for the project and your overall thought process. For the model itself, we will be checking that it works (makes reasonable predictions), but we won't evaluate its overall performance.
We do not expect a production grade solution, but we would like you to showcase the areas in which you excel.

You’re welcome to explore any topic of interest, though the project should be substantial enough to support a 1h technical discussion upon submission. Here are some ideas to inspire you:

- Exploratory data analysis of the `iris` dataset
- Model training pipeline
- Comparing models
- Reasoning behind model choice / features
- Robust inference API
- Containerization
- Automated testing
- Project documentation

NOTE: You are free to use any tools to help you with this. Generative AI tools like Github Copilot (or similar) are not prohibited and are even recommended, but be sure to make good use of it (ie.: use in moderation where applicable) and review what it generated.

## Requirement

Your API will be need to follow theses requirements:

`HTTP POST /predict`

Request:

```json
{
  "sepal_length": 0.0, // float representing the sepal length in cm
  "sepal_width": 0.0, // float representing the sepal width in cm
  "petal_length": 0.0, // float representing the petal length in cm
  "petal_width": 0.0 // float representing the petal width in cm
}
```

Response

```json
{
  "prediction": "setosa" // The predicted iris type between "setosa", "versicolor" or "virginica"
}
```

## Testing

We have joined a postman collection and a bash script depending on your preference to test your API. Feel free to edit them to match your API's config (ex.: port, host, etc.).
