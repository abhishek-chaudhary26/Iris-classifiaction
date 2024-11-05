# Iris Classification

This project is my first attempt at implementing a machine learning model, and I have chosen to keep it on my GitHub as a reminder of my journey and growth in the field of data science.

## Introduction
The Iris Classification project is a classic example in the world of machine learning, where the goal is to classify Iris flower species based on four features: 
- Sepal length
- Sepal width
- Petal length
- Petal width

I used a **Logistic Regression** model, which is implemented using a **pipeline** that includes feature standardization and model training. This simple approach provided a strong foundation in understanding how to preprocess data and train a basic model effectively.

## Technologies Used
- **Python** for the implementation
- **Scikit-Learn** for machine learning and data preprocessing
- **Seaborn** for visualizing data distributions

## Dataset
The Iris dataset is a well-known dataset in the machine learning community, containing 150 samples with 3 classes (Iris Setosa, Iris Versicolor, Iris Virginica), and each class has 50 samples. The dataset is part of Scikit-Learn's built-in datasets and can be loaded easily.

## Model
The project uses a **Logistic Regression** model with the following steps:
1. **Standardization**: The features are scaled using `StandardScaler` to improve model performance.
2. **Logistic Regression**: A simple and interpretable model used for classification.

### Pipeline Implementation
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a pipeline with standardization and logistic regression
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
