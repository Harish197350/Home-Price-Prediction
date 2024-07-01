# Home Price Prediction using XGBoost

## Description

This project uses the XGBoost machine learning algorithm to predict home prices based on various factors such as income, schools, hospitals, and crime rates. The goal is to provide a reliable model that can estimate the price of a house given these features.

## Features

- **Data Loading and Preprocessing**:
  - Load data from a CSV file.
  - Preprocess the data by normalizing the features.

- **Model Training**:
  - Train an XGBoost regressor model on the training data.

- **Model Evaluation**:
  - Evaluate the model's performance using the Root Mean Squared Error (RMSE) metric.

## Prerequisites

- Python 3.x
- Pandas
- Scikit-learn
- XGBoost

## Installation

1. Clone the repository or download the `home_price_prediction.py` file.
    ```sh
    git clone https://github.com/yourusername/home-price-prediction.git
    cd home-price-prediction
    ```

2. Install the required libraries using pip:
    ```sh
    pip install pandas scikit-learn xgboost
    ```

## Usage

1. Ensure you have a dataset named `home_prices.csv` with the following columns: `income`, `schools`, `hospitals`, `crime_rate`, and `price`.
   
2. Place the `home_prices.csv` file in the same directory as the script.

3. Run the application by executing the following command in the terminal or command prompt:
    ```sh
    python home_price_prediction.py
    ```

4. The script will train the model and print the Root Mean Squared Error (RMSE) indicating the performance of the model.

## Code Overview

The main components of the script include:

- **Data Loading and Preprocessing**:
  - The data is loaded from a CSV file into a pandas DataFrame.
  - Features and target variable are separated, and the features are normalized using `StandardScaler`.

- **Model Training**:
  - An `XGBRegressor` model is created and trained on the training set.

- **Model Evaluation**:
  - Predictions are made on the test set.
  - The model's performance is evaluated using the Root Mean Squared Error (RMSE).

## Example

Here's a quick overview of what the dataset (`home_prices.csv`) might look like:

```csv
income,schools,hospitals,crime_rate,price
50000,5,2,0.02,300000
60000,8,3,0.01,400000
...
