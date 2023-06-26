# Stock Price and Volatility Forecaster

This project is a Flask application that predicts stock prices and volatilities using a neural network model. It utilizes the Polygon API to retrieve historical stock data and trains a Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) model to forecast future prices and volatilities.

## Installation

1. Clone the repository to your local machine:

``` 
git clone <repository_url>
```


2. Install the required dependencies:

```python
 pip install -r requirements.txt
```

3. Obtain an API key from Polygon.io by signing up for an account.

4. Replace the placeholder API key in the code with your own API key; or use mine and make no edits:

```python
URL = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{three_months_ago}/{current_date}?apiKey=<your_api_key>'
```

# Usage
## Run the Flask application:

``` 
flask run
```
Open a web browser and go to
``` 
http://localhost:5000
``` 
to access the application.

## Enter the stock ticker symbol in the provided input field and submit the form.

The application will retrieve historical stock data, train the neural network model, and display the predicted price for the next day and the forecasted volatility.

# Additional Information
The application consists of the following classes:

### StockPriceForecaster
1. get_data(ticker): Retrieves historical stock data from the Polygon API.
2. preprocess_data(data): Preprocesses the data by scaling the prices.
3. create_model(): Initializes the neural network model (LSTM-GRU).
4. forecast_price_tomorrow(): Predicts the stock price for the next day.
5. forecast_historical_data(): Performs backtesting to evaluate the model's performance.
### StockVolatilityPredictor
1. get_data(ticker): Retrieves historical stock data from the Polygon API.
2. preprocess_data(data): Preprocesses the data by scaling the prices.
3. create_model(): Initializes the neural network model (LSTM-GRU).
4. calculate_volatility(prices): Calculates the volatility based on the price data.
5. forecast_volatility_tomorrow(): Predicts the stock volatility for the next day.
6. get_predictions(data): Performs backtesting to evaluate the model's performance.

# The Flask application has two routes:

1. '/': Renders the index.html template which is some context on skills used to execute the project and takes us to '/forecast_historical' route.
2. '/forecast_historical': Handles user requests, predicts stock prices and volatilities, and renders the forecast_historical.html template with the predicted values.

# NEXT STEPS
1. Create a interactive UI to take different hyper tuning as from input of original model and visualize the performance. 
2. Store All forecasts in a MYSQL instance so that the forecasts can be analyzed to retrain the original the model on its errors. 
3. Create a portfolio for User (login/Logout) That allows users to manage there own portfolios. 
4. Create a route that allows for feature engineering and labeling to increase model accuracy. 
5. Host the application on the cloud using NGINX as a reverse proxy to handle requests. 
