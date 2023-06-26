from flask import Flask, render_template, request
import time
import json
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense
import numpy as np
from flask.json import JSONEncoder

"""
A class to appropriaely convert arrays to parsable JSON to dynamically populate Graphs in UI with Javascript.
"""
class NumpyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

"""
Initializing the Flask Application
"""   
app = Flask(__name__)
app.json_encoder = NumpyJSONEncoder
app = Flask(__name__)

"""
Creating a stockPriceForecaster Class 

get_data function pulls the data in a 90 day range as a self chosen default and returns a json response.
Try and except statement if there was an error pulling this data.
If there wasnt an error, the result is created and the timestamps are converted to a timestamp interpretable by the nueral network. 

preprocess_data function appropriately reshapes the array for scaling of the data and returns the scaled prices. 

create_model class initializes a Nueral Network I call a LSTM-GRU but is a customized nueral network that yielded nice results. Can be better; but for 
the sake of this project I left it as is. 

forecast_price_tomorroww will use the get_data function to pull in the dadta, the preprocess_data function to scale our data, 
create the neccessary arrays to train the data on, reshape the X and Y values to apropriately fit the model, make the predictions for tomorrow. 

will use the get_data function to pull in the dadta, the preprocess_data function to scale our data, create the neccessary arrays to train the data on,
reshape the X and Y values to apropriately fit the model, make the predictions for tomorrow. each day and in addition populate a list of the actual prices
and the predicted prices for the backtest of the models performance that will be rendered in the UI. Dates are also passed so we can populate the X axis of the
backtested graphs in the UI. 



"""
class StockPriceForecaster:
    def __init__(self, ticker):
        self.ticker = ticker
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    def get_data(self, ticker):
        try:
            time.sleep(0.01)  # sleep for 0.01 seconds
            current_date = str(datetime.today())[:10]
            three_months_ago = str(datetime.today() - timedelta(days=90))[:10]
            URL = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{three_months_ago}/{current_date}?apiKey=1pgshzXCU0epjsQD1kHXC7XBc2nOcqqp'
            api_result = requests.get(url=URL)
            result_json = api_result.json()

            if 'results' in result_json:
                for i in range(len(result_json['results'])):
                    unix_time = int(result_json['results'][i]['t']) / 1000
                    datetime_obj = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
                    result_json['results'][i]['t'] = datetime_obj

                return result_json
            else:
                return None
        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {e}")
            return None
    
    def preprocess_data(self, data):
        prices = [result['c'] for result in data['results']]
        prices = np.array(prices).reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        return scaled_prices
    
    def create_model(self):
        model = Sequential()
        model.add(GRU(64, return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(GRU(32, return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def forecast_price_tomorrow(self):
        historical_data = self.get_data(self.ticker)
        
        if historical_data is None:
            return None
        
        prices = self.preprocess_data(historical_data)
        
        X = prices[:-1]
        y = prices[1:]
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        self.model = self.create_model()
        self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)
        
        last_price = prices[-1]
        last_price = last_price.reshape(1, 1, 1)
        
        predicted_price = self.model.predict(last_price)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        
        return predicted_price[0][0]
    def forecast_historical_data(self):
            historical_data = self.get_data(self.ticker)
            
            if historical_data is None:
                return None
            
            prices = self.preprocess_data(historical_data)
            
            X = prices[:-1]
            y = prices[1:]
            
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            self.model = self.create_model()
            self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)
            
            start_index = int(len(historical_data['results']) * 2 / 3)
            predicted_prices = []
            actual_prices = []
            dates = []

            for i in range(start_index, len(historical_data['results']) - 1):
                X_pred = prices[i].reshape(1, -1, 1)
                predicted_price = self.model.predict(X_pred)[0][0]
                actual_price = prices[i + 1][-1]

               
                predicted_price = self.scaler.inverse_transform([[predicted_price]])[0][0]
                actual_price = self.scaler.inverse_transform([[actual_price]])[0][0]

                predicted_prices.append(predicted_price)
                actual_prices.append(actual_price)

               
                date = historical_data['results'][i]['t']
                dates.append(date)
            
            return dates, predicted_prices, actual_prices
 
"""
Creating a sstockVolatilityPredictor Class 

get_data function pulls the data in a 90 day range as a self chosen default and returns a json response.
Try and except statement if there was an error pulling this data.
If there wasnt an error, the result is created and the timestamps are converted to a timestamp interpretable by the nueral network. 

preprocess_data function appropriately reshapes the array for scaling of the data and returns the scaled prices. 

create_model class initializes a Nueral Network I call a LSTM-GRU but is a customized nueral network that yielded nice results. Can be better; but for 
the sake of this project I left it as is. 

Calculate volatility returns volatillity. 

forcast_volatility_tomorrow will properly format the data to feed to the LSTM-GRU in the create_model function and return the forecast for tomorrow. 

get_predictions is parralel logic to forecast_historical_data in the last function; which will help to graph back-test and analyze model performance. 
"""   
class StockVolatilityPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    def get_data(self, ticker):
        try:
            time.sleep(0.01)  # sleep for 0.01 seconds
            current_date = str(datetime.today())[:10]
            three_months_ago = str(datetime.today() - timedelta(days=90))[:10]
            URL = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{three_months_ago}/{current_date}?apiKey=1pgshzXCU0epjsQD1kHXC7XBc2nOcqqp'
            api_result = requests.get(url=URL)
            result_json = api_result.json()

            if 'results' in result_json:
                for i in range(len(result_json['results'])):
                    unix_time = int(result_json['results'][i]['t']) / 1000
                    datetime_obj = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
                    result_json['results'][i]['t'] = datetime_obj

                return result_json
            else:
                return None
        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {e}")
            return None
    
    def preprocess_data(self, data):
        prices = [result['c'] for result in data['results']]
        prices = np.array(prices).reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        return scaled_prices
    
    def create_model(self):
        model = Sequential()
        model.add(GRU(64, return_sequences=True, input_shape=(1, 1)))
        model.add(LSTM(64, return_sequences=True))
        model.add(GRU(32, return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def calculate_volatility(self, prices):
        log_returns = np.log(prices[1:] / prices[:-1])
        log_returns = np.where(np.isfinite(log_returns), log_returns, 0)  # Replace NaN and -inf with 0
        avg_return = np.mean(log_returns)
        deviations = log_returns - avg_return
        squared_deviations = deviations**2
        variance = np.mean(squared_deviations)
        volatility = np.sqrt(variance)
        return volatility
    
    def forecast_volatility_tomorrow(self):
        historical_data = self.get_data(self.ticker)
        
        if historical_data is None:
            return None
        
        prices = self.preprocess_data(historical_data)
        
        volatilities = [self.calculate_volatility(prices[:i+1]) for i in range(len(prices))]
        
        # Remove NaN and 0 values from volatilities
        volatilities = [volatility for volatility in volatilities if not np.isnan(volatility) and volatility != 0]
        
        if len(volatilities) < 2:
            return None
        
        X = np.array(volatilities[:-1]).reshape(-1, 1)
        y = np.array(volatilities[1:]).reshape(-1, 1)
        
        X = self.scaler.fit_transform(X)
        y = self.scaler.transform(y)
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        self.model = self.create_model()
        self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)
        
        last_volatility = volatilities[-1]
        last_volatility = np.array(last_volatility).reshape(1, 1)
        
        last_volatility_scaled = self.scaler.transform(last_volatility)
        last_volatility_scaled = last_volatility_scaled.reshape(1, 1, 1)
        
        predicted_volatility_scaled = self.model.predict(last_volatility_scaled)
        predicted_volatility = self.scaler.inverse_transform(predicted_volatility_scaled)
        
        return predicted_volatility[0][0]
    
    def get_predictions(self, data):
        actual_volatilities = []
        predicted_volatilities = []
        dates = []  # List to store the corresponding dates

        prices = self.preprocess_data(data)

        volatilities = [self.calculate_volatility(prices[:i+1]) for i in range(len(prices))]

        # Remove NaN and 0 values from volatilities
        volatilities = [volatility for volatility in volatilities if not np.isnan(volatility) and volatility != 0]

        if len(volatilities) < 2:
            return None

        X = np.array(volatilities[:-1]).reshape(-1, 1)
        y = np.array(volatilities[1:]).reshape(-1, 1)

        X = self.scaler.fit_transform(X)
        y = self.scaler.transform(y)

        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.model = self.create_model()
        self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)

        for i in range(len(prices) - 1):
            try:
                actual_volatilities.append(float(volatilities[i+1]))

                current_volatility = volatilities[i]
                current_volatility = np.array(current_volatility).reshape(1, 1)

                current_volatility_scaled = self.scaler.transform(current_volatility)
                current_volatility_scaled = current_volatility_scaled.reshape(1, 1, 1)

                predicted_volatility_scaled = self.model.predict(current_volatility_scaled)
                predicted_volatility = self.scaler.inverse_transform(predicted_volatility_scaled)

                predicted_volatilities.append(float(predicted_volatility[0][0]))

                # Extract the corresponding date
                date = data['results'][i]['t']
                dates.append(date)
            except IndexError:
                break

        return dates, actual_volatilities, predicted_volatilities

"""
Some Context on the project. 
"""
@app.route('/')
def index():
    return render_template('index.html', methods=['GET','POST'])

"""
The route to handle user request and to populate the HTML template
dynamically to render the data. 
"""
@app.route('/forecast_historical', methods=['GET', 'POST'])
def forecast_historical():
    if request.method == 'POST':
        ticker = request.form['ticker']
        forecaster = StockPriceForecaster(ticker)
        forecasted_price = forecaster.forecast_price_tomorrow()
        price_dates, predicted_prices, actual_prices = forecaster.forecast_historical_data()
        print(price_dates)
        time.sleep(2)
        predictor = StockVolatilityPredictor(ticker)
        forecasted_volatility = predictor.forecast_volatility_tomorrow()
        historical_data = predictor.get_data(ticker)
        volatility_dates, predicted_volatilities, actual_volatilities = predictor.get_predictions(historical_data)

        predicted_prices_json = json.dumps(predicted_prices, cls=NumpyJSONEncoder)
        actual_volatilities_json = json.dumps(actual_volatilities, cls=NumpyJSONEncoder)
        predicted_volatilities_json = json.dumps(predicted_volatilities, cls=NumpyJSONEncoder)
        actual_prices_json = json.dumps(actual_prices, cls=NumpyJSONEncoder)

        price_dates_json = json.dumps(price_dates)
        volatility_dates_json = json.dumps(volatility_dates)

        
        return render_template('forecast_historical.html', predicted_prices=predicted_prices_json,
                            actual_volatilities=actual_volatilities_json,
                            predicted_volatilities=predicted_volatilities_json,
                            actual_prices=actual_prices_json, forecast_price=forecasted_price, forecasted_volatility=forecasted_volatility, ticker=ticker, price_dates=price_dates_json, volatility_dates=volatility_dates_json)
    return render_template('forecast_historical.html')


if __name__ == '__main__':
    app.run(debug=True)