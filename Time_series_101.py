# Import libraries and dependencies
import numpy as np
import pandas as pd
%matplotlib inline
import yfinance as yf

# Retrieve AMZN data
amzn_data = yf.download("AMZN", start = "2010-01-01", end = "2021-05-28")
# Visualize top rows
amzn_data.head()

# lets take a look at the closing prices for Amzn
close = amzn_data["Close"].plot(figsize = (10,10), title='Amazon Close Price', ylabel = "Price")

# Trim the data so it is from 2015 and on
amzn_data = amzn_data.loc["2015-01-01":, :]
amzn_data.head()
#plot the close price from 2015 and on
amzn_data["Close"].plot(figsize=(20,10), title='Amazon stock from 2015 to May 2021', ylabel = "Price");

# We want to look at the noise and the trend within this plot, we will use the Hodrick-Prescott-Filter to separate the noise and the trend
# We make a new data set just for the Close and then we will add columns noise and trend to it
# in order to do this we need to import the statsmodel 
import statsmodels.api as sm
# We are using the Hodrick-Prescott to break the close column into a noise variable and a trend variable
amzn_noise, amzn_trend = sm.tsa.filters.hpfilter(amzn_data.Close)

# Construct a new dataframe with just the close and trend and noise
new_df = pd.DataFrame(amzn_data["Close"])
new_df["Noise"] = amzn_noise
new_df["Trend"] = amzn_trend
new_df.head()

# we should compare our Close price to our trend pice 
new_df[["Close", "Trend"]]["2015-01-01":].plot(figsize=(10,10), title = "Close price Vs Trend Price", ylabel = "Price");

# lets plot the noise and see what it looks like
new_df["Noise"].plot(figsize=(10,10), title = "Amazon Noise")

## Forecast Returns With and ARMA Model

# We want to use the ARMA Model to help predict the future returns
# To do this we need to make a new dataframe and add a column "Retruns"
# The frist graph we plotted of the Close prices was not a stationary graph so we need to make that a stationary graph
amzn_returns = pd.DataFrame(amzn_data["Close"].pct_change() * 100).dropna()
amzn_returns.rename(columns = {"Close" : "Returns"}, inplace = True)
amzn_returns.head()

#plot model to make sure it is stationary
# This graph is not stationary
amzn_returns.plot(figsize =(10,10), title = "Stationary Model"); 

# We now need to plot the auto-correlation and partial-autocorrelation for the returns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#plot the partial autocorrelation to get number of outliers
pacf = plot_pacf(amzn_returns)

# When we anaylze this graph we can see there is 1 significant outlier on both and there is 1 other outlier that is less signaficant 
# based on this grpah we can either use an order of (1,1), or (2,2). we will try both to see what gives us better results
from statsmodels.tsa.arima_model import ARMA

# We can now create the ARMA model using the return values and the order
# The order variable is defining the AR and the MA or the Auto-Regressive and Moving Average, both of these components will help predict the future values based on the past values
model = ARMA(amzn_returns, order = (5,1))

# Set results equal to model.fit so we fit the model to the data
results = model.fit()

# We will now plot the forecasted return for the next 10 days for the return on the amazon stock
amzn_forecast = pd.DataFrame(results.forecast(steps = 10)[0]).plot(title = "Amazon 10 Day Return Forecast", xlabel = "Days", ylabel = "Returns", figsize = (10,10))
amzn_forecast;

# Plot the summary of the model 
results.summary()

# Lets try and use 2,2 for our AR and MA to see how those results work for the ARMA Model
# Build ARMA Model 
model_2 = ARMA(amzn_returns, order = (2,2)) 


# We need to fit this model like we did before
results_2 = model_2.fit()

# plot the forecasted return for the next 10 days for the return on the amazon stock
amzn_forecast_2 = pd.DataFrame(results_2.forecast(steps = 10)[0]).plot(title = "Amazon 10 Day Return Forecast")
amzn_forecast_2;

# If we compare this graph the to the first one we can see a significant change in them. This one tells me the returns for amazon stock over the next 10 days will be increase and decrease everyday with a downward slope or a decline.
# Lets take a look at the summary to see how this model did 
results_2.summary()

##  Looking at the p-values we can see the first AR and MA we very accruate however the second AR and MA were not. I would not use this model to predict the returns of the stock. The p-values for them all need to be atlease 0.1 but prefereably below 0.05. The AR2 and the MA2 have p-values of .414 and .439.

# ARIMA Model Forecast

# Plot just the close data from the original dataset to see if this is stationary or not
amzn_data.Close.plot(figsize=(10,10), title = "Close price Vs Trend Price", ylabel = "Price"); 

# As excpected this is nonstationary data, when we put this data into the ARIMA model we have to to choose our order. 
# The order will be similar to the ARMA model because we are using the same number for the AR and MA (1,1)
# However the ARIMA model calls for one more input into the order and this will be the middle number and it lets the model know how many time to diff the dataset or do a pct_change
# Typically for financial data we will use 1
# We can check and see if this data would need to use more than one pct_changes ro diffs by looking at a graph
pct_change = amzn_data.Close.pct_change().plot(figsize =(10,10), title = "Stationary Model");

# This is stationary so we will only have to use 1 in our order parameter

# With this model we also want to preditc the closing price of Amazon.
from statsmodels.tsa.arima.model import ARIMA
ARIMA_model = ARIMA(amzn_data["Close"], order = (1,1,1))
# We now need to fit the ARIMA Model
# If an error appears that states ConvergenceWarning the below line of code can be used inside the fit paranthesis
#method_kwargs = {"maxiter":200}
# if this error appears it is letting you know that it did not iteriate over enough data
# it reached the deafault max and then stated that there are more possibilites but the default stopped us here
# the maxiter :200 will iterate over the data 200 times to get make sure it uses enough data 
ARIMA_results = ARIMA_model.fit()
ARIMA_results.summary()

# Forecasting Garch to Determine Volatility
from arch import arch_model
# Again we will be using the returns of Amazon Stock
amzn_returns.head()

# p and q are our lags that we have been using and if you are unsure we can look back at the pacf_model to get this 
model_garch = arch_model(amzn_returns, mean="Zero", vol="GARCH", p=1, q=1)

# Fit the garch model
# disp="off" just doesn't display the fitting
garch_fit = model_garch.fit(disp="off")
# retrieve our summary
garch_fit.summary()

# Lets plot the garch model and look at the voalatility of amazon stock
garch_plot = garch_fit.plot()
# lets look at the next 5 days of volatility using this model.
prediction_days = 5
forecast = garch_fit.forecast(start = "2021-05-27", horizon = prediction_days)
forecast
# we now want to annualize this forecast
ann_forecast = np.sqrt(forecast.variance.dropna()*252)
ann_forecast

# To use this data in a plot we need to transpose it so it will fit a graph better
final_plot = ann_forecast.T
final_plot.plot()