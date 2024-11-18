import yfinance as yf
import pandas as pd
import numpy as np

# Function to get historical data for a stock ticker
def get_historical_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Return'] = stock_data['Adj Close'].pct_change()  # Calculate daily returns
    return stock_data

# Function to combine data of two tickers given a start and end date
def combine_data(stock_ticker, benchmark_ticker, start_date, end_date):
    # Get historical data for the stock and benchmark (S&P 500)
    stock_data = get_historical_data(stock_ticker, start_date, end_date)
    benchmark_data = get_historical_data(benchmark_ticker, start_date, end_date)
    
    # Drop rows with missing data
    stock_data = stock_data.dropna()
    benchmark_data = benchmark_data.dropna()
    
    # Align both dataframes by date
    combined_data = pd.merge(stock_data[['Return']], benchmark_data[['Return']], left_index=True, right_index=True, suffixes=('_stock', '_benchmark'))

    return combined_data

# Function to calculate beta
def calculate_beta(combined_data):

    # Calculate covariance and variance
    covariance = np.cov(combined_data['Return_stock'], combined_data['Return_benchmark'])[0, 1]
    variance = np.var(combined_data['Return_benchmark'])
    
    # Calculate beta
    beta = covariance / variance
    return beta

# Function to calculate alpha
def calculate_alpha(beta, combined_data, risk_free_rate=0.02/252):  # Assume annual risk-free rate of 2%, divide by 252 for daily
    # Average return for stock and benchmark
    avg_stock_return = combined_data['Return_stock'].mean()
    avg_benchmark_return = combined_data['Return_benchmark'].mean()
    
    # CAPM predicted return
    capm_expected_return = risk_free_rate + beta * (avg_benchmark_return - risk_free_rate)
    
    # Alpha
    alpha = avg_stock_return - capm_expected_return
    return alpha

# Function to calculate error (residuals)
def calculate_error(beta, combined_data, risk_free_rate=0.02/252):
    # CAPM expected return for each day
    expected_return = risk_free_rate + beta * (combined_data['Return_benchmark'] - risk_free_rate)
    
    # Calculate residuals (errors)
    combined_data['Error'] = combined_data['Return_stock'] - expected_return
    return combined_data['Error']

# Function to calculate idiosyncratic volatility
def calculate_idiosyncratic_volatility(errors):
    idio_volatility = np.std(errors)  # Standard deviation of residuals (errors)
    return idio_volatility

# Function to calculate average return
def calculate_average_return(combined_data):
    avg_return = combined_data['Return_stock'].mean()  # Mean return
    return avg_return

# Function to calculate all metrics (alpha, beta, error, idiosyncratic volatility, return)
def calculate_all_metrics(stock_ticker, benchmark_ticker='^GSPC', start_date='2020-01-01', end_date='2023-01-01', risk_free_rate=0.02/252):
    #Combine data
    combined_data = combine_data(stock_ticker, benchmark_ticker, start_date, end_date)

    # Calculate beta
    beta = calculate_beta(combined_data)
    
    # Calculate alpha
    alpha = calculate_alpha(beta, combined_data, risk_free_rate)
    
    # Calculate errors (residuals)
    errors = calculate_error(beta, combined_data, risk_free_rate)
    
    # Calculate idiosyncratic volatility
    idio_volatility = calculate_idiosyncratic_volatility(errors)
    
    # Calculate average return
    avg_return = calculate_average_return(combined_data)
    
    # Print results
    print(f"Metrics for {stock_ticker} (vs. {benchmark_ticker}):")
    print(f"Beta: {beta}")
    print(f"Alpha: {alpha}")
    print(f"Idiosyncratic Volatility: {idio_volatility}")
    print(f"Average Return: {avg_return}")