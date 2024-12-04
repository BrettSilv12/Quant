import yfinance as yf
from datetime import datetime
import pandas as pd

def calculate_portfolio_return(start_date, portfolio_weights):
    """
    Calculate the return of a portfolio from start_date to current date.
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    portfolio_weights (dict): Dictionary mapping stock tickers to their weights
                            Example: {'AAPL': 0.4, 'GOOGL': 0.6}
    
    Returns:
    dict: Dictionary containing portfolio metrics including:
          - total_return: Overall portfolio return as a percentage
          - individual_returns: Dictionary of individual stock returns
          - portfolio_value: Final portfolio value assuming $100 initial investment
    """
    # Validate inputs
    try:
        start_date = pd.to_datetime(start_date)
        if start_date > datetime.now():
            raise ValueError("Start date cannot be in the future")
    except Exception as e:
        raise ValueError(f"Invalid date format. Please use 'YYYY-MM-DD'. Error: {str(e)}")
    
    if not isinstance(portfolio_weights, dict):
        raise ValueError("Portfolio weights must be a dictionary")
    
    if not abs(sum(portfolio_weights.values()) - 1.0) < 0.0001:
        raise ValueError("Portfolio weights must sum to 1")
    
    # Initialize results dictionary
    results = {
        'individual_returns': {},
        'total_return': 0,
        'portfolio_value': 0
    }
    
    # Calculate returns for each stock
    weighted_returns = 0
    for ticker, weight in portfolio_weights.items():
        try:
            # Download stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date)
            
            if hist.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Calculate return
            initial_price = hist['Close'].iloc[0]
            final_price = hist['Close'].iloc[-1]
            stock_return = ((final_price - initial_price) / initial_price) * 100
            
            # Store individual stock return
            results['individual_returns'][ticker] = {
                'return': stock_return,
                'initial_price': initial_price,
                'final_price': final_price,
                'weight': weight
            }
            
            # Add to weighted return
            weighted_returns += stock_return * weight
            
        except Exception as e:
            raise ValueError(f"Error processing {ticker}: {str(e)}")
    
    # Calculate final portfolio metrics
    results['total_return'] = weighted_returns
    results['portfolio_value'] = 100 * (1 + weighted_returns/100)  # Assuming $100 initial investment
    
    return results

def format_results(results):
    """Format the results into a readable string."""
    output = "\nPortfolio Performance Summary:\n"
    output += "=" * 30 + "\n\n"
    
    output += f"Total Portfolio Return: {results['total_return']:.2f}%\n"
    output += f"Final Portfolio Value (from $100): ${results['portfolio_value']:.2f}\n\n"
    
    output += "Individual Stock Performance:\n"
    output += "-" * 30 + "\n"
    for ticker, data in results['individual_returns'].items():
        output += f"{ticker} ({data['weight']*100:.1f}% weight):\n"
        output += f"  Return: {data['return']:.2f}%\n"
        output += f"  Price: ${data['initial_price']:.2f} → ${data['final_price']:.2f}\n\n"
    
    return output

# Example usage
if __name__ == "__main__":
    # Example portfolio: 40% AAPL, 30% GOOGL, 30% MSFT
    portfolio = {
        'UNP': 0.4,
        'UPS': 0.3,
        'COKE': 0.3
    }

    portfolioB = {
        'VOO': 1
    }

    portfolioC = {
        'COKE': .5,
        'UNP' : .5
    }
    
    try:
        results = calculate_portfolio_return('2016-01-01', portfolioC)
        print(format_results(results))
    except Exception as e:
        print(f"Error: {str(e)}")