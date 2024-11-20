import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_market_factor(stock_data):
    """Calculate market factor (size of the company based on market cap)"""
    market_cap = stock_data.info.get('marketCap', 0)
    if market_cap > 10e9:  # Large-cap
        return 1
    elif market_cap > 2e9:  # Mid-cap
        return 0
    else:  # Small-cap
        return -1

def calculate_size_factor(stock_data):
    """Calculate SMB (Small Minus Big) factor"""
    market_cap = stock_data.info.get('marketCap', 0)
    # Using natural log of market cap to smooth the values
    return -1 * (np.log(market_cap) if market_cap > 0 else 0)

def calculate_value_factor(stock_data):
    """Calculate HML (High Minus Low) factor using Book-to-Market ratio"""
    book_value = stock_data.info.get('bookValue', 0)
    market_price = stock_data.info.get('regularMarketPrice', 0)
    shares_outstanding = stock_data.info.get('sharesOutstanding', 0)
    
    if market_price and shares_outstanding:
        market_value = market_price * shares_outstanding
        book_to_market = book_value / market_value if market_value != 0 else 0
        return book_to_market
    return 0

def calculate_profitability_factor(stock_data):
    """Calculate RMW (Robust Minus Weak) factor using Operating Profitability"""
    revenue = stock_data.info.get('totalRevenue', 0)
    operating_income = stock_data.info.get('operatingIncome', 0)
    
    if revenue:
        return operating_income / revenue if revenue != 0 else 0
    return 0

def calculate_investment_factor(stock_data):
    """Calculate CMA (Conservative Minus Aggressive) factor using Asset Growth"""
    total_assets = stock_data.info.get('totalAssets', 0)
    prev_total_assets = stock_data.quarterly_balance_sheet.iloc[:, 1].get('Total Assets', 0)
    
    if prev_total_assets:
        asset_growth = (total_assets - prev_total_assets) / prev_total_assets if prev_total_assets != 0 else 0
        # Invert so conservative (low growth) gets positive score
        return -1 * asset_growth
    return 0

def get_factor_values(tickers, date):
    """
    Get Fama-French 5 factor values for given stocks at a specific date
    
    Parameters:
    tickers (list): List of stock tickers
    date (str): Date in 'YYYY-MM-DD' format
    
    Returns:
    dict: Dictionary mapping tickers to their factor values
    """
    results = {}
    
    for ticker in tickers:
        try:
            # Download stock data
            stock = yf.Ticker(ticker)
            
            # Calculate all factors
            factors = {
                'Market': calculate_market_factor(stock),
                'SMB': calculate_size_factor(stock),
                'HML': calculate_value_factor(stock),
                'RMW': calculate_profitability_factor(stock),
                'CMA': calculate_investment_factor(stock)
            }
            
            # Normalize factors to be between -1 and 1
            max_abs = max(abs(min(factors.values())), abs(max(factors.values())))
            if max_abs > 0:
                factors = {k: v/max_abs for k, v in factors.items()}
            
            results[ticker] = factors
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            results[ticker] = None
    
    return results

def create_radar_plot(factor_values, save_path=None):
    """
    Create a radar plot for the factor values
    
    Parameters:
    factor_values (dict): Dictionary of tickers and their factor values
    save_path (str, optional): Path to save the plot. If None, displays the plot
    """
    # Set up the angles of the radar plot
    categories = ['Market', 'SMB', 'HML', 'RMW', 'CMA']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data for each stock
    colors = plt.cm.rainbow(np.linspace(0, 1, len(factor_values)))
    for (ticker, factors), color in zip(factor_values.items(), colors):
        if factors is not None:
            values = [factors[cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=ticker, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title("Fama-French 5-Factor Comparison", y=1.08)
    
    # Add gridlines
    ax.set_ylim(-1, 1)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example stocks
    sample_tickers = ['AAPL', 'UPS', 'MSFT', 'PLNT']
    evaluation_date = '2023-01-01'
    
    # Get factor values
    results = get_factor_values(sample_tickers, evaluation_date)
    
    # Print numerical results
    for ticker, factors in results.items():
        if factors is not None:
            print(f"\nFactor values for {ticker}:")
            for factor, value in factors.items():
                print(f"{factor}: {value:.4f}")
        else:
            print(f"\nNo results available for {ticker}")
    
    # Create radar plot
    create_radar_plot(results)