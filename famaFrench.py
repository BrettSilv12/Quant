import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FamaFrenchAnalyzer:
    def __init__(self):
        """Initialize the Fama-French analyzer with factor data"""
        self.ff_factors = None
        self.load_ff_factors()

    def load_ff_factors(self):
        """
        Load Fama-French 5 factors data from Kenneth French's website
        """
        # URL for Fama-French 5 factors daily data
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        
        try:
            # Read the CSV file directly from the zip
            ff_factors = pd.read_csv(url, skiprows=3)
            
            # Clean up the data
            ff_factors.columns = ['Date', 'MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            
            # Convert date to datetime
            ff_factors['Date'] = pd.to_datetime(ff_factors['Date'].astype(str), format='%Y%m%d')
            
            # Set date as index
            ff_factors.set_index('Date', inplace=True)
            
            # Convert percentage values
            for col in ff_factors.columns:
                ff_factors[col] = ff_factors[col] / 100.0
            
            self.ff_factors = ff_factors
            
        except Exception as e:
            raise Exception(f"Error loading Fama-French factors: {str(e)}")

    def get_stock_returns(self, ticker, start_date, end_date):
        """
        Calculate daily returns for a given stock
        """
        try:
            # Download stock data
            stock = yf.Ticker(ticker)
            stock_data = stock.history(start=start_date, end=end_date)
            
            if stock_data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Calculate daily returns
            stock_returns = stock_data['Close'].pct_change().dropna()
            stock_returns.index = pd.to_datetime(stock_returns.index.date)
            
            return stock_returns
            
        except Exception as e:
            raise Exception(f"Error processing {ticker}: {str(e)}")

    def run_factor_regression(self, stock_returns, factor_data):
        """
        Run regression analysis for Fama-French 5 factors
        """
        # Align dates between stock returns and factors
        common_dates = stock_returns.index.intersection(factor_data.index)
        if len(common_dates) < 30:  # Require at least 30 days of data
            raise ValueError("Insufficient overlapping data points")
            
        stock_returns = stock_returns[common_dates]
        factors = factor_data.loc[common_dates]
        
        # Calculate excess returns
        excess_returns = stock_returns - factors['RF']
        
        # Prepare factors for regression
        X = factors[['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X)
        
        # Run regression
        model = sm.OLS(excess_returns, X).fit()
        
        return model

    def analyze_stocks(self, tickers, lookback_days=252):
        """
        Analyze multiple stocks using Fama-French 5-factor model
        
        Parameters:
        tickers (list): List of stock tickers
        lookback_days (int): Number of trading days to analyze (default: 252 ≈ 1 year)
        
        Returns:
        dict: Dictionary containing factor analysis results for each stock
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)  # Add buffer for non-trading days
        
        results = {}
        
        for ticker in tickers:
            try:
                # Get stock returns
                stock_returns = self.get_stock_returns(ticker, start_date, end_date)
                
                # Run factor regression
                model = self.run_factor_regression(stock_returns, self.ff_factors)
                
                # Calculate R-squared and adjusted R-squared
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj
                
                # Store results
                results[ticker] = {
                    'coefficients': {
                        'alpha': model.params['const'],
                        'market_beta': model.params['MKT-RF'],
                        'size': model.params['SMB'],
                        'value': model.params['HML'],
                        'profitability': model.params['RMW'],
                        'investment': model.params['CMA']
                    },
                    'p_values': {
                        'alpha': model.pvalues['const'],
                        'market_beta': model.pvalues['MKT-RF'],
                        'size': model.pvalues['SMB'],
                        'value': model.pvalues['HML'],
                        'profitability': model.pvalues['RMW'],
                        'investment': model.pvalues['CMA']
                    },
                    'model_fit': {
                        'r_squared': r_squared,
                        'adj_r_squared': adj_r_squared
                    },
                    'factor_scores': {
                        'market_score': self._calculate_factor_score(model.params['MKT-RF'], model.pvalues['MKT-RF']),
                        'size_score': self._calculate_factor_score(model.params['SMB'], model.pvalues['SMB']),
                        'value_score': self._calculate_factor_score(model.params['HML'], model.pvalues['HML']),
                        'profitability_score': self._calculate_factor_score(model.params['RMW'], model.pvalues['RMW']),
                        'investment_score': self._calculate_factor_score(model.params['CMA'], model.pvalues['CMA'])
                    }
                }
                
            except Exception as e:
                results[ticker] = {'error': str(e)}
                
        return results

    def _calculate_factor_score(self, coefficient, p_value, significance_threshold=0.05):
        """
        Calculate a score for each factor based on coefficient and statistical significance
        """
        if p_value > significance_threshold:
            return 0  # Not statistically significant
        
        # Normalize coefficient to a -1 to 1 scale
        return np.clip(coefficient, -1, 1)

    def format_results(self, results):
        """
        Format the analysis results into a readable string
        """
        output = "\nFama-French 5-Factor Analysis Results:\n"
        output += "=" * 50 + "\n\n"
        
        for ticker, result in results.items():
            output += f"{ticker}:\n"
            output += "-" * 30 + "\n"
            
            if 'error' in result:
                output += f"Error: {result['error']}\n\n"
                continue
            
            output += "Factor Coefficients (t-stats):\n"
            for factor, value in result['coefficients'].items():
                p_value = result['p_values'][factor]
                stars = '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
                output += f"  {factor}: {value:.4f} {stars}\n"
            
            output += "\nFactor Scores (-1 to 1 scale):\n"
            for factor, score in result['factor_scores'].items():
                output += f"  {factor}: {score:.4f}\n"
            
            output += f"\nModel Fit:\n"
            output += f"  R-squared: {result['model_fit']['r_squared']:.4f}\n"
            output += f"  Adjusted R-squared: {result['model_fit']['adj_r_squared']:.4f}\n\n"
            
        return output

    def create_factor_comparison_plot(self, results):
        """
        Create an interactive radar chart comparing factor exposures across stocks
        
        Parameters:
        results (dict): Results dictionary from analyze_stocks()
        
        Returns:
        plotly.graph_objects.Figure: Interactive plot comparing factor exposures
        """
        # Prepare data for plotting
        categories = ['Market Beta', 'Size', 'Value', 'Profitability', 'Investment']
        
        fig = go.Figure()
        
        for ticker, result in results.items():
            if 'error' in result:
                continue
                
            # Extract factor scores
            scores = [
                result['coefficients']['market_beta'],
                result['coefficients']['size'],
                result['coefficients']['value'],
                result['coefficients']['profitability'],
                result['coefficients']['investment']
            ]
            
            # Add a trace for each stock
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                name=ticker,
                fill='toself',
                opacity=0.6
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]  # Adjust range based on your data
                )
            ),
            showlegend=True,
            title="Factor Exposure Comparison",
            title_x=0.5,
            title_y=0.95,
            title_font_size=20
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = FamaFrenchAnalyzer()
    
    # Example stock tickers
    tickers = ['UPS', 'NVDA', 'PLTR']
    
    try:
        # Run analysis
        results = analyzer.analyze_stocks(tickers)
        
        # Print formatted results
        print(analyzer.format_results(results))
        
        # Create and display the comparison plot
        fig = analyzer.create_factor_comparison_plot(results)
        fig.show()  # This will open the plot in your default browser
        
    except Exception as e:
        print(f"Error: {str(e)}")

"""
LINEAR REGRESSION CHEAT SHEET:


Market Beta (MKT-RF):

If beta = 1: The stock moves exactly with the market
If beta = 2: The stock moves twice as much as the market
If beta = 0.5: The stock moves half as much as the market


Size (SMB - Small Minus Big):

Positive score: Stock behaves more like small-cap companies
Negative score: Stock behaves more like large-cap companies


Value (HML - High Minus Low):

Positive score: Stock shows characteristics of value stocks (high book-to-market ratio)
Negative score: Stock shows characteristics of growth stocks (low book-to-market ratio)


Profitability (RMW - Robust Minus Weak):

Positive score: Stock behaves like highly profitable companies
Negative score: Stock behaves like less profitable companies


Investment (CMA - Conservative Minus Aggressive):

Positive score: Stock behaves like companies that invest conservatively
Negative score: Stock behaves like companies that invest aggressively

The R-squared value (ranges from 0 to 1) tells us how well these factors explain the stock's behavior:

R-squared = 0.7 means 70% of the stock's price movements can be explained by these factors
The remaining 30% would be due to company-specific factors

Think of it like a recipe - the regression tells us how much of each "ingredient" (factor) 
contributes to the stock's behavior, and the R-squared tells us how good our recipe is at 
explaining the final "dish" (stock returns).
"""