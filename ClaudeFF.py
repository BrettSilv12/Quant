import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import statsmodels.api as sm
import sectorinfo as si
from scipy import stats
from typing import List, Tuple, Dict, Union

class FactorScoreAnalyzer:
    def __init__(self):
        """
        Initialize the Factor Score Analyzer with Fama-French data sources
        Note: In a production environment, you'd want to use a more robust 
        financial data API or database for factor calculations
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
            ff_factors.set_index('Date', inplace=False)
            
            # Convert percentage values
            for col in ff_factors.columns:
                if col != 'Date':
                    ff_factors[col] = ff_factors[col] / 100.0
            
            # Store factors
            self.ff_factors = {
                'Date': ff_factors['Date'],
                'SMB': ff_factors['SMB'],  # Size Factor
                'HML': ff_factors['HML'],  # Value Factor
                'RMW': ff_factors['RMW'],  # Profitability Factor
                'CMA': ff_factors['CMA'],  # Investment Factor
                'Rf': ff_factors['RF'],   # Risk-free Rate
                'Market_Rf': ff_factors['MKT-RF']  # Market Risk Premium
            }

        except Exception as e:
            raise Exception(f"Error loading Fama-French factors: {str(e)}")

    def fetch_stock_data(self, tickers: List[str], date: str) -> pd.DataFrame:
        """
        Fetch stock data for given tickers on the specified date
        
        :param tickers: List of stock ticker symbols
        :param date: Date for factor calculation
        :return: DataFrame with stock characteristics
        """
        stock_data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Fetch historical data around the specified date
                hist = stock.history(start=date, end=pd.to_datetime(date) + pd.Timedelta(days=30))
                
                # Calculate key metrics
                stock_data[ticker] = {
                    'Market Cap': hist['Close'].iloc[0] * stock.info.get('sharesOutstanding', 0),
                    'Book to Market': stock.info.get('bookValue', 0) / (hist['Close'].iloc[0] * stock.info.get('sharesOutstanding', 1)),
                    'ROE': stock.info.get('returnOnEquity', 0),
                    'Total Asset Growth': self._calculate_asset_growth(stock)
                }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        return pd.DataFrame.from_dict(stock_data, orient='index')

    def _calculate_asset_growth(self, stock) -> float:
        """
        Calculate asset growth for a given stock
        
        :param stock: yfinance Ticker object
        :return: Asset growth percentage
        """
        # Placeholder implementation
        # In a real scenario, you'd fetch historical balance sheet data
        return stock.info.get('totalAssetsGrowth', 0)

    def calculate_factor_scores(self, tickers: List[str], date: str) -> pd.DataFrame:
        """
        Calculate Fama-French 5-factor scores for given tickers
        
        :param tickers: List of stock ticker symbols
        :param date: Date for factor calculation
        :return: DataFrame with factor scores
        """
        # Fetch stock data
        stock_data = self.fetch_stock_data(tickers, date)
        
        # Calculate factor scores
        factor_scores = {}
        for ticker, row in stock_data.iterrows():
            factor_scores[ticker] = {
                'Size_SMB': self._calculate_size_factor(ticker),
                'Value_HML': self._calculate_value_factor(ticker),
                'Profitability_RMW': self._calculate_profitability_factor(ticker),
                'Investment_CMA': self._calculate_investment_factor(ticker),
                'Market_Beta': self._calculate_market_beta(ticker, date, '2024-10-01')
            }
        
        return pd.DataFrame.from_dict(factor_scores, orient='index')

    def _calculate_size_factor(self, ticker: str) -> float:
        """
        Advanced Size Factor (Small Minus Big Market Cap)
        
        :param ticker: Stock ticker symbol
        :return: Size characteristic score
        Ideally, the closer to max market cap, the closer to -2, and
        the closer to 0, the closer to 2
        """
        stock = yf.Ticker(ticker)
        
        # Market cap with multiple validation approaches
        market_cap = stock.info.get('marketCap', np.nan)
        
        # Compute enterprise value as alternative size metric
        #   m_cap + freecashflow
        try:
            enterprise_value = (
                market_cap + 
                stock.info.get('totalDebt', 0) - 
                stock.info.get('totalCash', 0)
            )
        except:
            enterprise_value = market_cap

        # Compute percentile rank in market
        size_percentile = stats.percentileofscore([
            2000000000, #small-mid cap limit
            10000000000, #mid-high cap limit
            3620000000000 #Nvidia highest cap
        ], ((market_cap + enterprise_value) / 2))
        if (((market_cap + enterprise_value) / 2) < 2000000000):
            size_percentile += 33 * ((market_cap + enterprise_value) / 2) / 2000000000
        elif (((market_cap + enterprise_value) / 2) < 10000000000):
            size_percentile += 33 * ((market_cap + enterprise_value) / 2) / 10000000000
        else:
            size_percentile += 33 * ((market_cap + enterprise_value) / 2) / 3620000000000
        print(f"SIZE PERCENTILE: {size_percentile}\n")
    
        
        # Size factor: lower percentile means smaller company
        size_factor = 2* (50 - size_percentile) / 50
        return max(-2, min(2, size_factor))
        

    def _calculate_value_factor(self, ticker: str) -> float:
        """
        Advanced Value Factor (High vs Low Book-to-Market)
        
        :param ticker: Stock ticker symbol
        :return: Value characteristic score
        IF P/B of a stock is 1, the value_score will be = 2
        IF P/B is > 55ish, the value_score will be -2
        """
        stock = yf.Ticker(ticker)
        
        # Comprehensive value metrics
        value_metrics = {
            'Book to Market': stock.info.get('bookValue', np.nan) / 
                              (stock.info.get('marketCap', 1) / stock.info.get('sharesOutstanding', 1)),
            'Price to Book': stock.info.get('priceToBook', np.nan),
            'Price to Earnings': stock.info.get('trailingPE', np.nan),
            'Price to Sales': stock.info.get('priceToSalesTrailing12Months', np.nan)
        }

        # Get PB of associated sector
        sector_pb = si.get_sector_pb(stock.info["sector"])

        # Get PB of sp500
        sp5_pb = si.get_sp500_pb_ratio()#['average_pb']

        sector_relative_pb = sector_pb - value_metrics['Price to Book']
        market_relative_pb = sp5_pb - value_metrics['Price to Book']
        value_score = sp5_pb - np.log(value_metrics['Price to Book']) - (sp5_pb - 2)
        return max(-2, min(value_score, 2)) #clip values to -2:2 range
        if(value_metrics['Price to Book'] < sp5_pb * .5):
            return 0.5
        #PB more growth than sector
        elif(value_metrics['Price to Book'] > sp5_pb * 1.25):
            return -0.5
        #PB close to sector
        else:
            return 0

    def _calculate_profitability_factor(self, ticker: str, lookback_years: int = 5) -> float:
        """
        Advanced Profitability Factor (Robust vs Weak Profitability)
        
        :param ticker: Stock ticker symbol
        :param lookback_years: Years of historical data to analyze
        :return: Profitability characteristic score
        """
        stock = yf.Ticker(ticker)
        
        # Retrieve income statements
        income_statements = stock.financials
        
        if income_statements.empty:
            print("EMPTY INCOME STATEMENT")
            return np.nan
            
        # Multiple profitability metrics
        metrics = {
            'Operating Income': income_statements.loc['Operating Income'],
            'Net Income': income_statements.loc['Net Income'],
            'Return on Equity': stock.info.get('returnOnEquity', np.nan),
            'Return on Assets': stock.info.get('returnOnAssets', np.nan)
        }
        #print(f"ROE: {metrics['Return on Equity']}\n ROA: {metrics['Return on Assets']}\n")
        #print(f"Op In: {metrics['Operating Income']}\n Net In: {metrics['Net Income']}\n")
        #ROE is more sensitive to a company's leverage, whereas ROA is not
        #ROE : Success of using leverage to generate profit
        #ROA : Efficiency of managing their assets
        #operating income: post operating-expense income - how well a company's core-business is running
        #net income: post-everything income / take-home. 
        #ASK MCKAY: 
        #1. How to accurately measure market cap percentile on the sliding scale
        #2. How to properly evaluate 2 variables above another (emphasize roa > roe but both over op. income)
        profitability_factor = metrics['Return on Assets'] + metrics['Return on Equity']
        return profitability_factor


    def _calculate_investment_factor(self, ticker: str, lookback_years: int = 5) -> float:
        """
        Advanced Investment Factor (Conservative vs Aggressive Investment)
        
        :param ticker: Stock ticker symbol
        :param lookback_years: Years of historical data to analyze
        :return: Investment characteristic score
        """
        # Fetch financial statements
        stock = yf.Ticker(ticker)
        
        # Retrieve historical balance sheets
        balance_sheets = stock.balance_sheet
        
        # Validate data availability
        if balance_sheets.empty:
            print("BALANCE SHEET EMPTY")
            return np.nan
        
        # Calculate asset growth rates
        print("1\n")
        total_assets = balance_sheets.loc['Total Assets'] # reverse order
        print(f"2 {total_assets}\n")
        asset_growth_rates = total_assets.pct_change()
        
        # Robust growth rate estimation
        print(f"3 {asset_growth_rates}\n")
        median_growth = np.mean(asset_growth_rates) # not properly getting a mean - just returning nan (values are ina map?)
        
        # Variation analysis
        print(f"4 {median_growth}\n")
        growth_volatility = np.std(asset_growth_rates)
        
        # Incorporate investment consistency
        print(f"5 {growth_volatility}\n")
        consistency_score = 1 - stats.variation(asset_growth_rates)
        
        # Investment factor: combine growth, volatility, and consistency
        print(f"6 {consistency_score}\n")
        investment_factor = (
            median_growth * 0.5 +  # Directional growth
            (-growth_volatility) * 0.3 +  # Penalize volatility
            consistency_score * 0.2  # Reward consistency
        )
        print(f"7 {investment_factor}\n")
        
        return investment_factor

    def _calculate_market_beta(self, ticker: str, start_date: str, end_date: str) -> float:
        """
        Advanced Market Beta Calculation using Rolling Regression
        
        :param ticker: Stock ticker symbol
        :param start_date: Start date for regression
        :param end_date: End date for regression
        :return: Robust market beta estimate
        """
        # Fetch stock and market returns
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        market_data = yf.download('^GSPC', start=start_date, end=end_date)  # S&P 500 as market proxy
        
        # Calculate excess returns
        stock_returns = stock_data['Adj Close'].pct_change().dropna()
        market_returns = market_data['Adj Close'].pct_change().dropna()
        
        # Prepare data for regression
        X = market_returns.values
        y = stock_returns.values
        X = sm.add_constant(X)
        
        # Perform robust regression (Huber Regression to minimize impact of outliers)
        huber_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        huber_results = huber_model.fit()
        
        # Extract beta and perform additional diagnostics
        beta = huber_results.params[1]
        
        # Confidence interval and significance testing
        conf_int = huber_results.conf_int()
        t_stat = huber_results.tvalues[1]
        p_value = huber_results.pvalues[1]
        
        # Advanced beta estimation: incorporate Bayesian shrinkage
        prior_beta = 1.0  # Market average beta
        shrinkage_factor = min(max(abs(t_stat), 0.1), 2)
        shrunk_beta = (shrinkage_factor * beta + prior_beta) / (1 + shrinkage_factor)
        
        return shrunk_beta

    def create_factor_radar_plot(self, factor_scores: pd.DataFrame) -> go.Figure:
        """
        Create a radar plot comparing factor scores across tickers
        
        :param factor_scores: DataFrame of factor scores
        :return: Plotly radar chart
        """
        # Prepare data for radar plot
        categories = ['Size', 'Value', 'Profitability', 'Investment', 'Market Beta']
        
        # Create radar plot
        fig = go.Figure()
        
        for ticker in factor_scores.index:
            fig.add_trace(go.Scatterpolar(
                r=factor_scores.loc[ticker].values,
                theta=categories,
                fill='toself',
                name=ticker
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2]
                )),
            showlegend=True,
            title='Fama-French 5-Factor Model Comparison'
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Example tickers and date
    tickers = ['MSFT', 'PLTR', 'UPS', 'TSLA', 'NVDA', 'NAUT']
    analysis_date = '2023-12-31'
    
    # Initialize analyzer
    analyzer = FactorScoreAnalyzer()
    
    # Calculate factor scores
    factor_scores = analyzer.calculate_factor_scores(tickers, analysis_date)
    
    # Generate radar plot
    radar_plot = analyzer.create_factor_radar_plot(factor_scores)
    
    # Optional: Save or display results
    print(factor_scores)
    radar_plot.show()



































"""
1. Market-RF: get market-rf and r-f (how succeptible is company to market movement)
2. Size: get percentile size of company to market
3. Value: get percentile value of company to market and sector
4. Profitability: get percentile profitability to sector
5. Investment: ignore for now

"""