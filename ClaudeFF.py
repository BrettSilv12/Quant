import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import statsmodels.api as sm
from scipy import stats
from typing import List, Tuple, Dict, Union
from getAllMarketCap import getMarketCapPercentile

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
        # Placeholder for Fama-French factor data sources
        # You would replace these with actual data retrieval methods

        self.calculator = MarketCapPercentileCalculator()

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
        
        print(f"MCap: {market_cap}\nEVal: {enterprise_value}\n")
        # Log transformation to handle exponential distribution of market caps
        log_market_cap = np.log(market_cap) if market_cap > 0 else np.nan
        log_enterprise_value = np.log(enterprise_value) if enterprise_value > 0 else np.nan
        print(f"LOGMCap: {log_market_cap}\nLOGEVal: {log_enterprise_value}\n")
        # Compute percentile rank in market
        # In practice, this would compare against a comprehensive market database
        size_percentile = stats.percentileofscore([
            0,
            2000000000, #small-mid cap limit
            10000000000, #mid-high cap limit
            3620000000000 #Nvidia highest cap
        ], ((market_cap + enterprise_value) / 2))
        #Beta: Use module to calculate total market size
        #size_percentile = getMarketCapPercentile(market_cap, self.calculator)
        print(f"SizePercentile: {size_percentile}\n")
        
        # Size factor: lower percentile means smaller company
        size_factor = (50 - size_percentile) / 50
        
        return size_factor

    def _calculate_value_factor(self, ticker: str) -> float:
        """
        Advanced Value Factor (High vs Low Book-to-Market)
        
        :param ticker: Stock ticker symbol
        :return: Value characteristic score
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
        
        # Filter out NaN values
        valid_metrics = []
        for v in value_metrics.values():
            if not np.isnan(v):
                valid_metrics.append(v)
            else:
                valid_metrics.append(0)
        #valid_metrics = [v for v in value_metrics.values() if not np.isnan(v) else 0]
        
        if not valid_metrics:
            return np.nan
        
        # Z-score normalization of value metrics
        z_scores = stats.zscore(valid_metrics)
        
        # Combine z-scores with custom weighting
        # Reward stocks with high book-to-market and low price multiples
        value_factor = np.mean([
            z_scores[0],  # Book to Market
            -z_scores[1],  # Inverse of Price to Book (lower is better)
            -z_scores[2],  # Inverse of Price to Earnings (lower is better)
            -z_scores[3]   # Inverse of Price to Sales (lower is better)
        ])
        return value_factor

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
            return np.nan
        
        # Multiple profitability metrics
        metrics = {
            'Operating Margin': income_statements.loc['Operating Income'],
            'Net Profit Margin': income_statements.loc['Net Income'],
            'Return on Equity': stock.info.get('returnOnEquity', np.nan),
            'Return on Assets': stock.info.get('returnOnAssets', np.nan)
        }
        
        # Compute percentage change and remove outliers
        profitability_series = []
        for name, data in metrics.items():
            if isinstance(data, pd.Series):
                pct_changes = data.pct_change()
                # Winsorize to remove extreme values
                profitability_series.extend(stats.mstats.winsorize(pct_changes))
        
        # Compute robust profitability score
        if profitability_series:
            robust_profitability = np.mean(profitability_series)
            profitability_volatility = np.std(profitability_series)
            
            # Advanced scoring: reward consistent, high profitability
            profitability_factor = (
                robust_profitability * 0.6 +  # Primary profitability
                (-profitability_volatility) * 0.4  # Penalize inconsistency
            )
            
            return profitability_factor
        
        return np.nan


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
            return np.nan
        
        # Calculate asset growth rates
        total_assets = balance_sheets.loc['Total Assets']
        asset_growth_rates = total_assets.pct_change()
        
        # Robust growth rate estimation
        # Use median growth rate to reduce impact of outliers
        median_growth = np.median(asset_growth_rates)
        
        # Variation analysis
        growth_volatility = np.std(asset_growth_rates)
        
        # Incorporate investment consistency
        consistency_score = 1 - stats.variation(asset_growth_rates)
        
        # Investment factor: combine growth, volatility, and consistency
        investment_factor = (
            median_growth * 0.5 +  # Directional growth
            (-growth_volatility) * 0.3 +  # Penalize volatility
            consistency_score * 0.2  # Reward consistency
        )
        
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
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'PLNT', 'UPS', 'TSLA']
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