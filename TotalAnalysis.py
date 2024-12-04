import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from datetime import date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class StockFF:
    def __init__(self, ticker):
        self.ticker = ticker
        self.yfticker = yf.Ticker(ticker)
        self.value = {None}
        self.size = {None}
        self.profitability = {None}
        self.investment = {None}
        self.MBeta = None
        self.value_score = None
        self.size_score = None
        self.profitability_score = None
        self.investment_score = None
        self.getFactorAttributes()
        self.evaluateFactorScores()

    def getFactorAttributes(self):
        self.value = self.getValueFactor()
        self.size = self.getSizeFactor()
        self.profitability = self.getProfitabilityFactor()
        self.investment = self.getInvestmentFactor()
        self.MBeta = self.calculateMarketBeta()
    
    def evaluateFactorScores(self):
        self.value_score = None
        self.size_score = None
        self.profitability_score = None
        self.investment_score = None
        
    def rawData(self):
        return {
            'Value': self.value,
            'Size' : self.size,
            'Profitability' : self.profitability,
            'Investment'    : self.investment,
            'Market Beta'   : self.MBeta
        }
    
    def scoredData(self):
        return {
            'Value Score': self.value_score,
            'Size Score' : self.size_score,
            'Profitability Score' : self.profitability_score,
            'Investment Score'    : self.investment_score,
            'Market Beta Score'   : self.MBeta
        }
    


    def getValueFactor(self):
        value_metrics = {
            'Book to Market': self.yfticker.info.get('bookValue', np.nan) / 
                              (self.yfticker.info.get('marketCap', 1) / self.yfticker.info.get('sharesOutstanding', 1)),
            'Price to Book': self.yfticker.info.get('priceToBook', np.nan),
            'Price to Earnings': self.yfticker.info.get('trailingPE', np.nan),
            'Price to Sales': self.yfticker.info.get('priceToSalesTrailing12Months', np.nan)
        }
        return value_metrics
    


    def getSizeFactor(self):
        market_cap = self.yfticker.info.get('marketCap', np.nan)
        
        # Compute enterprise value as alternative size metric
        #   m_cap + freecashflow
        try:
            enterprise_value = (
                market_cap + 
                self.yfticker.info.get('totalDebt', 0) - 
                self.yfticker.info.get('totalCash', 0)
            )
        except:
            enterprise_value = market_cap
        return {
            'Market Cap': market_cap,
            'Enterprise Value': enterprise_value
        } 
    


    def getProfitabilityFactor(self):
        income_statements = self.yfticker.financials
        
        if income_statements.empty:
            print("EMPTY INCOME STATEMENT")
            return np.nan
            
        # Multiple profitability metrics
        metrics = {
            'Operating Income': income_statements.loc['Operating Income'],
            'Net Income': income_statements.loc['Net Income'],
            'Return on Equity': self.yfticker.info.get('returnOnEquity', np.nan),
            'Return on Assets': self.yfticker.info.get('returnOnAssets', np.nan)
        }
        return metrics
    


    def getInvestmentFactor(self):
        # Retrieve historical balance sheets
        balance_sheets = self.yfticker.balance_sheet
        
        # Validate data availability
        if balance_sheets.empty:
            print("BALANCE SHEET EMPTY")
            return np.nan
        
        # Calculate asset growth rates
        total_assets = balance_sheets.loc['Total Assets'] 
        total_assets = total_assets.iloc[::-1] # reverses order
        asset_growth_rates = total_assets.pct_change()
        
        # Robust growth rate estimation
        mean_growth = asset_growth_rates.mean()

        return {
            'Total Assets': total_assets,
            'Asset Growth': asset_growth_rates,
            'Mean Growth' : mean_growth
        }
    


    def calculateMarketBeta(self, end_date: str = date.today().strftime("%Y-%m-%d")) -> float:
        """
        Advanced Market Beta Calculation using Rolling Regression
        
        :param ticker: Stock ticker symbol
        :param start_date: Start date for regression
        :param end_date: End date for regression
        :return: Robust market beta estimate
        """
        ticker = self.ticker
        original_date = end_date
        # Subtract 5 months
        five_months_prior = original_date - relativedelta(months=5)
        # Convert back to YYYY-MM-DD string format
        five_months_prior = five_months_prior.strftime("%Y-%m-%d")
        start_date = five_months_prior

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
        
        return -1 * max(-2, min(2, shrunk_beta)) #inverse sign, because a higher score now means less volatile