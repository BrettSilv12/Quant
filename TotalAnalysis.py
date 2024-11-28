"""
    Get dict for each of 5 factors:
        1. Value: {}
        2. Size: {}
        3. Profitability: {}
        4. Investment: {}
        5. Market Beta: how it's done in Claude FF is fine - get rolling beta over time
    
    Use dict to evaluate a score:
        1. Value:
        2. Size:
        3. Profitability:
        4. Investment:
        5. Market Beta:
"""
import pandas as pd
import numpy as np
import yfinance as yf

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

    def getFactorAttributes(self):
        self.value = {None}
        self.size = {None}
        self.profitability = {None}
        self.investment = {None}
        self.MBeta = None
    
    def evaluateFactorScores(self):
        self.value_score = None
        self.size_score = None
        self.profitability_score = None
        self.investment_score = None
        
