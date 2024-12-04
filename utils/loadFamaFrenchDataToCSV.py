import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import statsmodels.api as sm
from scipy import stats
from typing import List, Tuple, Dict, Union
from datetime import datetime

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
        if col != 'Date':
            ff_factors[col] = ff_factors[col] / 100.0



    sp500 = yf.download('^GSPC', start=datetime.strptime('1985-01-01', '%Y-%m-%d'), end=datetime.now())
    
    # Calculate daily average (using (High + Low) / 2)
    sp500['DailyAverage'] = (sp500['High'] + sp500['Low']) / 2
    
    # Create new DataFrame with just the date index and sp500 column
    result_df = pd.DataFrame({
        'sp500': sp500['DailyAverage']
    })
    
    # Ensure the index is a DateTimeIndex
    result_df.index = pd.to_datetime(result_df.index).tz_localize(None)
    
    # Sort by date
    result_df.sort_index(inplace=True)

    df_combined = ff_factors.combine_first(result_df)

except Exception as e:
    raise Exception(f"Error loading Fama-French factors: {str(e)}")
# Placeholder for Fama-French factor data sources
# You would replace these with actual data retrieval methods

df_combined.to_csv("FamaFrenchData.csv")