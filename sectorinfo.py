import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas_datareader as pdr
import time

sector_etfs = {
    'Technology': [
        'XLK',    # Technology Select Sector SPDR Fund
        'VGT',    # Vanguard Information Technology ETF
        'IYW'     # iShares U.S. Technology ETF
    ],
    'Healthcare': [
        'XLV',    # Health Care Select Sector SPDR Fund
        'VHT',    # Vanguard Health Care ETF
        'IYH'     # iShares U.S. Healthcare ETF
    ],
    'Financial': [
        'XLF',    # Financial Select Sector SPDR Fund
        'VFH',    # Vanguard Financials ETF
        'IYF'     # iShares U.S. Financials ETF
    ],
    'Energy': [
        'XLE',    # Energy Select Sector SPDR Fund
        'VDE',    # Vanguard Energy ETF
        'IYE'     # iShares U.S. Energy ETF
    ],
    'Consumer Discretionary': [
        'XLY',    # Consumer Discretionary Select Sector SPDR Fund
        'VCR',    # Vanguard Consumer Discretionary ETF
        'IYC'     # iShares U.S. Consumer Services ETF
    ],
    'Consumer Staples': [
        'XLP',    # Consumer Staples Select Sector SPDR Fund
        'VDC',    # Vanguard Consumer Staples ETF
        'IYK'     # iShares U.S. Consumer Goods ETF
    ],
    'Industrials': [
        'XLI',    # Industrial Select Sector SPDR Fund
        'VIS',    # Vanguard Industrials ETF
        'IYJ'     # iShares U.S. Industrials ETF
    ],
    'Materials': [
        'XLB',    # Materials Select Sector SPDR Fund
        'VAW',    # Vanguard Materials ETF
        'IYM'     # iShares U.S. Materials ETF
    ],
    'Utilities': [
        'XLU',    # Utilities Select Sector SPDR Fund
        'VPU',    # Vanguard Utilities ETF
        'IDU'     # iShares U.S. Utilities ETF
    ],
    'Real Estate': [
        'XLRE',   # Real Estate Select Sector SPDR Fund
        'VNQ',    # Vanguard Real Estate ETF
        'IYR'     # iShares U.S. Real Estate ETF
    ],
    'Communication Services': [
        'XLC',    # Communication Services Select Sector SPDR Fund
        'VOX',    # Vanguard Communication Services ETF
        'IYZ'     # iShares U.S. Telecommunications ETF
    ],
    'Aerospace & Defense': [
        'ITA',    # iShares U.S. Aerospace & Defense ETF
        'PPA',    # Invesco Aerospace & Defense ETF
        'XAR'     # SPDR S&P Aerospace & Defense ETF
    ],
    'Biotechnology': [
        'IBB',    # iShares Biotechnology ETF
        'XBI',    # SPDR S&P Biotech ETF
        'FBT'     # First Trust NYSE Arca Biotechnology ETF
    ],
    'Semiconductors': [
        'SMH',    # VanEck Semiconductor ETF
        'SOXX',   # iShares Semiconductor ETF
        'NVDA'    # NVIDIA ETF (Roundhill)
    ],
    'Banking': [
        'KBE',    # SPDR S&P Bank ETF
        'IAT',    # iShares U.S. Regional Banks ETF
        'VFG'     # Vanguard Financials ETF
    ],
    'Retail': [
        'XRT',    # SPDR S&P Retail ETF
        'VDC',    # Vanguard Consumer Staples ETF
        'IBUY'    # Amplify Online Retail ETF
    ],
    'Insurance': [
        'IAK',    # iShares U.S. Insurance ETF
        'KIE',    # SPDR S&P Insurance ETF
        'KBWI'    # Invesco KBW Insurance ETF
    ],
    'Transportation': [
        'IYT',    # iShares Transportation Average ETF
        'XTN',    # SPDR S&P Transportation ETF
        'VTI'     # Vanguard Total Stock Market ETF
    ],
    'Pharmaceuticals': [
        'PJP',    # Invesco Dynamic Pharmaceuticals ETF
        'IHE',    # iShares U.S. Pharmaceutical ETF
        'XPH'     # SPDR S&P Pharmaceuticals ETF
    ],
   'Financial Services': [
       'XLF',    # Financial Select Sector SPDR Fund
       'VFH',    # Vanguard Financials ETF
       'IYF'     # iShares U.S. Financials ETF
   ],
   'Consumer Cyclical': [
       'XLY',    # Consumer Discretionary Select Sector SPDR Fund
       'VCR',    # Vanguard Consumer Discretionary ETF
       'IYC'     # iShares U.S. Consumer Services ETF
   ],
   'Consumer Defensive': [
       'XLP',    # Consumer Staples Select Sector SPDR Fund
       'VDC',    # Vanguard Consumer Staples ETF
       'IYK'     # iShares U.S. Consumer Goods ETF
   ],
   'Basic Materials': [
       'XLB',    # Materials Select Sector SPDR Fund
       'VAW',    # Vanguard Materials ETF
       'IYM'     # iShares U.S. Materials ETF
   ]
}

def find_sector_etfs(sector):
    # Normalize sector name
    normalized_sector = sector.title()
    
    # Return ETFs if found, otherwise return None
    return sector_etfs.get(normalized_sector, None)

def get_sector_pb(sector):
    etf_list = find_sector_etfs(sector)
    avg = 0
    for etf_ticker in etf_list:
        etf = yf.Ticker(etf_ticker)
        try:
            holdings = etf.get_funds_data().top_holdings.index.to_list()
        except:
            print("Could not retrieve ETF holdings")
            continue
        pb_ratios = get_pb_of_holdings(holdings)
        avg += pb_ratios["average_pb"]
    return avg / len(etf_list)

def get_pb_of_holdings(holdings):
    pb_ratios = []
    for ticker in holdings[:100]:  # Limit to top 100 holdings to avoid API rate limits
        try:
            stock = yf.Ticker(ticker)
            pb = stock.info.get('priceToBook')
            if pb and pb > 0:
                pb_ratios.append(pb)
        except Exception as e:
            print(f"Could not fetch P/B for {ticker}: {e}")
    
    # Calculate average P/B ratio
    if pb_ratios:
        return {
            'average_pb': np.mean(pb_ratios),
            'median_pb': np.median(pb_ratios),
            'pb_count': len(pb_ratios)
        }
    else:
        return None
    
def get_sp500_pb_ratio():
    # Fetch S&P 500 components (using VOO as a representative ETF)
    sp500_etf = yf.Ticker('VOO')
    
    # Collect P/B ratios
    pb_ratios = []
    
    try:
        # Get holdings
        holdings = sp500_etf.get_funds_data().top_holdings.index.to_list()
        
        if not holdings:
            # Fallback method: use manual list of top S&P 500 stocks
            top_sp500_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 
                'GOOG', 'BRK-B', 'UNH', 'XOM', 'JPM', 'JNJ', 'V', 'PG'
            ]
            holdings = top_sp500_tickers
        
        # Collect P/B ratios
        for ticker in holdings[:500]:  # Limit to avoid API rate limits
            try:
                stock = yf.Ticker(ticker)
                pb = stock.info.get('priceToBook')
                if pb and pb > 0:
                    pb_ratios.append(pb)
            except Exception as e:
                print(f"Could not fetch P/B for {ticker}: {e}")
        
        # Calculate statistics
        if pb_ratios:
            return {
                'average_pb': np.mean(pb_ratios),
                'median_pb': np.median(pb_ratios),
                'pb_count': len(pb_ratios),
                'min_pb': min(pb_ratios),
                'max_pb': max(pb_ratios)
            }
        else:
            print("No P/B ratios could be retrieved")
            return None
    
    except Exception as e:
        print(f"Error retrieving S&P 500 holdings: {e}")
        return None