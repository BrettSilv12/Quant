#Quant
This repo contains loose files used for Quant analysis


<h1>ClaudeFF.py</h1>
This file takes in a list of tickers and evaluates them using the criteria of the FF 5-factor model.</br>


<h1>portfolioReturns.py</h1>
This file takes in a date and a map of stock tickers to ratios, then returns the market return for the given portfolio starting at the given date. </br>


<h1>TotalAnalysis.py</h1>
This file is a class definition that will automatically turn a given ticker into a full FF analysis. </br>
It gets raw data for each of the five factors, as well as calculating a score, before storing it all in the ticker-specific class</br>


<h1>experimentalFamaFrench.py</h1>
This file takes in a list of stock tickers, then runs a linear regression between the tickers' returns and market 5-factor analyses from French's website to define the behavior of a given stock. </br>
It outputs:</br>
Market-beta: stock movement relative to market movement (smaller value = lower market-relative movement),</br>
Size: Small Minus Big - the higher the score, the more this stock behaves like a small-cap, </br>
Value: High Minus Low - the higher the score, the more this stock behaves like a value stock, </br>
Profitability: the higher the score, the more profitable the company is, </br>
Investment: somewhat negligible due to profitability axis - the more positive the score, the more conservatively they reinvest </br>
Scores - somewhat arbitrary - used to test different weights on each factor </br>
The linear regression is run as an experimental exploration of these factors compared to the stock's returns. 