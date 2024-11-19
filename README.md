#Quant
This repo contains loose files used for Quant analysis

<h1>famaFrench.py</h1>
This file takes in a list of stock tickers, then runs a linear regression between the tickers' returns and their scoring using a Fama-French 5-factor model. </br>
It outputs:</br>
Market-beta: stock movement relative to market movement (smaller value = lower market-relative movement),</br>
Size: Small Minus Big - the higher the score, the more this stock behaves like a small-cap, </br>
Value: High Minus Low - the higher the score, the more this stock behaves like a value stock, </br>
Profitability: the higher the score, the more profitable the company is, </br>
Investment: somewhat negligible due to profitability axis - the more positive the score, the more conservatively they reinvest </br>
Scores - somewhat arbitrary - used to test different weights on each factor </br>
The linear regression is run as an experimental exploration of these factors compared to the stock's returns. 

<h1>simpleFamaFrench.py</h1>
This file similarly takes in a list of stock tickers, and just returns values / scores for each of the 5 factors covered in the above description

<h1>portfolioReturns.py</h1>
This file takes in a date and a map of stock tickers to ratios, then returns the market return for the given portfolio starting at the given date.

<h1>main.py</h1>
This file is still being built, but the goal is to provide an interface with all of the tools I am building in this repo.