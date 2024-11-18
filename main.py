import utils.py as ut

stock_ticker = input("Enter the stock ticker symbol: ").upper()
mode = input("Enter the mode: ").upper()
benchmark_ticker='^VOO'
start_date='2000-01-01'
end_date='2023-01-01'
risk_free_rate=0.02/252
data = ut.combine_data(stock_ticker, benchmark_ticker, start_date, end_date)

MODES = {
    'BETA': lambda: ut.calculate_beta(data),
    'ALPHA': lambda: ut.calculate_alpha(ut.calculate_beta(data), data, risk_free_rate),
    'ERROR': lambda: ut.calculate_error(ut.calculate_beta(data), data, risk_free_rate),
    'RETURN': lambda: ut.calculate_average_return(data),
}

selected_mode = MODES.get(mode)
if selected_mode is None:
    print(f"MODE {mode} is not a valid mode.")
else:
    print(f"{mode} of {stock_ticker} is: {selected_mode()}.")