import yfinance as yf
import time
import datetime
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
from datetime import datetime as seconddatetime
start_time = time.time()
from IPython.display import display, HTML
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



actualstock = "MOD"
period_ = '2mo'
forecasted_pricetarget = 205




def format_text_html_3d(text):
    return f'<div style="font-size:24px;color:white;font-family:Arial;text-shadow: 2px 2px 4px #000000;">{text}</div>'
# Function to format text using HTML
def format_text_html(text):
    return f'<div style="font-size:16px;color:#333;font-family:Arial;">{text}</div>'
# Define the function to be called when the button is clicked
def on_button_clicked(b):
    selected_date = date_picker.value
    print("Selected date:", selected_date)
# Create a DatePicker widget for start date
start_date_picker = widgets.DatePicker(description='Select a start date', value=seconddatetime.today(), disabled=False)
# Create a DatePicker widget for end date
end_date_picker = widgets.DatePicker(description='Select an end date', value=seconddatetime.today(), disabled=False)
# Create a button widget
button = widgets.Button(description="Click me")
# Connect the button widget to the function
button.on_click(on_button_clicked)
# Display the DatePickers and the button
display(widgets.VBox([start_date_picker, end_date_picker, button]))
# Define the function to handle text entry
def on_text_change(change):
    entered_text = text_entry.value
    print("Entered ticker name:", entered_text)
# Create a text entry widget
text_entry = widgets.Text(description='Enter text:', value='', disabled=False)
# Connect the text entry widget to the function
text_entry.observe(on_text_change, names='value')
# Display the text entry widget
display(text_entry)
# Define the function to search for stocks
def search_stocks(query):
    # Use yfinance to search for stocks matching the query
    search_results = yf.Tickers(query)
    return search_results.tickers
# Function to handle search button click
def search_button_clicked(b):
    query = search_bar.value
    if query:
        results = search_stocks(query)
        print("Search results:")
        for result in results:
            print(result.ticker)
            # You can print additional information about the stock if needed
    else:
        print("Enter a search query")

# Create a text input widget for search
search_bar = widgets.Text(placeholder='Search stocks...', description='Search:')
# Create a button widget for triggering search
search_button = widgets.Button(description='Search')
# Connect the button widget to the search function
search_button.on_click(search_button_clicked)
# Display the search bar and button
display(widgets.VBox([search_bar, search_button]))





start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2023, 12, 31)
period_specified_ = ''

#ticker = yf.Ticker("BTCC-B.TO")
ticker = yf.Ticker(actualstock)
market_index_input = '^GSPC'
#ticker = yf.Ticker("TECK-A.TO")
ticker_balsheet = pd.DataFrame(ticker.balance_sheet)
balsheet_categories = ticker_balsheet.index
ticker_histdata = ticker.history(period=period_)
histdata_categories = ticker_histdata.index
close_vals = ticker_histdata['Close']
np_close_vals = np.array(close_vals)

options_data = ticker.option_chain()
options_index_data = list(options_data[2].items())


print('--------')
for i in options_index_data:
    if 'trailingPE' in i:
        print(i)


call_option_data = options_data[0]
put_option_data = options_data[1]
#print(call_option_data)


print('--------')
current_price = ticker.history(period='1d')['Close'].iloc[-1]
formatted_current_price = (f'Current Price: ${round((current_price),2)}')
print(formatted_current_price)
print('--------')

call_strike_list = call_option_data['strike']
put_strike_list = put_option_data['strike']
call_lastprice_data = call_option_data['lastPrice']
put_lastprice_data = put_option_data['lastPrice']
#print(strike_list)



integer_ct = 0
liszt = []
for i in call_strike_list:
    if i > current_price:
        muadib = i
        print(f'Closest Call to Current Price: ${i}')
        break  # Exit the loop when the condition is met
    integer_ct+=1

print(f'Last Price at Closest Call Strike: ${call_lastprice_data[integer_ct]}')
minbuy = 100 # units
print(f'Minimum Buy: x{minbuy}')

call_total_min_incurred = (call_lastprice_data[integer_ct])
print(f'Minimum Incurred Cost: ${call_total_min_incurred*minbuy}')
print(f'Expiry: ')
print('--------')

put_integer_ct = 0
put_liszt = []
for i in put_strike_list:
    if i > current_price:
        put_liszt.append(put_strike_list[put_integer_ct-1])
        break  # Exit the loop when the condition is met
    put_integer_ct+=1

print(f'Closest Put to Current Price: ${(put_liszt[0])}')
print(f'Last Price at Closest Call Strike: ${put_lastprice_data[put_integer_ct-1]}')
print(f'Minimum Buy: x{minbuy}')

put_total_min_incurred = (put_lastprice_data[put_integer_ct-1])
print(f'Minimum Incurred Cost: ${put_total_min_incurred*minbuy}')
print(f'Expiry: ')
print('--------')





print('Simulated call results')
print('--------')

print(f'Forecast Price Target: ${forecasted_pricetarget}')
print(f'Purchased Call Price: ${muadib}')

call_profit = forecasted_pricetarget - muadib
total_call_profit = call_profit * minbuy
total_call_contractfees = call_total_min_incurred*minbuy

put_profit = forecasted_pricetarget - put_liszt[0]

print(f'Profit per Share: ${call_profit}')
print(f'Total Profit: ${total_call_profit}')
print(f'Net Contract Fees: ${total_call_contractfees}')
print('--------')
print('Excercised: Yes')
print(f'Net Differential: +${total_call_profit-total_call_contractfees}')
print('--------')

print('Simulated put results')
print('--------')
print(f'Forecast Price Target: ${forecasted_pricetarget}')
print(f'Purchased Put Price: ${(put_liszt[0])}')
print(f'Loss per Share: ${put_profit}')
print(f'Total Loss: -${put_profit*minbuy}')
print('--------')
print('Excercised: No')
print(f'Net Differential: -${put_total_min_incurred*minbuy}')


print('--------')
print('Simulated straddle')
print('--------')

print(f'Straddle Differential: +${(total_call_profit-total_call_contractfees)-(put_total_min_incurred*minbuy)}')
print('--------')

implied_volatility = 1 - (put_liszt[0] / forecasted_pricetarget)
print(f'Implied Volatility: {round((implied_volatility*100),2)}%')
print('--------')









print('----------------')

def closevals_change_calculator(array):
    index = 0
    change_closevals = []
    while index < (len(array)-1):
        change_at_interval = round(((1 - (array[index] / array[index+1]))*100),3)
        change_closevals.append(change_at_interval)
        index +=1
    #print(change_closevals)
    print('----------------')

    x_axis = range(len(array)-1)

    plt.plot(x_axis, change_closevals, color='black')
    plt.xlabel('Days')
    plt.ylabel('Daily Change (%)')
    plt.title('Volatility per Day',fontsize=30)
    plt.xticks(np.arange(0, len(array)-1, step=10))
    plt.axhline(y=0, color='grey', linestyle=':')
    for i in range(len(change_closevals)):
        if change_closevals[i] > 0:
            plt.plot(i, change_closevals[i], marker='o', markersize=4, color='green')
        elif change_closevals[i] < 0:
            plt.plot(i, change_closevals[i], marker='o', markersize=4, color='red')
        else:
            plt.plot(i, change_closevals[i], marker='o', markersize=4, color='grey')

    print('----------------')

    sum_change_closevals = sum(abs(val) for val in change_closevals)
    avg_velocity_perday = sum_change_closevals / len(change_closevals)
    std_closevals = np.std(change_closevals)
    ticker_name = ticker.info['longName']
    
    
    
    
    formatted_name = (f'{ticker_name}')
    display(HTML(format_text_html_3d(formatted_name)))
    
    apdv = (f'Average per Day Volatility: {round(avg_velocity_perday,3)} % per day')
    display(HTML(format_text_html_3d(apdv)))

    ods = (f'Overall Deviation of Series: {(round((std_closevals),3))} % bounds')
    display(HTML(format_text_html_3d(ods)))




    current_price = ticker.history(period='1d')['Close'].iloc[-1]
    formatted_current_price = (f'Current Price:${current_price}')
    display(HTML(format_text_html_3d(formatted_current_price)))

    #display(HTML('<textarea rows="10" cols="80">{}</textarea>'.format(text)))

    baseline = 0
    upper_bound = baseline + avg_velocity_perday + std_closevals
    lower_bound = baseline - avg_velocity_perday - std_closevals

    def boundgrapher(array):
        index = 0
        upper_bounds_chain = []
        lower_bounds_chain = []

        while index < len(array):
            upper_bounds_chain.append(change_closevals[index] + upper_bound)
            lower_bounds_chain.append(change_closevals[index] + lower_bound)
            index += 1

        x_axis_forbounds = len(change_closevals)

        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(range(0, x_axis_forbounds), change_closevals, color='lightgrey')
        ax.plot(range(0, x_axis_forbounds), upper_bounds_chain, color='purple')
        ax.plot(range(0, x_axis_forbounds), lower_bounds_chain, color='purple')

        print('----------------')
        print('----------------')
        print('----------------')

        # Adding labels above the lines for every 3 points
        for i in range(0, len(change_closevals), 3):
            ax.text(i, upper_bounds_chain[i], f'{upper_bounds_chain[i]:.2f}', ha='center', va='bottom')
            ax.text(i, lower_bounds_chain[i], f'{lower_bounds_chain[i]:.2f}', ha='center', va='bottom')

        ax.set_title('Volatility Deviation Bounds',fontsize=25)

        plt.show()

    boundgrapher(change_closevals)

    # Plotting close prices
    plt.figure(figsize=(20, 6))
    plt.plot(array[1:], color='dodgerblue')
    plt.xlabel('Days')
    plt.xticks(np.arange(0, len(array)-1, step=10))
    plt.ylabel('Close Price ($)')
    plt.title('Close Price Over Time',fontsize=25)
    plt.show()

    print('----------------')
    print('----------------')
    print('----------------')
    print('----------------')

    print('----------------')
    print('----------------')
    print('----------------')

closevals_change_calculator(np_close_vals)

end_time = time.time()
execution_time = end_time - start_time
print(f'Executed in {round((execution_time),3)} seconds.')
print('----------------')


def start_function():

    ## Currency formatter function
    def currency_formatter(value):
        return locale.currency(value, grouping=True)

    # Part 1 - Fundamental Analysis #
    def main():
        def all():
            ## Sample Stock Input: Apple / AAPL
            stock1 = yf.Ticker(actualstock)
            ## String slice & print
            print(f'Ticker Name: {actualstock}')

            ## Set "S&P 500 Index" as the Industry Sector for all Beta Calculations
            market_index = yf.Ticker(market_index_input) # ...so S&P 500 only

            def main_body():
                def wacc_and_related():
                    def dcf_model():
                        def enterprise_valuator():
                            
                            ## Find enterprise value
                            ent_val = mkt_cap + total_debt - cash_and_ce

                            ## Calculate equity value of ticker (convert from ent_val)
                            equity_val = mkt_cap + cash_and_ce - total_debt
                            
                            ## Find Implied Share Price
                            implied_share_price = equity_val / no_of_shares

                            ## Print Current price
                            ## Print ISP
                            print(f'Enterprise Value: {currency_formatter(ent_val)}')
                            print(f'Estimated Intrinsic Value: {currency_formatter(equity_val)}')
                            print(f'Current Price: {currency_formatter(current_price)}')
                            print(f'Implied Share Price: {currency_formatter(implied_share_price)}')

                            ## Find pct difference between current price and implied share price
                            dif = (current_price/implied_share_price)

                            ## Establish Margin of Safety
                            mrg_of_sfty = 0.1

                            acceptable_buy_price = (1 - mrg_of_sfty)*implied_share_price
                            print(f'Acceptable Buy Price Given Margin of Safety (0.30 or 30%): {currency_formatter(acceptable_buy_price)}')

                            if current_price > acceptable_buy_price:
                                print('Sell')
                            #ROUND HOLD TO NEAREST (BOUNDS)
                            elif current_price == acceptable_buy_price:
                                print('Hold')
                            else:
                                print('Buy')

                        ## Discounted Cash Flow Model Calculator
                        ## 1) Find FCF (Free Cash Flow)
                        fcf = cashflow_st['Free Cash Flow']

                        ## 2) Calculate Market Growth Rate. ('*2' as it is a ...
                        ## ... 6mo period. 'iloc' for more-accurate integer key (less output text desc.)
                        annual_g = (market_returns.iloc[-1] - market_returns.iloc[0])*2
                        
                        ## 3) Calculate Terminal Value using Perpetuity Growth Method
                        perp_growth_terminal_val = (fcf * (1 + annual_g)) / (wCoC - annual_g)

                        ## 4) Calculate Enterprise Value (ent_val)
                        enterprise_valuator()

                    ## Calculate WACC (wCoC)
                    wCoC = ((erp * beta) + rf) + cost_of_debt
                    final_wacc = wCoC*100
                    print(f'Weighted Average Cost of Capital (WACC): {round((final_wacc),2)}%')

                    ## Calculate industry returns for period
                    print(f'Industry Returns for Period {period_}: {round((variance_of_market*100000),2)}%')

                    ## Call DCF discounted cash flow model calculator
                    dcf_model()
                

                ## Find bal sheet, income statement & cashflow stmt for ticker
                scraped_dates = (stock1.balance_sheet.iloc[0]).index
                bal_sheet_date = (str(scraped_dates[0]))[:10]
                print(f'Date of Most Recent Info: {bal_sheet_date}')
                bal_sheet = stock1.balance_sheet[bal_sheet_date]
                inc_st = stock1.income_stmt[bal_sheet_date]
                cashflow_st = stock1.cashflow[bal_sheet_date]
                df_cashflow_st = pd.DataFrame(cashflow_st)

                ## Find EPS
                eps = inc_st["Diluted EPS"]
                print(f'Diluted EPS: {eps}')

                ## Find market cap & no. of shares for ticker
                mkt_cap = stock1.basic_info['marketCap']
                no_of_shares = bal_sheet['Ordinary Shares Number']
                
                ## Gather 6mo recent close value history for ticker,...
                ## ...arrange close values in pandas dataframe
                stock1_info = stock1.history
                df_stock1_info = pd.DataFrame(stock1_info(period=(period_)))
                stock1_cvals = df_stock1_info['Close']
                
                ## Find Current Price
                current_price = stock1_cvals.iloc[-1]

                ## Find P/E
                pe_ratio = current_price / eps

                ## Gather 6mo recent close value history for market index...
                ## ...performance, ditto above
                ## arrange close values in pandas dataframe again
                market_data = market_index.history(period=(period_))
                df_market_data = pd.DataFrame(market_data)

                ## Calculate stock and market returns (pct change) for ticker
                stock1_returns = (stock1_cvals.pct_change().dropna())
                market_returns = df_market_data['Close'].pct_change().dropna()
                
                ## Check if difference
                ## Equalized if different number of items
                ## Append to final list
                difference = int(len(stock1_returns) - len(market_returns))
                stock1_returns_equalized = []
                market_returns_equalized = []
                
                if difference > 0:
                    stock1_returns_equalized = stock1_returns.iloc[:-difference]
                    market_returns_equalized = market_returns
                elif difference < 0:
                    stock1_returns_equalized = stock1_returns
                    market_returns_equalized = market_returns.iloc[-difference:]
                else:
                    stock1_returns_equalized = stock1_returns
                    market_returns_equalized = market_returns

                ## Calculate covariance & variance of returns (ticker v market)
                ## MODIFY TO ADJUST FOR EQUALIZED NUM!
                covariance = np.cov(stock1_returns_equalized, market_returns_equalized)[0,1]
                variance_of_market = np.var(market_returns_equalized)

                ## Calculate beta of ticker
                beta = covariance / variance_of_market

                ## Designate Equity Risk Premium (erp) for USA (based on S&P), 2024 rate (4.6%)
                erp = 0.046
                ## Designate Rf (Risk-Free Rate). Assume time period 20y. (4.22%)
                ## Treasury Info Link: https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2023
                rf = 0.0422
                ## Designate Rm (Expected Market Return). Assume 8% (based on S&P Historical)
                rm = 0.08

                ## The below is possibly needed in the future
                ## Calculate Cost of Equity (Re)
                cost_of_equity = rf + beta * (rm - rf)

                ## Find total assets, debt, and liabilities. calc future debt:
                total_assets = bal_sheet['Total Assets']
                total_debt = bal_sheet['Total Debt']
                total_liabilities = bal_sheet['Total Liabilities Net Minority Interest']
                future_debt = (bal_sheet['Total Non Current Liabilities Net Minority Interest']) \
                                + (bal_sheet['Other Non Current Liabilities'])

                ## TOTAL EQUITY
                ## APIC Assume as $0 due to missing yfinance SoCI.
                cmn_stock = bal_sheet['Common Stock']
                cmn_stock_equity = bal_sheet['Common Stock Equity']
                treasury_stock = cmn_stock - cmn_stock_equity
                apic = bal_sheet['Capital Stock']
                retained_earnings = bal_sheet['Retained Earnings']
                ## Calc TE
                total_equity = cmn_stock + (apic*0) + retained_earnings + treasury_stock #+ oci 

                ## Calculate Market Value of Debt. Assume interest = rf above
                ## Also, assume 'n' (YTM) = 5, based on S&P typicals.
                ## Recall: CMVoD: pv = fv / (1+r)^n
                mkt_val_ofdebt = future_debt / (1+rf)**5
                ## Calculate Annual Interest Payment
                ## AIP = Total Debt x Interest Rate. Also assume interest = rf above
                ann_int_pmt = total_debt * rf
                ## Calculate Cost of Debt (Rd)
                cost_of_debt = ann_int_pmt / mkt_val_ofdebt

                ## Find weightages
                equity_weightage = (total_equity / mkt_cap) * 100
                debt_weightage = (total_debt / mkt_cap) * 100

                ## Find cash and cash equivalents
                cash_and_ce = bal_sheet['Cash And Cash Equivalents']

                ## Calculate debt ratio
                debt_ratio = total_debt / total_assets

                ## Call WACC_etc to begin technical analysis
                wacc_and_related()         

            main_body()

        all()
    main()

start_function()