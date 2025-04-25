import time
import requests
import pandas as pd
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_robinhood_bearer_token():
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=640,360")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Add user agent to mimic a real browser
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
    
    # Enable logging for network requests
    chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    
    print("Starting Chrome in headless mode...")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to a specific stock page
        print("Navigating to S&P 500 ETF page...")
        driver.get("https://robinhood.com/stocks/SPY")
        time.sleep(2)  # Wait for page to load
        
        # Extract bearer token from network logs
        print("Extracting bearer token from network requests...")
        bearer_token = None
        
        logs = driver.get_log('performance')
        for log in logs:
            network_log = json.loads(log['message'])
            
            # Look for network requests
            if ('message' in network_log and 
                'method' in network_log['message'] and 
                network_log['message']['method'] == 'Network.requestWillBeSent'):
                
                request = network_log['message']['params']
                
                # Check if this request has authorization headers
                if ('request' in request and 
                    'headers' in request['request'] and 
                    'Authorization' in request['request']['headers']):
                    
                    auth_header = request['request']['headers']['Authorization']
                    if auth_header.startswith('Bearer '):
                        bearer_token = auth_header.replace('Bearer ', '')
                        print("Bearer token found!")
                        break
        
        return bearer_token
        
    finally:
        # Always close the browser
        print("Closing browser...")
        driver.quit()
        
        
def get_ticker_instrument_id(ticker): # does NOT require a bearer token or any type of authentication
    url = f"https://api.robinhood.com/quotes/{ticker}/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}) # pretend to be a regular FireFox browser
    data = response.json()
    return data["instrument_id"]


def get_latest_quote_by_instrument_id(bearer_token, instrument_id): 
    url = f"https://bonfire.robinhood.com/instruments/{instrument_id}/detail-page-live-updating-data/"

    params = {
        "display_span": "day",
        "hide_extended_hours": "false"
    }

    headers = { # Taken from Network tab in Chrome DevTools
        "authority": "bonfire.robinhood.com",
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "authorization": f"Bearer {bearer_token}",
        "dnt": "1",
        "origin": "https://robinhood.com",
        "priority": "u=1, i",
        "referer": "https://robinhood.com/",
        "sec-ch-ua": "\"Google Chrome\";v=\"135\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "x-hyper-ex": "enabled"
    }

    # Make a single GET request
    response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()  
        #print(data.keys())
        #print(data['chart_section']['quote'])
        #print(data.items())  # Print the response data once
        last_trade_price = data['chart_section']['quote']['last_trade_price']
        last_non_reg_price = data['chart_section']['quote']['last_non_reg_trade_price']
        extended_hours_price = data['chart_section']['quote']['last_extended_hours_trade_price']
        previous_close_price = data['chart_section']['quote']['previous_close']
        adjusted_previous_close_price = data['chart_section']['quote']['adjusted_previous_close']
        dollar_change = round(float(adjusted_previous_close_price) - float(last_non_reg_price), 2)
        percent_change = round(dollar_change / float(adjusted_previous_close_price) * 100, 2)
        overnight = previous_close_price != last_non_reg_price
        
        return dollar_change, percent_change, last_trade_price, last_non_reg_price, extended_hours_price, previous_close_price, adjusted_previous_close_price, overnight
        
    else:
        return 0 # Return 0 if the request failed


def get_sp500_index_info():
    url = 'https://www.wikitable2json.com/api/List_of_S%26P_500_companies?table=0'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]: # the first element is the header, format is: SYMBOL / SECURITY / GICS SECTOR / GICS SUB-INDUSTRY / HEADQUARTERS LOCATION / DATE FIRST ADDED / CIK / FOUNDED
        stock_attributes.append([company[0], company[2], company[3]])  
    return stock_attributes
    

def get_nasdaq_index_info():   
    url = 'https://www.wikitable2json.com/api/Nasdaq-100?table=3'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]: # the first element is the header, format is: COMPANY / TICKER / GICS Sector / GICS Sub Industry
        stock_attributes.append([company[1], company[2], company[3]])
    return stock_attributes

token = get_robinhood_bearer_token()

spx_df = pd.DataFrame(get_sp500_index_info(), columns=["Symbol", "Sector", "Subsector"])
nasdaq_df = pd.DataFrame(get_nasdaq_index_info(), columns=["Symbol", "Sector", "Subsector"])

# Create empty columns first
price_columns = ["Price Change", "Percent Change", "Last Trade Price", "Last Non-Reg Price", 
                "Extended Hours Price", "Previous Close Price", "Adjusted Previous Close Price", "Overnight"]

begin = time.time()
for col in price_columns:
    spx_df[col] = None

# Update row by row using loc
for index, row in spx_df.iterrows():
    symbol = row['Symbol']
    try:
        values = get_latest_quote_by_instrument_id(token, get_ticker_instrument_id(symbol))
        if values != 0:  # Check if we got valid data
            spx_df.loc[index, price_columns] = values
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

print(spx_df)

end = time.time()
print(f"Time taken: {end - begin} seconds")