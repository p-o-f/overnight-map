import time
import requests
import pandas as pd
import re
import json
import numpy as np

# Selenium 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Plotly 
import plotly.graph_objects as go
import plotly.express as px
import dash
import pyotp
from dash import dcc, html
from dash.dependencies import Input, Output

# Performance optimizations
import concurrent.futures
import functools
import asyncio
import aiohttp
import random

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
    #(response.text)
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
        if last_trade_price is None:
            last_trade_price = 0
        last_non_reg_price = data['chart_section']['quote']['last_non_reg_trade_price']
        extended_hours_price = data['chart_section']['quote']['last_extended_hours_trade_price']
        previous_close_price = data['chart_section']['quote']['previous_close']
        adjusted_previous_close_price = data['chart_section']['quote']['adjusted_previous_close']
        dollar_change = round(float(last_non_reg_price) - float(adjusted_previous_close_price), 2)
        percent_change = round(dollar_change / float(adjusted_previous_close_price) * 100, 2)
        overnight = previous_close_price != last_non_reg_price
        
        return dollar_change, percent_change, last_trade_price, last_non_reg_price, extended_hours_price, previous_close_price, adjusted_previous_close_price, overnight
        
    else:
        return 0 # Return 0 if the request failed


def get_fundamentals_by_instrument_id(instrument_id): 
    url = f"https://api.robinhood.com/marketdata/fundamentals/{instrument_id}/?bounds=trading&include_inactive=true"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}) # pretend to be a regular FireFox browser
    
    if response.status_code == 200:
        data = response.json()
        market_cap = data['market_cap']
        #print(data)
        #volume = data['volume']
        #average_volume = data['average_volume'] # avg volume for last 2 weeks
        
        return market_cap #, volume, average_volume
    

    else:
        return 0 # Return 0 if the request failed


def get_sp500_index_info():
    url = 'https://www.wikitable2json.com/api/List_of_S%26P_500_companies?table=0'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]: # the first element is the header, format is: SYMBOL / SECURITY / GICS SECTOR / GICS SUB-INDUSTRY / HEADQUARTERS LOCATION / DATE FIRST ADDED / CIK / FOUNDED
        stock_attributes.append([company[1], company[0], company[2], company[3]])  
    return stock_attributes
    

def get_nasdaq_index_info():   
    url = 'https://www.wikitable2json.com/api/Nasdaq-100?table=3'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]: # the first element is the header, format is: COMPANY / TICKER / GICS Sector / GICS Sub Industry
        stock_attributes.append([company[0], company[1], company[2], company[3]])
    return stock_attributes


token = get_robinhood_bearer_token()

# Create empty columns to add to index dataframes
price_columns = ["Price Change", "Percent Change", "Last Trade Price", "Last Non-Reg Price", 
                "Extended Hours Price", "Previous Close Price", "Adjusted Previous Close Price", "Overnight", "Market Cap"]


def create_spx_df():
    spx_df = pd.DataFrame(get_sp500_index_info(), columns=["Name", "Symbol", "Sector", "Subsector"])
    for col in price_columns:
        spx_df[col] = None
    # Update row by row using loc
    for index, row in spx_df.iterrows():
        symbol = row['Symbol']
        try:
            symbol_id = get_ticker_instrument_id(symbol)
            values = list(get_latest_quote_by_instrument_id(token, symbol_id))
            values.append(get_fundamentals_by_instrument_id(symbol_id))  # Append market cap to the values list
            
            if values != 0:  # Check if we got valid data
                spx_df.loc[index, price_columns] = values
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    return spx_df


def create_nasdaq_df():
    nasdaq_df = pd.DataFrame(get_nasdaq_index_info(), columns=["Name", "Symbol", "Sector", "Subsector"])
    
    for col in price_columns:
        nasdaq_df[col] = None
    # Update row by row using loc
    for index, row in nasdaq_df.iterrows():
        symbol = row['Symbol']
        try:
            symbol_id = get_ticker_instrument_id(symbol)
            values = list(get_latest_quote_by_instrument_id(token, symbol_id))
            values.append(get_fundamentals_by_instrument_id(symbol_id))  # Append market cap to the values list
            
            if values != 0:  # Check if we got valid data
                nasdaq_df.loc[index, price_columns] = values
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    return nasdaq_df

# begin = time.time()

# spx_df = create_spx_df()
# print(spx_df.head())

# end = time.time()
# print(f"Time taken: {end - begin} seconds")

# begin = time.time()

# nasdaq_df = create_nasdaq_df()
# print(nasdaq_df.head())

# end = time.time()
# print(f"Time taken: {end - begin} seconds")
print("TEST")

# print(get_latest_quote_by_instrument_id(token, get_ticker_instrument_id("GOOGL")))
# print(get_fundamentals_by_instrument_id(get_ticker_instrument_id("GOOGL")))


# Set this once
MAX_RETRIES = 3
CONCURRENT_REQUESTS = 50  # Tune this higher/lower based on network stability

async def fetch_json(session, url, headers=None, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Non-200 response {response.status} for {url}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    return None

async def fetch_instrument_ids(session, token, symbol):
    try:
        # Step 1: Get instrument ID
        url_id = f"https://api.robinhood.com/quotes/{symbol}/"
        basic_headers = {"User-Agent": "Mozilla/5.0"}
        id_data = await fetch_json(session, url_id, basic_headers)
        if not id_data or 'instrument_id' not in id_data:
            return symbol, [None] * 9

        instrument_id = id_data['instrument_id']
        return instrument_id
    except Exception as e:
        print(f"Error fetching instrument ID for {symbol}: {e}")

async def fetch_all_symbols(symbols, token):
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for symbol in symbols:
            task = fetch_instrument_ids(session, token, symbol)
            tasks.append(task)
            await asyncio.sleep(0.05)  # Small sleep between submissions (safe)
        results = await asyncio.gather(*tasks)
    return results

spx_df = pd.DataFrame(get_sp500_index_info(), columns=["Name", "Symbol", "Sector", "Subsector"])
symbols = spx_df['Symbol'].tolist()
results = asyncio.run(fetch_all_symbols(symbols, token))
print(results)