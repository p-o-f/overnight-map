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

# Constants for async requests
MAX_RETRIES = 3 # Arbitrary number of retries for failed requests
CONCURRENT_REQUESTS = 100  # Can be tuned higher/lower based on network stability


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


def get_robinhood_bearer_token(timeout=2): # Below 1 second does not work
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=640,360")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Speed optimizations
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Disable images
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-infobars")
    
    # Add user agent to mimic a real browser
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
    
    # Enable logging for network requests
    chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    
    # Set page load strategy to 'eager' to proceed as soon as the DOM is ready
    chrome_options.page_load_strategy = 'eager'
    
    print("Starting Chrome in headless mode...")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to any specific stock page
        print("Navigating to S&P 500 ETF page...")
        driver.get("https://robinhood.com/stocks/SPY")
        
        time.sleep(timeout)  # Wait for page to load
        
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


async def fetch_json(session, url, headers=None, params=None, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Non-200 response {response.status} for {url}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    return None #TODO <-- fix this line later to do something more useful


async def fetch_symbol_metrics(session, token, symbol):
    try:
        basic_headers = {"User-Agent": "Mozilla/5.0"}
        complex_headers = { # Taken from Network tab in Chrome DevTools; these are the headers that are required to get live quote data
            "authority": "bonfire.robinhood.com",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "authorization": f"Bearer {token}",
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
        
        # Step 1: Get instrument ID
        instrument_id_url = f"https://api.robinhood.com/quotes/{symbol}/"
        id_data = await fetch_json(session, instrument_id_url, basic_headers)
        instrument_id = id_data['instrument_id']

        # Step 2: Get market cap
        fundamental_data_url = f"https://api.robinhood.com/marketdata/fundamentals/{instrument_id}/?bounds=trading&include_inactive=true"
        fundamental_data = await fetch_json(session, fundamental_data_url, basic_headers)
        market_cap = fundamental_data['market_cap']
        volume = fundamental_data['volume']
        average_volume = fundamental_data['average_volume'] # avg volume for last 2 weeks
        
        # Step 3: Get latest quote
        quote_url = f"https://bonfire.robinhood.com/instruments/{instrument_id}/detail-page-live-updating-data/"
        quote_params = {
            "display_span": "day",
            "hide_extended_hours": "false"
        }
        data = await fetch_json(session, quote_url, complex_headers, quote_params)
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
        
        # Process and return everything important as a tuple
        return instrument_id, market_cap, volume, average_volume, dollar_change, percent_change, last_trade_price, last_non_reg_price, extended_hours_price, previous_close_price, adjusted_previous_close_price, overnight

    except Exception as e:
        print(f"Error fetching instrument ID for {symbol}: {e}")


async def fetch_all_symbols(symbols, token):
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for symbol in symbols:
            task = fetch_symbol_metrics(session, token, symbol)
            tasks.append(task)
            await asyncio.sleep(0.05)  # Small sleep between submissions (safe)
        results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    token = get_robinhood_bearer_token()
    
    begin = time.time()


    spx_df = pd.DataFrame(get_sp500_index_info(), columns=["Name", "Symbol", "Sector", "Subsector"])
    spx_symbols = spx_df['Symbol'].tolist()
    spx_results = asyncio.run(fetch_all_symbols(spx_symbols, token))
    metrics_df = pd.DataFrame(spx_results, columns=["Instrument ID", "Market Cap", "Volume", "Average Volume", "Dollar Change", "Percent Change", "Last Trade Price", "Last Non-Reg Price",
                        "Extended Hours Price", "Previous Close Price", "Adjusted Previous Close Price", "Overnight"])
    spx_total_df = pd.concat([spx_df, metrics_df], axis=1)
    spx_total_df = spx_total_df[spx_total_df["Symbol"] != "GOOGL"] # Remove GOOGL from S&P 500 DataFrame
    
    nasdaq_df = pd.DataFrame(get_nasdaq_index_info(), columns=["Name", "Symbol", "Sector", "Subsector"])
    nasdaq_symbols = nasdaq_df['Symbol'].tolist()
    nasdaq_results = asyncio.run(fetch_all_symbols(nasdaq_symbols, token))
    metrics_df = pd.DataFrame(nasdaq_results, columns=["Instrument ID", "Market Cap", "Volume", "Average Volume", "Dollar Change", "Percent Change", "Last Trade Price", "Last Non-Reg Price",
                        "Extended Hours Price", "Previous Close Price", "Adjusted Previous Close Price", "Overnight"])
    nasdaq_total_df = pd.concat([nasdaq_df, metrics_df], axis=1)
    nasdaq_total_df = nasdaq_total_df[nasdaq_total_df["Symbol"] != "GOOGL"] # Remove GOOGL from NASDAQ DataFrame

    print("S&P 500 Data:")
    print(spx_total_df.head())
    print(spx_total_df.shape)
    
    print("\nNASDAQ Data:")
    print(nasdaq_total_df.head())
    print(nasdaq_total_df.shape)
    
    
    end = time.time()

    print(f"Time taken: {end - begin} seconds")