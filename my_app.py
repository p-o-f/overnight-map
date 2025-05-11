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
import functools
import asyncio
import aiohttp
import random

# Constants for async requests
MAX_RETRIES = 3 # Arbitrary number of retries for failed requests
CONCURRENT_REQUESTS = 100  # Can be tuned higher/lower based on network stability

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # This is for Gunicorn to use


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
        market_cap = float(fundamental_data['market_cap'])
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


async def fetch_symbol_metrics_limited(session, token, symbol):
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with sem:
        # This will limit the number of concurrent requests to CONCURRENT_REQUESTS
        return await fetch_symbol_metrics(session, token, symbol)


async def fetch_all_symbols(symbols, token):
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        #---------------------------------------------------------------------------------- To use this without sempahore 
        # tasks = []
        # for symbol in symbols:
        #     task = fetch_symbol_metrics(session, token, symbol)
        #     tasks.append(task)
        #     await asyncio.sleep(0.02)  # Small sleep between submissions (safe)
        # results = await asyncio.gather(*tasks)
        #----------------------------------------------------------------------------------
        tasks = [fetch_symbol_metrics_limited(session, token, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

    return results


def create_heat_map(dataframe):    
    palette = {
        -3: "#e74b3e", -2: "#b44b48", -1: "#84494e",
        0: "#414553", 1: "#457351", 2: "#509957", 3: "#63c667"
    }
    
    black = "#262930"
    
    # Define a custom diverging color scale with more granularity around ±1%
    color_scale = [
        [0.0, palette[-3]], [0.125, palette[-2]], [0.25, palette[-1]], 
        [0.5, palette[0]], [0.75, palette[1]], [0.875, palette[2]], [1.0, palette[3]]
    ]

    # Apply a power transformation to the market cap values
    power = 0.6  # Adjust this value to control the transformation strength
    dataframe['transformed_market_cap'] = np.power(dataframe['market_cap'], power)

    # Create a new column that combines the name with the percentage change and create symbol_with_change column with HTML formatting
    dataframe['symbol_with_change'] = dataframe.apply(
        lambda row: f"<span style='font-size: larger; color: white;'>{row['symbol']}</span><br><span style='color: white;'>{row['percent_change']:+.2f}%</span>",
        axis=1
    )

    # Create Plotly treemap
    fig = px.treemap(
        dataframe,
        path=[px.Constant(" "), 'sector', 'subsector', 'symbol_with_change'], 
        values='transformed_market_cap',
        color='percent_change',
        color_continuous_scale=color_scale, 
        range_color=(-3.1,3.1),
        custom_data=['percent_change', "last_non_reg_price", "name"] # solely affects the hover text (tooltips)
    )
    # Adjust annotation position and style
    fig.update_traces(
        textposition='middle center',
        hovertemplate='<b>%{label}</b><br>' +
                    '%{customdata[2]}<br>'+
                    #'Rolling % change: %{customdata[0]:.2f}%<br>' +
                    'Last price: $%{customdata[1]:,.2f}<br>' + 
                    '<extra></extra>'
    )
    fig.update_traces(marker_line_width = 0.0, marker_line_color=black)
    fig.update_traces(marker=dict(cornerradius=5))
    fig.update_traces(pathbar_visible=False)
    #fig.update_traces(tiling=dict(pad=0, packing='binary'))

    # Modify the colorbar
    fig.update_layout(
        paper_bgcolor='gray',  # or any desired background color
        plot_bgcolor='white',
        coloraxis_colorbar=dict(
            title="Rolling % Change",
            thicknessmode="pixels", thickness=20,
            lenmode="fraction", len=0.33,
            yanchor="bottom", y=-0.1,
            xanchor="center", x=0.5,
            orientation="h",
            )#,
        
    )

    return fig


# Set the layout of the app
app.layout = html.Div([
    html.H1("Overnight Stock Market Heat Map", style={'color': 'white'}),
    dcc.Graph(id='heatmap-graph'),
    dcc.Interval(
        id='interval-component',
        interval=300*1000,  # 5 minutes
        n_intervals=0
    )
])


# Define callback to update the graph
@app.callback(Output('heatmap-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n_intervals):
    token = get_robinhood_bearer_token()
    
    spx_df = pd.DataFrame(get_sp500_index_info(), columns=["name", "symbol", "sector", "subsector"])
    spx_symbols = spx_df['symbol'].tolist()
    spx_results = asyncio.run(fetch_all_symbols(spx_symbols, token))
    metrics_df = pd.DataFrame(spx_results, columns=["instrument_id", "market_cap", "volume", "average_volume", "dollar_change", "percent_change", "last_trade_price", "last_non_reg_price",
                                                    "extended_hours_price", "previous_close_price", "adjusted_previous_close_price", "overnight"])
    spx_total_df = pd.concat([spx_df, metrics_df], axis=1)
    spx_total_df = spx_total_df[spx_total_df["symbol"] != "GOOGL"] # Remove GOOGL from S&P 500 DataFrame
    
    fig = create_heat_map(spx_total_df)
    return fig


if __name__ == "__main__":
    
    app.run(debug=True)
    # begin = time.time()


    # spx_df = pd.DataFrame(get_sp500_index_info(), columns=["name", "symbol", "sector", "subsector"])
    # spx_symbols = spx_df['Symbol'].tolist()
    # spx_results = asyncio.run(fetch_all_symbols(spx_symbols, token))
    # metrics_df = pd.DataFrame(spx_results, columns=["instrument_id", "market_cap", "volume", "average_volume", "dollar_change", "percent_change", "last_trade_price", "last_non_reg_price",
    #                                                 "extended_hours_price", "previous_close_price", "adjusted_previous_close_price", "overnight"])
    # spx_total_df = pd.concat([spx_df, metrics_df], axis=1)
    # spx_total_df = spx_total_df[spx_total_df["Symbol"] != "GOOGL"] # Remove GOOGL from S&P 500 DataFrame
    
    # #time.sleep(3)
    
    # nasdaq_df = pd.DataFrame(get_nasdaq_index_info(), columns=["name", "symbol", "sector", "subsector"])
    # nasdaq_symbols = nasdaq_df['Symbol'].tolist()
    # nasdaq_results = asyncio.run(fetch_all_symbols(nasdaq_symbols, token))
    # metrics_df = pd.DataFrame(nasdaq_results, columns=["instrument_id", "market_cap", "volume", "average_volume", "dollar_change", "percent_change", "last_trade_price", "last_non_reg_price",
    #                                                     "extended_hours_price", "previous_close_price", "adjusted_previous_close_price", "overnight"])
    # nasdaq_total_df = pd.concat([nasdaq_df, metrics_df], axis=1)
    # nasdaq_total_df = nasdaq_total_df[nasdaq_total_df["symbol"] != "GOOGL"] # Remove GOOGL from NASDAQ DataFrame
    
    
    # #pd.set_option('display.max_columns', None)
    # #pd.set_option('display.width', None)  

    
    # print("S&P 500 Data:")
    # print(spx_total_df.head())
    # print(spx_total_df.shape)
    
    # print("\nNASDAQ Data:")
    # print(nasdaq_total_df.head())
    # print(nasdaq_total_df.shape)
    
    # end = time.time()

    # print(f"Time taken: {end - begin} seconds")

