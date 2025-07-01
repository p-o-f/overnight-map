import time
import requests
import pandas as pd
import re
import json
import numpy as np
from datetime import datetime
import pytz

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager  # auto-manage ChromeDriver

# Plotly
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import callback_context
from dash import dash_table
import dash_bootstrap_components as dbc

# Performance optimizations
import functools
import asyncio
import aiohttp
import random

# Constants for async requests
MAX_RETRIES = 3  # Arbitrary number of retries for failed requests
CONCURRENT_REQUESTS = 100  # Can be tuned higher/lower based on network stability

# Initialize Dash app
# app = dash.Dash(__name__)
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
)
server = app.server  # This is for Gunicorn to use

# Global constants for Dash
spx_fig = None
nasdaq_fig = None
spx_total_df = None
nasdaq_total_df = None
last_token = None  # To save on Chrome startup time, we will only fetch the token once every TOKEN_REFRESH_SECONDS seconds
last_token_time = 0
TOKEN_REFRESH_SECONDS = 1200  # 20 mins

BOTTOM_CAPTION = html.P([
    "Note: this data is sourced from Robinhood, though this site is not affiliated with Robinhood. ",
    "The provided data for should not be considered as any type of financial advice and may be inaccurate. ",
    "If you find this service useful, please consider supporting the server costs for this project by ",
    html.A("DONATING HERE.", href="https://buymeacoffee.com/pfdev",
           target="_blank", style={'color': 'lightblue'})
], style={'color': 'white', 'marginTop': '10px', 'fontSize': '12px', 'textAlign': 'center'})

PAGE_TITLE = html.H1(
    html.A("Overnight Stock Market Heat Map", href="https://buymeacoffee.com/pfdev", target="_blank", style={'color': 'lightblue'}),
    )


def get_sp500_index_info():
    url = 'https://www.wikitable2json.com/api/List_of_S%26P_500_companies?table=0'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]:  # the first element is the header, format is: SYMBOL / COMPANY_NAME / GICS SECTOR / GICS SUB-INDUSTRY / HEADQUARTERS LOCATION / DATE FIRST ADDED / CIK / FOUNDED
        stock_attributes.append(
            [company[1], company[0], company[2], company[3]])
    return stock_attributes


def get_nasdaq_index_info():
    url = 'https://www.wikitable2json.com/api/Nasdaq-100?table=3'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    # the first element is the header, format is: SYMBOL / COMPANY_NAME / GICS SECTOR / GICS SUB-INDUSTRY
    for company in data[1:]:
        stock_attributes.append(
            [company[1], company[0], company[2], company[3]])
    return stock_attributes


def get_robinhood_bearer_token(timeout=2):  # Below 1 second does not work
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.binary_location = "/usr/bin/chromium-browser"
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=640,360")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Speed optimizations
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument(
        "--blink-settings=imagesEnabled=false")  # Disable images
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-infobars")

    # Add user agent to mimic a real browser
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")

    # Enable logging for network requests
    chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    # Set page load strategy to 'eager' to proceed as soon as the DOM is ready
    chrome_options.page_load_strategy = 'eager'

    print("Starting Chrome in headless mode...")
    service = Service("/usr/bin/chromedriver")
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
        if not bearer_token:
            print("⚠️ Failed to extract bearer token from Chrome logs")
            return None

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
    return None  # TODO <-- fix this line later to do something more useful


async def fetch_symbol_metrics(session, token, symbol):
    try:
        basic_headers = {"User-Agent": "Mozilla/5.0"}
        complex_headers = {  # Taken from Network tab in Chrome DevTools; these are the headers that are required to get live quote data
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
        # avg volume for last 2 weeks
        average_volume = fundamental_data['average_volume']

        # Step 3: Get latest quote
        quote_url = f"https://bonfire.robinhood.com/instruments/{instrument_id}/detail-page-live-updating-data/"
        quote_params = {
            "display_span": "day",
            "hide_extended_hours": "false"
        }

        data = await fetch_json(session, quote_url, complex_headers, quote_params)
        last_trade_price = data['chart_section']['quote']['last_trade_price'] or 0
        last_non_reg_price = data['chart_section']['quote']['last_non_reg_trade_price'] or last_trade_price
        extended_hours_price = data['chart_section']['quote']['last_extended_hours_trade_price'] or 0
        previous_close_price = data['chart_section']['quote']['previous_close'] or 0
        adjusted_previous_close_price = data['chart_section']['quote']['adjusted_previous_close'] or 0

        dollar_change = round(float(last_non_reg_price) -
                              float(adjusted_previous_close_price), 2)
        percent_change = round(
            dollar_change / float(adjusted_previous_close_price) * 100, 2)
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
        # Create a list of tasks for each symbol
        tasks = [fetch_symbol_metrics_limited(
            session, token, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

    return results


def create_heat_map(dataframe, map_title):
    palette = {
        -3: "#e74b3e", -2: "#b44b48", -1: "#84494e",
        0: "#414553", 1: "#457351", 2: "#509957", 3: "#63c667"
    }

    black = "#262930"

    # Define a custom diverging color scale with more granularity around ±1%
    color_scale = [
        [0.0, palette[-3]], [0.125, palette[-2]], [0.25, palette[-1]],
        [0.5, palette[0]], [0.75, palette[1]], [
            0.875, palette[2]], [1.0, palette[3]]
    ]

    # Apply a power transformation to the market cap values
    power = 0.6
    dataframe['transformed_market_cap'] = np.power(
        dataframe['market_cap'], power)

    # Create formatted label with percent change
    dataframe['symbol_with_change'] = dataframe.apply(
        lambda row: f"<span style='font-size: larger; color: white;'>{row['symbol']}</span><br><span style='color: white;'>{row['percent_change']:+.2f}%</span>",
        axis=1
    )

    overnight_on = dataframe['overnight'].value_counts().iloc[0]
    try:
        # If there is no overnight trading data, this will raise an IndexError
        overnight_off = dataframe['overnight'].value_counts().iloc[1]
    except IndexError:
        # If there is no overnight trading data, set overnight_off to 0
        overnight_off = 0
    print(f"{map_title}\nOvernight on: {overnight_on}, Overnight off: {overnight_off}")

    # New York timezone
    ny_tz = pytz.timezone("America/New_York")
    # Current time in New York
    ny_time = datetime.now(ny_tz)
    # Format like "5/13/2025, 6:14 AM, EST"
    # For cross-platform zero-stripping
    formatted_time = ny_time.strftime(
        "%m/%d/%Y, %I:%M %p, %Z").lstrip("0").replace(" 0", " ")
    graph_top_title = f"{map_title} - Overnight Trading is currently enabled for " + \
        str(overnight_on) + " symbols and disabled for " + \
        str(overnight_off) + " symbols"
    total_title = f"Last refreshed at: {formatted_time}"

    # Create Plotly treemap
    fig = px.treemap(
        dataframe,
        path=[px.Constant(graph_top_title), 'sector',
              'subsector', 'symbol_with_change'],
        values='transformed_market_cap',
        color='percent_change',
        color_continuous_scale=color_scale,
        range_color=(-3.1, 3.1),
        custom_data=['name', 'last_non_reg_price',
                     'overnight', "volume", "average_volume"],
    )

    # Function to format each row based on the count of '(?)'; this is necessary because Plotly does not allow for custom hover text
    # to be set for each individual node in a treemap
    hover_data = fig.data[0].customdata

    def format_row(row):
        # Count how many instances of '(?)' there are
        question_marks_count = np.count_nonzero(row == '(?)')

        # If there are no '(?)' in the row, format as a detailed string
        if question_marks_count == 0:
            name, last_price, overnight_trading, volume, avg_volume, percent_change = row
            formatted_last_price = round(float(last_price), 3)
            formatted_volume = round(float(volume) / 1000000, 2)
            formatted_avg_volume = round(float(avg_volume) / 1000000, 2)
            formatted_volume = f"{formatted_volume}M" if formatted_volume >= 1 else f"{formatted_volume * 1000}K"
            formatted_avg_volume = f"{formatted_avg_volume}M" if formatted_avg_volume >= 1 else f"{formatted_avg_volume * 1000}K"
            formatted_percent_change = round(float(percent_change), 3)
            return [
                f"Name: {name}",
                f"Last Price: ${formatted_last_price}",
                f"Overnight Trading Enabled: {overnight_trading}",
                f"Volume: {formatted_volume}",
                f"Average Volume: {formatted_avg_volume}",
                f"Percent Change: {formatted_percent_change}%"
            ]

        # If there are 4 instances of '(?)', format with placeholders and 'Overnight Trading Enabled' and 'Percent Change'
        elif question_marks_count == 4:
            _, _, overnight_trading, _, _, percent_change = row
            formatted_percent_change = round(float(percent_change), 3)
            return [
                " ",
                " ",
                " ",
                " ",
                f"Overnight Trading Enabled: {overnight_trading}",
                f"Percent Change of Subsector: {formatted_percent_change}%"
            ]

        # If there are 5 instances of '(?)', format with placeholders and just 'Percent Change'
        elif question_marks_count == 5:
            _, _, _, _, _, percent_change = row
            formatted_percent_change = round(float(percent_change), 3)

            return [
                " ",
                " ",
                " ",
                " ",
                " ",
                f"Percent Change of Sector: {formatted_percent_change}%"
            ]

        # If other cases occur, return the row as-is
        return row

    fig.data[0].customdata = np.array([format_row(row) for row in hover_data])

    # Set hover text
    fig.update_traces(
        hovertemplate='<span style="color:white;">%{label}</span><br><br>' +
        # '<span style="color:white;">Market Cap: $%{value}</span><br>' + # TODO fix this to be proper units
        '<span style="color:white;">Parent Category: %{parent}</span><br>' +
        '<span style="color:white;">Percentage of Index: %{percentRoot:.2%}</span><br>' +
        '<span style="color:white;">Percentage of Parent Category: %{percentParent:.2%}</span><br><br>' +
        '<span style="color:white;">%{customdata[0]}</span><br>' +
        '<span style="color:white;">%{customdata[1]}</span><br>' +
        '<span style="color:white;">%{customdata[2]}</span><br>' +
        '<span style="color:white;">%{customdata[3]}</span><br>' +
        '<span style="color:white;">%{customdata[4]}</span><br>' +
        '<span style="color:white;">%{customdata[5]}</span><br>' +
        '<extra></extra>'
    )

    fig.update_layout(
        hoverlabel=dict(
            bgcolor='rgb(66, 73, 75)',     # background color of hovertext
            font_size=13,
            font_color="white",  # text color
            bordercolor="black"  # optional, default is automatic
        )
    )

    fig.update_layout(
        paper_bgcolor='gray',
        plot_bgcolor='white',
        coloraxis_colorbar=dict(
            title="% Change",
            thicknessmode="pixels",
            thickness=16, 
            lenmode="fraction",
            len=0.5,        # 50% width
            yanchor="bottom",
            y=-0.1,        # move coloraxis down a bit
            xanchor="center",
            x=0.5,
            orientation="h",
            title_font=dict(size=10, color="white"),
            tickfont=dict(size=8, color="white"),
        )
    )

    fig.update_layout(
        title=dict(
            text=total_title,
            font=dict(
                size=14,
                color='white'
            ),
            x=0.5,             # center horizontally
            xanchor='center'   # aanchor the x=0.5 position at the center of the text
        )
    )

    #  Styling behavior
    fig.update_traces(
        textposition='middle center',
        marker_line_width=0.0,
        marker_line_color=black,
        marker=dict(cornerradius=5),
        pathbar_visible=False,
        textfont=dict(
            color="white",  # Text color for labels
        )
    )

    # Better view for mobile
    fig.update_layout(
        autosize=True,
        margin=dict(l=30, r=30, t=30, b=0),  # Remove margins
    )

    print("Fig created for " + map_title)
    return fig


def preload_figures(token):
    global spx_fig, nasdaq_fig
    global spx_total_df, nasdaq_total_df

    # Debug
    if not token:
        print("❌ Bearer token was None — likely token fetch failure.")
    else:
        print("✅ Bearer token successfully retrieved")

    # S&P 500
    spx_df = pd.DataFrame(get_sp500_index_info(), columns=[
                          "name", "symbol", "sector", "subsector"])
    spx_results = asyncio.run(fetch_all_symbols(
        spx_df['symbol'].tolist(), token))
    spx_metrics_df = pd.DataFrame(spx_results, columns=["instrument_id", "market_cap", "volume", "average_volume",
                                                        "dollar_change", "percent_change", "last_trade_price",
                                                        "last_non_reg_price", "extended_hours_price",
                                                        "previous_close_price", "adjusted_previous_close_price", "overnight"])
    spx_total_df = pd.concat([spx_df, spx_metrics_df], axis=1)
    spx_total_df = spx_total_df[spx_total_df["symbol"] != "GOOGL"]
    spx_fig = create_heat_map(spx_total_df, "S&P 500")

    # NASDAQ
    nasdaq_df = pd.DataFrame(get_nasdaq_index_info(), columns=[
                             "name", "symbol", "sector", "subsector"])
    nasdaq_results = asyncio.run(fetch_all_symbols(
        nasdaq_df['symbol'].tolist(), token))
    nasdaq_metrics_df = pd.DataFrame(nasdaq_results, columns=["instrument_id", "market_cap", "volume", "average_volume",
                                                              "dollar_change", "percent_change", "last_trade_price",
                                                              "last_non_reg_price", "extended_hours_price",
                                                              "previous_close_price", "adjusted_previous_close_price", "overnight"])
    nasdaq_total_df = pd.concat([nasdaq_df, nasdaq_metrics_df], axis=1)
    nasdaq_total_df = nasdaq_total_df[nasdaq_total_df["symbol"] != "GOOGL"]
    nasdaq_fig = create_heat_map(nasdaq_total_df, "NASDAQ 100")


def generate_table(df, title, max_rows=30):
    df_sorted = df.sort_values(by="percent_change", ascending=False)

    # Columns to display and their headers
    columns = ['symbol', 'name', 'sector', 'percent_change',
               'market_cap', 'volume', 'average_volume', 'last_non_reg_price']
    pretty_names = ['Symbol', 'Name', 'Sector', '% Change',
                    'Market Cap', 'Volume', 'Avg Volume', 'Last Price']

    def format_large_number(value):
        try:
            value = float(value)
            if value >= 1_000_000_000_000:
                return f"{value / 1_000_000_000_000:.2f}T"
            if value >= 1_000_000_000:
                return f"{value / 1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"{value / 1_000_000:.2f}M"
            elif value >= 1_000:
                return f"{value / 1_000:.2f}K"
            else:
                return f"{value:.2f}"
        except:
            return value

    # TODO can make this sortable by column later (optionally); also need to fix margins and aesthetic stuff
    return html.Div([
        html.H3(f"{title} - Top {max_rows} by % Change",
                style={'color': 'white', 'marginTop': '20px'}),
        html.Table([
            html.Thead(
                html.Tr([html.Th(col, style={
                        'color': 'white', 'border': '1px solid white'}) for col in pretty_names])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(
                        format_large_number(df_sorted.iloc[i][col]) if col in ['market_cap', 'volume', 'average_volume', 'last_non_reg_price']
                        else f"{df_sorted.iloc[i][col]:+.2f}%" if col == 'percent_change'
                        else df_sorted.iloc[i][col],
                        style={'border': '1px solid white'}
                    )
                    for col in columns
                ]) for i in range(min(len(df_sorted), max_rows))
            ])
        ], style={'color': 'white', 'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '40px'})
    ])


# Set the layout of the app
app.layout = html.Div([
    PAGE_TITLE,
    dcc.Tabs(id="index-tabs", value='sp500', children=[
        dcc.Tab(label='S&P 500', value='sp500'),
        dcc.Tab(label='NASDAQ 100', value='nasdaq'),
        dcc.Tab(label='List View S&P 500', value='listview_spx'),
        dcc.Tab(label='List View NASDAQ 100', value='listview_nasdaq'),
    ]),
    html.Div(id='content-container'),
    dcc.Interval(id='refresh-interval', interval=5 *
                 60 * 1000, n_intervals=0),  # 5 minutes
], style={'backgroundColor': 'rgb(66, 73, 75)', 'padding': '0px', 'margin': '0px'}) # this bg color sets the color when loading initially

# Title for tab name; Favicon for browser tab
app.title = "PF's 24/5 Stock Map"
app._favicon = ("assets/icon.ico")


# Define callback to update the graph
@app.callback(
    Output('content-container', 'children'),
    Input('index-tabs', 'value'),
    Input('refresh-interval', 'n_intervals')
)
def update_content(selected_index, n):
    global last_token, last_token_time

    ctx = callback_context
    print("Callback was triggered by:", ctx.triggered)

    now = time.time()
    if now - last_token_time > TOKEN_REFRESH_SECONDS:
        print("🔁 Refreshing token and figures...")
        last_token = get_robinhood_bearer_token()
        preload_figures(last_token)
        last_token_time = now
    else:
        print("✅ Using cached figures")

    if selected_index == 'sp500':
        return html.Div([
            dcc.Graph(figure=spx_fig, id='heatmap-graph', config={'responsive': True}),
            BOTTOM_CAPTION
        ])
    elif selected_index == 'nasdaq':
        return html.Div([
            dcc.Graph(figure=nasdaq_fig, id='heatmap-graph', config={'responsive': True}),
            BOTTOM_CAPTION
        ])
    elif selected_index == 'listview_spx':
        return html.Div([
            generate_table(spx_total_df, "S&P 500", len(
                spx_total_df) if spx_total_df is not None else 0),
            BOTTOM_CAPTION
        ])
    elif selected_index == 'listview_nasdaq':
        return html.Div([
            generate_table(nasdaq_total_df, "NASDAQ 100", len(
                nasdaq_total_df) if nasdaq_total_df is not None else 0),
            BOTTOM_CAPTION
        ])


# Do this at global level for gunicorn to pick up
last_token = get_robinhood_bearer_token()
while last_token is None:
    print("❌ Failed to get bearer token, retrying...")
    time.sleep(1)
    last_token = get_robinhood_bearer_token()

print("Bearer token retrieved successfully, preloading figures...")

preload_figures(last_token)  # preload both S&P 500 and Nasdaq heatmaps
last_token_time = time.time()  # Set initial token time


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)