import requests
import pandas as pd
import numpy as np
from datetime import datetime, time
import time as time_module
import pytz

# Plotly 
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import callback_context
from dash import dash_table

# Performance optimizations
import asyncio
import aiohttp

# Constants for async requests
MAX_RETRIES = 3 # Arbitrary number of retries for failed requests
CONCURRENT_REQUESTS = 100  # Can be tuned higher/lower based on network stability
FIRST_LOAD = True  # Flag to indicate if this is the first load of the app

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # This is for Gunicorn to use

# Global constants
spx_fig = None
nasdaq_fig = None
spx_total_df = None
nasdaq_total_df = None
spx_table = None
nasdaq_table = None

BOTTOM_CAPTION = html.P([
    "Note: this data is sourced from Robinhood, though this site is not affiliated with Robinhood. ",
    "The provided data for should not be considered as any type of financial advice and may be inaccurate. ",
    "If you find this service useful, please consider supporting the server costs for this project by ",
    html.A("DONATING HERE.", href="https://buymeacoffee.com/pfdev", target="_blank", style={'color': 'lightblue'})
    ], style={'color': 'white', 'marginTop': '10px', 'fontSize': '12px', 'textAlign': 'center'})

PAGE_TITLE = html.H1(
    html.A("Overnight Stock Market Heat Map", href="https://buymeacoffee.com/pfdev", target="_blank", style={'color': 'lightblue'}),
    )

ny_tz = pytz.timezone("America/New_York")


def skipRefreshDueToWeekend():
    now = datetime.now(ny_tz)
    
    weekday = now.weekday()  # Monday is 0, Sunday is 6
    current_time = now.time()
    
    print("in skipRefreshDuetoWeekend(): now | weekday | current_time")
    print("----------------------------------------")
    print(now)
    print(weekday)
    print(current_time)
    print("----------------------------------------")
    
    # RH non 24-5 trading happens from Friday 8:00 PM until Sunday 5:00 PM 
    if weekday == 4:  # Friday
        if current_time > time(20, 0): # later than 8:00 PM on Friday
            return True
    elif weekday in [5]:  # Saturday
        return True
    elif weekday == 6:  # Sunday
        if current_time < time(17, 0): # before 5:00 PM on Sunday
            return True
    
    return False


def get_sp500_index_info():
    url = 'https://www.wikitable2json.com/api/List_of_S%26P_500_companies?table=0'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]: # the first element is the header, format is: SYMBOL / SECURITY / GICS SECTOR / GICS SUB-INDUSTRY / HEADQUARTERS LOCATION / DATE FIRST ADDED / CIK / FOUNDED
        stock_attributes.append([company[1], company[0], company[2], company[3]]) # <- Formal name, ticker symbol, sector, subsector 
    return stock_attributes
    

def get_nasdaq_index_info():   
    url = 'https://www.wikitable2json.com/api/Nasdaq-100?table=3'
    response = requests.get(url)
    data = response.json()[0]
    stock_attributes = []
    for company in data[1:]: # the first element is the header, format is: COMPANY / TICKER / GICS Sector / GICS Sub Industry
        stock_attributes.append([company[1], company[0], company[2], company[3]]) # <- Ticker symbol, formal name, sector, subsector FIXED HERE
    return stock_attributes


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
    print(f"❌ Final failure for URL: {url}")
    return None #TODO <-- fix this line later to do something more useful


async def fetch_symbol_metrics(session, symbol):
    try:
        basic_headers = {"User-Agent": "Mozilla/5.0"}
        complex_headers = {  # Taken from Network tab in Chrome DevTools; these are the headers that are required to get live quote data
            "authority": "bonfire.robinhood.com",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            #"authorization": f"Bearer {token}",
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
        if not id_data or 'instrument_id' not in id_data:
            print(f"Skipping symbol {symbol} due to missing instrument_id.")
            return None  # Skip this symbol if fetch_json failed

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
        
        dollar_change = round(float(last_non_reg_price) - float(adjusted_previous_close_price), 2)
        percent_change = round(dollar_change / float(adjusted_previous_close_price) * 100, 2)
        overnight = previous_close_price != last_non_reg_price
                
        # Process and return everything important as a tuple
        return instrument_id, market_cap, volume, average_volume, dollar_change, percent_change, last_trade_price, last_non_reg_price, extended_hours_price, previous_close_price, adjusted_previous_close_price, overnight

    except Exception as e:
        print(f"Error fetching instrument ID for {symbol}: {e}")


async def fetch_symbol_metrics_limited(session, symbol, sem):
    async with sem:
        return await fetch_symbol_metrics(session, symbol)


async def fetch_all_symbols(symbols):
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create a list of tasks for each symbol
        tasks = [fetch_symbol_metrics_limited(session, symbol, sem) for symbol in symbols]
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
        [0.5, palette[0]], [0.75, palette[1]], [0.875, palette[2]], [1.0, palette[3]]
    ]

    # Apply a power transformation to the market cap values
    power = 0.6
    dataframe['transformed_market_cap'] = np.power(dataframe['market_cap'], power)

    # Create formatted label with percent change
    dataframe['symbol_with_change'] = (
    "<span style='font-size: larger; color: white;'>" + dataframe['symbol'] + "</span><br>"
    "<span style='color: white;'>" + dataframe['percent_change'].map("{:+.2f}%".format) + "</span>"
        )

    # overnight_on = dataframe['overnight'].value_counts().iloc[0]
    # try:
    #     # If there is no overnight trading data, this will raise an IndexError
    #     overnight_off = dataframe['overnight'].value_counts().iloc[1]
    # except IndexError:
    #     # If there is no overnight trading data, set overnight_off to 0
    #     overnight_off = 0
    counts = dataframe['overnight'].value_counts()
    overnight_on = counts.get(True, 0)
    overnight_off = counts.get(False, 0)

    print(f"{map_title}, Overnight on: {overnight_on}, Overnight off: {overnight_off}")

    # Current time in New York
    ny_time = datetime.now(ny_tz)
    # Format like "5/13/2025, 6:14 AM, EST"
    formatted_time = ny_time.strftime("%m/%d/%Y, %I:%M %p, %Z").lstrip("0").replace(" 0", " ") # For cross-platform zero-stripping
    graph_top_title = f"{map_title} - Overnight Trading is currently enabled for " + str(overnight_on) + " symbols and disabled for " + str(overnight_off) + " symbols" 
    total_title = f"Last refreshed at: {formatted_time}"
    
    # Create Plotly treemap
    fig = px.treemap(
        dataframe,
        path=[px.Constant(graph_top_title), 'sector', 'subsector', 'symbol_with_change'], 
        values='transformed_market_cap',
        color='percent_change',
        color_continuous_scale=color_scale, 
        range_color=(-3.1, 3.1),
        custom_data=['name', 'last_non_reg_price', 'overnight', "volume", "average_volume"],
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
            formatted_volume = round(float(volume)/ 1000000, 2) 
            formatted_avg_volume = round(float(avg_volume)/ 1000000, 2) 
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
        hovertemplate=
            '<span style="color:white;">%{label}</span><br><br>' +
            #'<span style="color:white;">Market Cap: $%{value}</span><br>' + # TODO fix this to be proper units
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
        font_size=13, #13
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
            thickness=16,  # a bit thinner
            lenmode="fraction",
            len=0.5,        # 50% width
            yanchor="bottom",
            y=-0.1,        # moves it further down
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
            x=0.5,             # ← center horizontally
            xanchor='center'   # ← anchor the x=0.5 position at the center of the text
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
        color="white", # Text color for labels
        )
    )
    
    # Better view for mobile
    fig.update_layout(
        autosize=True,
        margin=dict(l=30, r=30, t=30, b=0),  # Remove margins
    )

    print("Fig created for " + map_title + "\n")
    return fig


def load_figures():
    start = time_module.time()
    
    global spx_fig, nasdaq_fig
    global spx_total_df, nasdaq_total_df
    global spx_table, nasdaq_table

    async def load_both_indices():
        # Fetch index metadata (blocking)
        spx_df = pd.DataFrame(get_sp500_index_info(), columns=["name", "symbol", "sector", "subsector"])
        nasdaq_df = pd.DataFrame(get_nasdaq_index_info(), columns=["name", "symbol", "sector", "subsector"])

        # Run both fetches concurrently
        spx_task = fetch_all_symbols(spx_df['symbol'].tolist())
        nasdaq_task = fetch_all_symbols(nasdaq_df['symbol'].tolist())
        spx_results, nasdaq_results = await asyncio.gather(spx_task, nasdaq_task)

        column_names = [
            "instrument_id", "market_cap", "volume", "average_volume", "dollar_change", "percent_change",
            "last_trade_price", "last_non_reg_price", "extended_hours_price", "previous_close_price",
            "adjusted_previous_close_price", "overnight"
        ]
        # --- Process S&P 500 ---
        spx_valid_data = [(i, r) for i, r in enumerate(spx_results) if r is not None]
        spx_valid_indices, spx_metrics = zip(*spx_valid_data) if spx_valid_data else ([], [])
        spx_metrics_df = pd.DataFrame(spx_metrics, columns=column_names)
        spx_df_valid = spx_df.iloc[list(spx_valid_indices)].reset_index(drop=True)

        spx_total = pd.concat([spx_df_valid, spx_metrics_df], axis=1)
        spx_total = spx_total[spx_total["symbol"] != "GOOGL"]
        # ------------------------

        # --- Process NASDAQ 100 ---
        nasdaq_valid_data = [(i, r) for i, r in enumerate(nasdaq_results) if r is not None]
        nasdaq_valid_indices, nasdaq_metrics = zip(*nasdaq_valid_data) if nasdaq_valid_data else ([], [])
        nasdaq_metrics_df = pd.DataFrame(nasdaq_metrics, columns=column_names)
        nasdaq_df_valid = nasdaq_df.iloc[list(nasdaq_valid_indices)].reset_index(drop=True)

        nasdaq_total = pd.concat([nasdaq_df_valid, nasdaq_metrics_df], axis=1)
        nasdaq_total = nasdaq_total[nasdaq_total["symbol"] != "GOOGL"]
        # ------------------------

        # Store results to global state
        return spx_total, nasdaq_total

    # Run the async loader
    spx_total_df, nasdaq_total_df = asyncio.run(load_both_indices())

    # Build figures after data is loaded
    spx_fig = create_heat_map(spx_total_df, "S&P 500")
    nasdaq_fig = create_heat_map(nasdaq_total_df, "NASDAQ 100")
    
    # Cache pre-rendered tables from df
    spx_table = generate_table(spx_total_df, "S&P 500", len(spx_total_df) if spx_total_df is not None else 0)
    nasdaq_table = generate_table(nasdaq_total_df, "NASDAQ 100", len(nasdaq_total_df) if nasdaq_total_df is not None else 0)
    print(f"load_figures() took {time_module.time() - start:.2f} seconds")

    
def generate_table(df, title, max_rows=30):
    df_sorted = df.sort_values(by="percent_change", ascending=False)

    # Columns to display and their headers
    columns = ['symbol', 'name', 'sector', 'percent_change', 'market_cap', 'volume', 'average_volume', 'last_non_reg_price']
    pretty_names = ['Symbol', 'Name', 'Sector', '% Change', 'Market Cap', 'Volume', 'Avg Volume', 'Last Price']

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

    #TODO can make this sortable by column later (optionally); also need to fix margins and aesthetic stuff
    return html.Div([
        html.H3(f"{title} - Top {max_rows} by % Change", style={'color': 'white', 'marginTop': '20px', 'marginleft': '20px', 'marginRight': '20px'}),
        html.Table([
            html.Thead(
                html.Tr([html.Th(col, style={'color': 'white', 'border': '1px solid white'}) for col in pretty_names])
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

    # Interval that only updates data (invisible)
    dcc.Interval(id='data-refresh-interval', interval=14 * 60 * 1000, n_intervals=0),  # 14 minutes
    
    # Interval that triggers tab re-render
    dcc.Interval(id='ui-refresh-interval', interval=15 * 60 * 1000, n_intervals=0),  # 15 minutes
    
    # Dummy div for triggering data refresh callback
    html.Div(id='data-refresh-dummy', style={'display': 'none'})
    
], style={'backgroundColor': 'rgb(66, 73, 75)', 'padding': '0px', 'margin': '0px'}) # this bg color sets the color when loading initially

# Define callbacks to update the graph
@app.callback( # Every 14 mins
    Output('data-refresh-dummy', 'children'),
    Input('data-refresh-interval', 'n_intervals')
)
def background_data_refresh(n):
    global FIRST_LOAD
    ret = f"background_data_refresh() triggered at {datetime.now()}"  # dummy output
    ctx = callback_context
    print("Callback in background_data_refresh() was triggered by:", ctx.triggered)
    
    # Skip this the first time, then don't refresh on subsequent callbacks
    if (not FIRST_LOAD and ctx.triggered[0]['prop_id'] == '.'):
        print("Loaded from another device or window, and this is NOT the first time the webapp has been loaded, so skipping refresh...")
        print()
        return ret
    
    if FIRST_LOAD:
        print("Loading figures due to first load...")
        load_figures()
        FIRST_LOAD = False
        print()
        return ret
    
    print(f"🔄 Refreshing data at 14-minute interval (n={n})")
    if not skipRefreshDueToWeekend():
        print("It wasn't a weekend, so refresh is allowed... calling load_figures()")
        load_figures()
    else:
        print("It was a weekend, so skipping refresh...")
    print()
    return ret

@app.callback( # Every 15 mins
    Output('content-container', 'children'),
    Input('index-tabs', 'value'),
    Input('ui-refresh-interval', 'n_intervals')
)
def update_content(selected_index, n):
    ctx = callback_context
    print("Callback in update_content() was triggered by:", ctx.triggered)
    print(f"Updating UI at 15-minute interval (aka without data refresh)... (n={n}), tab={selected_index}")
    print()

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
            spx_table,
            BOTTOM_CAPTION
        ])
    elif selected_index == 'listview_nasdaq':
        return html.Div([
            nasdaq_table,
            BOTTOM_CAPTION
        ])


if __name__ == "__main__":
    print("Starting with intitial call to load_figures() upon first run...")
    #load_figures()
    app.run(debug=True)