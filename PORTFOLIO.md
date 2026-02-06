# Portfolio Entry: Overnight Map

## Project Overview

**Overnight Map** is a real-time financial data visualization web application that displays S&P 500 and NASDAQ-100 stock market heat maps with 24/5 trading data. Unlike traditional market maps that only show regular trading hours, this tool includes extended hours and overnight trading data, providing investors with a comprehensive view of market movements.

- **Live Site:** [https://www.247map.app](https://www.247map.app)
- **Repository:** [https://github.com/p-o-f/overnight-map](https://github.com/p-o-f/overnight-map)
- **Status:** Production (Live)
- **Year:** 2024

## Key Features

### Core Functionality
- **Real-time Heat Map Visualization**: Interactive treemap displays for both S&P 500 and NASDAQ-100 indices
- **24/5 Trading Data**: Includes extended hours and overnight trading sessions, not just regular market hours
- **Multiple View Modes**: Heat map visualization and detailed list view with sortable metrics
- **Automatic Updates**: Data refreshes every 14 minutes during active trading periods
- **Weekend-Aware Logic**: Intelligently skips data refresh when markets are closed (Friday 8 PM - Sunday 8 PM ET)

### User Experience
- **Responsive Design**: Mobile-friendly interface that scales properly on all devices
- **Interactive Navigation**: Tab-based interface for switching between indices and view modes
- **Color-Coded Indicators**: Visual representation of stock performance with intuitive color schemes
- **Performance Metrics**: Displays percentage change, dollar change, volume, and market cap for each stock

### Technical Features
- **Async Data Processing**: Concurrent fetching of data for 600+ stocks with configurable limits
- **Performance Optimization**: In-memory caching and client-side rendering for fast page loads
- **Visitor Analytics**: IP-based geolocation tracking with logging for site analytics
- **Error Handling**: Retry logic with exponential backoff for failed API requests

## Technical Stack

### Frontend
- **Dash** - Python web framework for building reactive web applications
- **Plotly.js** - Interactive visualization library for heat maps
- **Bootstrap** (dash-bootstrap-components) - Responsive UI components
- **Custom CSS** - Styled interface with dark theme

### Backend
- **Python 3.x** - Core programming language
- **Flask** - Web server framework
- **Gunicorn** - Production WSGI server
- **Asyncio** - Asynchronous programming for concurrent operations
- **aiohttp** - Async HTTP client for API requests

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **pytz** - Timezone handling (Eastern Time awareness)

### External APIs
- **Robinhood API** - Real-time stock quotes and market data
- **WikiTable2JSON** - S&P 500 and NASDAQ-100 index compositions
- **IPInfo.io** - Visitor geolocation data

### Infrastructure
- **Google Cloud Platform** - Compute Engine VM instance
- **Linux** - Production server environment
- **Git** - Version control

## Architecture & Implementation

### Data Pipeline
1. **Index Composition Fetching**: Retrieves current list of stocks from Wikipedia tables
2. **Multi-stage Data Collection** (per stock):
   - Fetch instrument ID from Robinhood quotes API
   - Retrieve fundamental data (market cap, volume)
   - Get live quote data (current price, previous close, extended hours price)
3. **Data Processing**: Calculate percentage changes, overnight indicators, and aggregations
4. **Visualization Generation**: Create Plotly treemap figures grouped by sector and sub-sector
5. **Caching**: Store processed data in memory to reduce redundant API calls

### Key Technical Decisions

#### Asynchronous Architecture
- **30 concurrent requests**: Balanced for optimal throughput without triggering rate limits
- **Semaphore-based concurrency control**: Prevents overwhelming the API endpoints
- **Exponential backoff**: Retry failed requests with increasing delays (max 3 retries)

#### Performance Optimizations
- **Client-side rendering**: Tab switching doesn't trigger data refresh, only UI updates
- **14-minute refresh cycle**: Balances data freshness with API usage
- **Memory caching**: Stores processed figures globally to avoid recomputation
- **Conditional updates**: Weekend detection prevents unnecessary API calls

#### Market Hours Logic
```python
# Sophisticated weekend detection
- Friday after 8:00 PM ET → Skip refresh
- All day Saturday → Skip refresh  
- Sunday before 8:00 PM ET → Skip refresh
- All other times → Refresh enabled
```

## Challenges & Solutions

### Challenge 1: Rate Limiting with 600+ Stocks
**Problem**: Fetching real-time data for 600+ stocks simultaneously could trigger API rate limits  
**Solution**: Implemented controlled concurrency with 30 concurrent requests, retry logic with exponential backoff, and connection pooling via aiohttp TCPConnector

### Challenge 2: Extended Hours Data Availability
**Problem**: Standard APIs often don't provide extended hours and overnight trading data  
**Solution**: Leveraged Robinhood's bonfire API endpoint with specific headers to access after-hours and pre-market data

### Challenge 3: Mobile Responsiveness
**Problem**: Complex treemap visualizations rendered poorly on mobile devices  
**Solution**: Added viewport meta tags, made Plotly graphs responsive, and implemented proper scaling with CSS

### Challenge 4: Multi-device State Management
**Problem**: Multiple users/sessions could cause conflicting global state  
**Solution**: Implemented session management with Flask sessions and added logic to detect first-time loads vs. subsequent refreshes

### Challenge 5: Server Costs and Uptime
**Problem**: Running 24/7 on GCP incurs costs, but manual restarts are inconvenient  
**Solution**: Currently using tmux for persistent sessions; planned improvements include systemd for auto-restart on crash

## Data Flow Diagram

```
User Request → Dash App
    ↓
Check if Weekend (skip if yes)
    ↓
Fetch Index Compositions (Wikipedia) → S&P 500 & NASDAQ-100 lists
    ↓
Async Fetch (30 concurrent) → For each stock:
    ├→ Robinhood Quotes API (Instrument ID)
    ├→ Robinhood Fundamentals API (Market Cap, Volume)
    └→ Robinhood Bonfire API (Live Quotes, Extended Hours)
    ↓
Data Processing (Pandas/NumPy)
    ├→ Calculate % changes
    ├→ Determine overnight status
    └→ Group by sector/subsector
    ↓
Visualization (Plotly Treemap)
    ├→ Color code by performance
    ├→ Size by market cap
    └→ Hierarchical grouping
    ↓
Cache in Memory
    ↓
Render to User
```

## Project Metrics

- **Stocks Tracked**: ~600 (S&P 500 + NASDAQ-100 with overlap)
- **API Calls per Refresh**: ~1,800 (3 endpoints × 600 stocks)
- **Refresh Frequency**: 14 minutes
- **Concurrent Requests**: 30
- **Average Response Time**: ~10-15 seconds for full data refresh
- **Uptime**: 24/5 (Monday-Friday including extended hours)

## Completed Improvements
- ✅ Client-side rendering for tab switching (improved performance)
- ✅ Weekend-aware refresh logic (reduced unnecessary API calls)
- ✅ Mobile scaling fixes (responsive design)
- ✅ Local memory caching (faster page loads)

## Future Enhancements

### Planned Features
1. **Database Integration**: PostgreSQL for storing visitor analytics and historical data
2. **Systemd Service**: Auto-start on boot and automatic crash recovery
3. **Cloud Logging**: GCP logging integration for remote monitoring without SSH
4. **Historical Views**: Time-series data for trend analysis
5. **User Accounts**: Custom watchlists and personalized dashboards
6. **Alert System**: Notifications for significant market movements

### Technical Improvements
- WebSocket integration for real-time updates
- Redis caching layer for better scalability
- Docker containerization for easier deployment
- Automated testing suite
- API rate limit monitoring and alerts

## Screenshots

### S&P 500 Heat Map
![S&P 500 Heat Map](https://github.com/user-attachments/assets/29f388d5-c883-4322-8f7a-cf39875b97ff)
*Interactive treemap showing S&P 500 stocks grouped by sector, colored by performance*

### NASDAQ-100 Heat Map
![NASDAQ-100 Heat Map](https://github.com/user-attachments/assets/a061112e-0a63-419a-93c5-5c64ee9fd3c1)
*Real-time visualization of NASDAQ-100 stocks with overnight trading data*

## Inspiration

This project was inspired by [Finviz](https://finviz.com/map.ashx) market maps but enhanced to include 24/5 trading data, providing a more complete picture of market activity that includes after-hours and overnight sessions.

## Deployment

The application is hosted on a Google Cloud Platform Compute Engine instance:
- **Server**: Linux VM
- **Web Server**: Gunicorn WSGI server
- **Port**: 8080 (proxied to port 80/443)
- **Domain**: 247map.app

## What I Learned

### Technical Skills
- Asynchronous programming patterns in Python with asyncio and aiohttp
- Working with financial APIs and handling real-time market data
- Building interactive data visualizations with Plotly
- Optimizing web applications for performance and cost-efficiency
- Deploying and managing applications on Google Cloud Platform

### Domain Knowledge
- Stock market structure (regular hours, extended hours, overnight trading)
- Market indices composition (S&P 500, NASDAQ-100)
- Trading hours and holiday schedules
- Financial data sources and their limitations

### Best Practices
- Importance of retry logic and error handling in production systems
- Balancing data freshness with API rate limits and costs
- Responsive design considerations for data-heavy applications
- State management in multi-user web applications

## Support

If you find this service useful, please consider supporting the server costs:
[☕ Buy Me a Coffee](https://buymeacoffee.com/pfdev)

---

## Code Highlights

### Async Data Fetching
```python
async def fetch_all_symbols(symbols):
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_symbol_metrics_limited(session, symbol, sem) 
                 for symbol in symbols]
        results = await asyncio.gather(*tasks)
    return results
```

### Weekend Detection Logic
```python
def skipRefreshDueToWeekend():
    now = datetime.now(ny_tz)
    weekday = now.weekday()
    current_time = now.time()
    
    if weekday == 4 and current_time > time(20, 0):  # Friday after 8 PM
        return True
    elif weekday == 5:  # Saturday
        return True
    elif weekday == 6 and current_time < time(20, 0):  # Sunday before 8 PM
        return True
    return False
```

## Contact

For questions or collaboration opportunities, visit the [GitHub repository](https://github.com/p-o-f/overnight-map).

---

**Tags**: Python, Dash, Plotly, Finance, Real-time Data, Visualization, Async Programming, GCP, API Integration, Data Science
