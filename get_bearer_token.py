from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import re

def get_robinhood_bearer_token():
    # Set up Chrome options for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Optional: Add user agent to mimic a real browser
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
    
    # Enable logging for network requests
    chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    
    print("Starting Chrome in headless mode...")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Navigate to a specific stock page
        print("Navigating to S&P 500 ETF page...")
        driver.get("https://robinhood.com/stocks/SPY")
        time.sleep(3)  # Wait for page to load
        
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

if __name__ == "__main__":
    # Replace with your Robinhood credentials

    token = get_robinhood_bearer_token()
    
    if token:
        print("\nBearer Token:")
        print(token)
    else:
        print("\nCould not find bearer token. Check credentials or try again.")