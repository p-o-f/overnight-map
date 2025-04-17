import time
import requests
import pandas as pd
import re


def get_ticker_info(ticker):
    url = f"https://api.robinhood.com/quotes/{ticker}/"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}) # pretend to be a regular FireFox browser
    data = res.json()
    print(data)
    
    last_regular_trade_price = float(data['last_trade_price'])
    supports_overnight = data['last_non_reg_trade_price'] != data['last_extended_hours_trade_price'] # can use tickers like ASML, RNMBY, SFTBY to test this 
    overnight_price = float(data['last_non_reg_trade_price'])
    difference = last_regular_trade_price - overnight_price
    percent_change = (difference / last_regular_trade_price) * 100
    return last_regular_trade_price, supports_overnight, overnight_price, difference, percent_change


def get_sp500_symbols():
    url = 'https://www.wikitable2json.com/api/List_of_S%26P_500_companies?table=0'
    response = requests.get(url)
    data = response.json()[0]
    symbols = []
    for company in data[1:]: # the first element is the header, format is: SYMBOL / SECURITY / GICS SECTOR / GICS SUB-INDUSTRY / HEADQUARTERS LOCATION / DATE FIRST ADDED / CIK / FOUNDED
        symbols.append(company[0])
    return symbols
    

def get_nasdaq_symbols():   
    url = 'https://www.wikitable2json.com/api/Nasdaq-100?table=3'
    response = requests.get(url)
    data = response.json()[0]
    symbols = []
    for company in data[1:]: # the first element is the header, format is: COMPANY / TICKER / GICS Sector / GICS Sub Industry
        symbols.append(company[1])
    return symbols

nasdaq_symbols = get_nasdaq_symbols()
sp500_symbols = get_sp500_symbols()


class company_list:
    overnight_mode = False # universal class attribute

    def __init__(self, symbol):
        self.symbol = symbol
        
    def overnight_mode(self):
        overnight_mode = (not self.overnight_mode) # instance attribute
        
#print(get_ticker_info('ASML')) # test the function
print(get_ticker_info("QQQ")) # test the function




# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# # Set up headless Chrome
# chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--window-size=1920,1080")
# chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# driver = webdriver.Chrome(options=chrome_options)

# try:
#     # Navigate to the Robinhood page
#     driver.get("https://robinhood.com/us/en/stocks/QQQ/")
    
#     # Wait for the container element to be present
#     wait = WebDriverWait(driver, 10)
#         # Wait for at least 2 containers to be present
#     containers = wait.until(
#         EC.presence_of_all_elements_located((By.CLASS_NAME, "css-kjj0ot"))
#     )

#     if len(containers) >= 2:
#         second_container = containers[1]
        
#         # Get the first span inside the second container
#         first_span = second_container.find_element(By.XPATH, "./span[1]")
        
#         # Extract text and percentage
#         full_text = first_span.text
#         percentage_match = re.search(r'\(([^)]+)\)', full_text)
#         if percentage_match:
#             percentage = percentage_match.group(1)
#             print(f"Percentage change from second container: {percentage}")
#         else:
#             print(f"Full text from second container: {full_text}")
#     else:
#         print("Less than two containers with class 'css-kjj0ot' were found.")
    
# except Exception as e:
#     print(f"An error occurred: {e}")
    
# finally:
#     # Always close the driver
#     driver.quit()
    
