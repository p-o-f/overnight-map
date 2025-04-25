# random stuff here for testing idk
import requests
url = "https://api.robinhood.com/marketdata/fundamentals/81733743-965a-4d93-b87a-6973cb9efd34/?bounds=trading&include_inactive=true"
#requests.get(url, headers={"User-Agent": "Mozilla/5.0"}) # pretend to be a regular FireFox browser
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
data = response.json()
print(data["market_cap"])
#print(response["market_cap"])