import time
from collections import OrderedDict

import requests

# -------------------------------------------- Historical Bitcoin values --------------------------------------------- #

# Start and end dates for the API to retrieve historic Bitcoin prices
# start = "2010-07-17" # Earliest date available
start = "2010-07-17"
end = (time.strftime("%Y-%m-%d")) # Current date
#
# The URL of the api
url = "http://api.coindesk.com/v1/bpi/historical/close.json?start="+start+"&end="+end

# Request URL
r = requests.get(url=url)

# Load JSON ordered by date
data = r.json(object_pairs_hook=OrderedDict)
bitcoin_price_date = data['bpi']

# Make list
bitcoin_price_date_list = []

# Put values in list
# with the first column containing the date and
# the second column containing the Bitcoin price in USD
for key, value in bitcoin_price_date.items():
    bitcoin_price_date_list.append([key, value])

# Print
print(bitcoin_price_date_list)
print("Length:", len(bitcoin_price_date))