import numpy as np
from datetime import datetime
import requests

import matplotlib.pyplot as plt

# ------------------------------------------------ Blockchain.info --------------------------------------------------- #

def return_data():

    # Calls the api, converts JSON data to vector
    def url_to_vector(url):
        # Request URL
        r = requests.get(url=url)

        # Load JSON
        data = r.json()
        total_number_of_transactions = data['values']

        # Make list
        total_number_of_transactions_list = []

        # Put values in list
        for item in total_number_of_transactions:
            # Convert from unix timestamp to date
            total_number_of_transactions_list.append([datetime.fromtimestamp(int(item['x'])).strftime('%Y-%m-%d %H:%M:%S'), item['y']])

        # Print
        # print("")
        # print(data['name'])
        # print("Per", data['period'])
        # print("Unit:", data['unit'])
        # print("Length:", len(total_number_of_transactions_list))

        # Return result
        return total_number_of_transactions_list


    # ------------------------------------------------ Index Statistics -------------------------------------------------- #

    # Average USD market price across major bitcoin exchanges.
    url_average_USD_price = "https://api.blockchain.info/charts/market-price?format=json&timespan=all"
    average_USD_price = url_to_vector(url_average_USD_price)

    # ------------------------------------------------- Block Details ---------------------------------------------------- #

    # The total size of all block headers and transactions in MB. Not including database indexes.
    url_blockchain_size = "https://api.blockchain.info/charts/blocks-size?format=json&timespan=all"
    blockchain_size = url_to_vector(url_blockchain_size)

    # The average block size in MB.
    url_average_block_size = "https://api.blockchain.info/charts/avg-block-size?format=json&timespan=all"
    average_block_size = url_to_vector(url_average_block_size)

    # print("average_USD_price:", average_USD_price)
    # print("blockchain_size:", blockchain_size)
    # print("average_block_size:", average_block_size)

    def match_on_date(list_of_features):
        # Holds the date to match all data with
        list_of_dates = []

        # Fills the dates with dates of first list item
        for feature in list_of_features[0]:
            list_of_dates.append(feature[0])

        matrix = []

        for features in list_of_features:
            vector = []
            for feature in features:
                for date in list_of_dates:
                    match_found = False
                    if feature[0] == date:
                        match_found = True
                        vector.append(feature[1])
                        break
                if not match_found:
                    vector.append("NaN")
            matrix.append(vector)
        return matrix

    matrix = match_on_date([average_USD_price, blockchain_size, average_block_size])

    y = matrix[0]
    X = matrix[1:]

    return np.array(matrix)

return_data()

matrix = return_data()


# # Show plots of data
# plt.plot(y)
# plt.show()
#
# plt.plot(X[0], y)
# plt.show()
#
# plt.scatter(X[1], y)
# plt.show()
#
# plt.scatter(X[0], X[1])
# plt.show()