import numpy as np
from datetime import datetime
import requests

# ------------------------------------------------ Blockchain.info --------------------------------------------------- #

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

    # # Print
    # print("")
    # print(data['name'])
    # print("Per", data['period'])
    # print("Unit:", data['unit'])
    # print("Length:", len(total_number_of_transactions_list))

    # Return result
    return total_number_of_transactions_list


def return_data():
    # ------------------------------------------------ Index Statistics -------------------------------------------------- #

    # Average USD market price across major bitcoin exchanges.
    url_average_USD_price = "https://api.blockchain.info/charts/market-price?format=json&timespan=all"
    average_USD_price = url_to_vector(url_average_USD_price)
    print("average_USD_price:", average_USD_price)

    # The total USD value of bitcoin supply in circulation, as calculated by the daily average market price across major
    # exchanges.
    url_market_capitalization = "https://api.blockchain.info/charts/market-cap?format=json&timespan=all"
    market_capitalization = url_to_vector(url_market_capitalization)
    print("market_capitalization:", market_capitalization)

    # The total number of bitcoins that have already been mined; in other words, the current supply of bitcoins on the
    # network.
    url_BTC_in_circulation = "https://api.blockchain.info/charts/total-bitcoins?format=json&timespan=all"
    BTC_in_circulation = url_to_vector(url_BTC_in_circulation)
    print("BTC_in_circulation:", BTC_in_circulation)

    # The total USD value of tr…major bitcoin exchanges.
    url_USD_exchange_trade_volume = "https://api.blockchain.info/charts/trade-volume?format=json&timespan=all"
    USD_exchange_trade_volume = url_to_vector(url_USD_exchange_trade_volume)
    print("USD_exchange_trade_volume:", USD_exchange_trade_volume)

    # ------------------------------------------------- Block Details ---------------------------------------------------- #

    # The total size of all block headers and transactions in MB. Not including database indexes.
    url_blockchain_size = "https://api.blockchain.info/charts/blocks-size?format=json&timespan=all"
    blockchain_size = url_to_vector(url_blockchain_size)
    print("blockchain_size:", blockchain_size)

    # The average block size in MB.
    url_average_block_size = "https://api.blockchain.info/charts/avg-block-size?format=json&timespan=all"
    average_block_size = url_to_vector(url_average_block_size)
    print("average_block_size:", average_block_size)

    # The total number of blocks mined but ultimately not attached to the main Bitcoin blockchain.
    url_no_orphaned_blocks = "https://api.blockchain.info/charts/n-orphaned-blocks?format=json&timespan=all"
    no_orphaned_blocks = url_to_vector(url_no_orphaned_blocks)
    print("no_orphaned_blocks:", no_orphaned_blocks)

    # The average number of transactions per block.
    url_transactions_per_block = "https://api.blockchain.info/charts/n-transactions-per-block?format=json&timespan=all"
    transactions_per_block = url_to_vector(url_transactions_per_block)
    print("transactions_per_block:", transactions_per_block)

    # The median time for a transaction to be accepted into a mined block and added to the public ledger (note: only
    # includes transactions with miner fees).
    url_median_confirmation_time = "https://api.blockchain.info/charts/median-confirmation-time?format=json&timespan=all"
    median_confirmation_time = url_to_vector(url_median_confirmation_time)
    print("median_confirmation_time:", median_confirmation_time)

    # # Percentage of blocks signalling SegWit support
    # url_segwit_adoption = "https://blockchain.info/nl/charts/bip-9-segwit?timespan=all"
    # print("segwit_adoption:", url_to_vector(url_segwit_adoption))

    # Percentage of blocks signalling Bitcoin Unlimited support
    url_BTC_unlimited_support = "https://api.blockchain.info/charts/bitcoin-unlimited-share?format=json"
    BTC_unlimited_support = url_to_vector(url_BTC_unlimited_support)
    print("BTC_unlimited_support:", BTC_unlimited_support)

    # ---------------------------------------------- Mining Information -------------------------------------------------- #

    # The estimated number of tera hashes per second (trillions of hashes per second) the Bitcoin network is performing.
    url_hash_rate = "https://api.blockchain.info/charts/hash-rate?format=json&timespan=all"
    hash_rate = url_to_vector(url_hash_rate)
    print("hash_rate:", hash_rate)

    # A relative measure of how difficult it is to find a new block. The difficulty is adjusted periodically as a function
    # of how much hashing power has been deployed by the network of miners.
    url_difficulty = "https://api.blockchain.info/charts/difficulty?format=json&timespan=all"
    difficulty = url_to_vector(url_difficulty)
    print("difficulty:", difficulty)

    # The estimated number of tera hashes per second (trillions of hashes per second) the Bitcoin network is performing.
    url_miners_revenue = "https://api.blockchain.info/charts/miners-revenue?format=json&timespan=all"
    miners_revenue = url_to_vector(url_miners_revenue)
    print("miners_revenue:", miners_revenue)

    # The total value of all transaction fees paid to miners in BTC (not including the coinbase value of block rewards).
    url_total_transaction_fees = "https://api.blockchain.info/charts/transaction-fees?format=json&timespan=all"
    total_transaction_fees = url_to_vector(url_total_transaction_fees)
    print("total_transaction_fees:", total_transaction_fees)

    # The total value of all transaction fees paid to miners in USD (not including the coinbase value of block rewards).
    url_total_transaction_fees_USD = "https://api.blockchain.info/charts/transaction-fees-usd?format=json"
    total_transaction_fees_USD = url_to_vector(url_total_transaction_fees_USD)
    print("total_transaction_fees_USD:", total_transaction_fees_USD)

    # A chart showing miners revenue as percentage of the transaction volume.
    url_cost_per_transaction_percent = "https://api.blockchain.info/charts/cost-per-transaction-percent?format=json&timespan=all"
    cost_per_transaction_percent = url_to_vector(url_cost_per_transaction_percent)
    print("cost_per_transaction_percent:", cost_per_transaction_percent)

    # A chart showing miners revenue divided by the number of transactions.
    url_cost_per_transaction = "https://api.blockchain.info/charts/cost-per-transaction?format=json&timespan=all"
    cost_per_transaction = url_to_vector(url_cost_per_transaction)
    print("cost_per_transaction:", cost_per_transaction)

    # ----------------------------------------------- Network Activity --------------------------------------------------- #

    # The total number of unique addresses used on the Bitcoin blockchain.
    url_n_unique_addresses = "https://api.blockchain.info/charts/n-unique-addresses?format=json&timespan=all"
    n_unique_addresses = url_to_vector(url_n_unique_addresses)
    print("n_unique_addresses:", n_unique_addresses)

    # The number of daily confirmed Bitcoin transactions per day.
    url_n_transactions_per_day = "https://api.blockchain.info/charts/n-transactions?format=json&timespan=all"
    n_transactions_per_day = url_to_vector(url_n_transactions_per_day)
    print("n_transactions_per_day:", n_transactions_per_day)

    # Total number of transactions
    url_total_number_of_transactions = "https://api.blockchain.info/charts/n-transactions-total?format=json&timespan=all"
    total_number_of_transactions = url_to_vector(url_total_number_of_transactions)
    print("total_number_of_transactions:", total_number_of_transactions)

    # The number of Bitcoin transactions added to the mempool per second.
    url_transaction_rate = "https://api.blockchain.info/charts/transactions-per-second?format=json&timespan=all"
    transaction_rate = url_to_vector(url_transaction_rate)
    print("transaction_rate:", transaction_rate)

    # The number of transactions waiting to be confirmed.
    url_n_mempool_transactions = "https://api.blockchain.info/charts/mempool-count?format=json&timespan=all"
    n_mempool_transactions = url_to_vector(url_n_mempool_transactions)
    print("n_mempool_transactions:", n_mempool_transactions)

    # The rate at which the mempool is growing per second.
    url_mempool_size_growth = "https://api.blockchain.info/charts/mempool-growth?format=json&timespan=all"
    mempool_size_growth = url_to_vector(url_mempool_size_growth)
    print("mempool_size_growth:", mempool_size_growth)

    # The aggregate size of transactions waiting to be confirmed.
    url_mempool_size = "https://api.blockchain.info/charts/mempool-size?format=json&timespan=all"
    mempool_size = url_to_vector(url_mempool_size)
    print("mempool_size:", mempool_size)

    # The number of unspent Bitcoin transactions outputs, also known as the UTXO set size.
    url_utxo_count = "https://api.blockchain.info/charts/utxo-count?format=json&timespan=all"
    utxo_count = url_to_vector(url_utxo_count)
    print("utxo_count:", utxo_count)

    # The total number of Bitcoin transactions, excluding those involving any of the network's 100 most popular addresses.
    url_n_transactions = "https://api.blockchain.info/charts/n-transactions-excluding-popular?format=json&timespan=all"
    n_transactions = url_to_vector(url_n_transactions)
    print("n_transactions:", n_transactions)

    # The total number of Bitcoin transactions per day excluding those part of long transaction chains. There are many
    # legitimate reasons to create long transaction chains; however, they may also be caused by coin mixing or possible
    # attempts to manipulate transaction volume.
    url_n_transactions_exc_chains_longer_than_100 = "https://api.blockchain.info/charts/n-transactions-excluding-chains-longer-than-100?format=json&timespan=all"
    n_transactions_exc_chains_longer_than_100 = url_to_vector(url_n_transactions_exc_chains_longer_than_100)
    print("n_transactions_exc_chains_longer_than_100:", n_transactions_exc_chains_longer_than_100)

    # The total value of all transaction outputs per day (includes coins returned to the sender as change).
    url_output_value = "https://api.blockchain.info/charts/output-volume?format=json&timespan=all"
    output_value = url_to_vector(url_output_value)
    print("output_value:", output_value)

    # The Estimated Transaction Value in USD value.
    url_estimated_USD_transaction_value = "https://api.blockchain.info/charts/estimated-transaction-volume-usd?format=json&timespan=all"
    estimated_USD_transaction_value = url_to_vector(url_estimated_USD_transaction_value)
    print("estimated_USD_transaction_value:", estimated_USD_transaction_value)

    # Makes a matrix from features, matched by date
    def match_by_date(list_of_features):
        # Holds the date to match all data with
        list_of_dates = []

        # Fills the dates with dates of first list item
        for feature in list_of_features[0]:
            list_of_dates.append(feature[0])

        print(list_of_dates)

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

    # # All
    # matrix = match_by_date([average_USD_price, blockchain_size, n_transactions_per_day, average_block_size, no_orphaned_blocks,
    #                         transactions_per_block, median_confirmation_time, BTC_unlimited_support, hash_rate, difficulty,
    #                         miners_revenue, total_transaction_fees, total_transaction_fees_USD, cost_per_transaction,
    #                         cost_per_transaction_percent])

    matrix = match_by_date([average_USD_price, blockchain_size, n_transactions_per_day, average_block_size, no_orphaned_blocks,
                            transactions_per_block, median_confirmation_time, hash_rate, difficulty, miners_revenue, total_transaction_fees, cost_per_transaction])

    y = matrix[0]
    X = matrix[1:]

    return y, X