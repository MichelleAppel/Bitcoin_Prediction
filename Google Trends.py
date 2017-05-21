from pytrends.request import TrendReq

# Enter your own credentials
google_username = "bitcoin.predictor@gmail.com"
google_password = "..."

# Login to Google. Only need to run this once, the rest of requests will use the same session.
pytrend = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)

# Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
pytrend.build_payload(kw_list=['Bitcoin'])

# Interest Over Time
interest_over_time_df = pytrend.interest_over_time()

# Keys and values as list
keys = interest_over_time_df.index.tolist()
values = interest_over_time_df.values.T.tolist()[0]

interest_over_time_list = []

for key, value in zip(keys, values):
    interest_over_time_list.append([str(key.date()), value])

print(interest_over_time_list)
print("Length:", len(interest_over_time_list))