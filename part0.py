import tweepy
import csv

consumer_key = "O5laxklKItehppjSkXjxmCbOF"
consumer_secret = "DVLtN4zeZM8L1gWIiGDqp0SiekN77qME2Hmmk8RJb3TJu0mlDI"
access_token = "740565057243750401-bj45QLuy3tZwkQ7EWZsXnr6V253SWV3"
access_token_secret = "53wrJo9YK6ITb36nlpNIo2DvObIhj6ypnNMVuyKdF21Wx"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

# Define hashtag to search
hashtag = "#FeminismIsCancer -filter:retweets"

# Open CSV file to write
with open('tweets_FeminismIsCancer.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Tweet ID', 'Tweet', 'Created Time', 'Language'])

    # Iterate through each tweet in the search results
    for tweet in tweepy.Cursor(api.search_tweets, q=hashtag, lang='en', tweet_mode='extended').items(100):
        tweet_id = tweet.id_str
        tweet_text = tweet.full_text
        created_time = tweet.created_at
        language = tweet.lang

        # Write tweet data to CSV file
        writer.writerow([tweet_id, tweet_text, created_time, language])
