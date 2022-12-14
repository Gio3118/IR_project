import snscrape.modules.twitter as twitter
import pandas as pd


def get_tweets(query: str, limit: int = 5000) -> pd.DataFrame:
    holding = []

    for i, tweet in enumerate(twitter.TwitterSearchScraper(query).get_items()):
        if i > limit:
            break
        holding.append([tweet.user.username, tweet.date, tweet.content])

    df = pd.DataFrame(holding, columns=["user", "date", "content"])
    return df
