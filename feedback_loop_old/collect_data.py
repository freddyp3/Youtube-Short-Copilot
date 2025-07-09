"""script i used to build the training csvs.

pulls recent brawl stars shorts from the public youtube data api. public
data only, dont need oauth for this.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("YOUTUBE_KEY")


def search_shorts(q="brawl stars shorts", max_results=25):
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "key": KEY,
            "part": "snippet",
            "type": "video",
            "videoDuration": "short",
            "order": "viewCount",
            "relevanceLanguage": "en",
            "q": q,
            "maxResults": max_results,
        },
    ).json()
    return [v["id"]["videoId"] for v in r["items"]]


def main():
    ids = search_shorts()
    print("got", len(ids), "video ids")
    print(ids[:5])


if __name__ == "__main__":
    main()
