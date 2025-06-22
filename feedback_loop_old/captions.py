import os
import anthropic
import xgboost as xgb

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])

model = xgb.XGBRegressor()
model.load_model("caption_model.json")

PROMPT = """write a caption for a brawl stars youtube short.

the video is about: {desc}

reply with ONLY the caption text, nothing else"""


def write_caption(desc):
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": PROMPT.format(desc=desc)}],
    )
    return r.content[0].text.strip()


def caption_features(text):
    """same idea as the thumbnail features but for text"""
    return [
        len(text),
        text.count("!"),
        sum(1 for c in text if ord(c) > 10000),  # emoji-ish
        int("?" in text),
        int(any(w in text.lower() for w in ["free", "insane", "crazy", "new", "win"])),
    ]


def make_caption(desc, tries=5):
    """same acceptance loop as thumbnails, claude writes, xgboost judges"""
    for i in range(tries):
        cap = write_caption(desc)
        s = model.predict([caption_features(cap)])[0]
        print("try", i + 1, "score", s, "->", cap)
    return cap


if __name__ == "__main__":
    print(make_caption("a 1v3 clutch in knockout"))
