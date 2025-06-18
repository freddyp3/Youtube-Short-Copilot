import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])

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


if __name__ == "__main__":
    print(write_caption("a 1v3 clutch in knockout"))
