# %%
import praw
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import sklearn

# %%
clientid = 'AfK4Ul3-lluVw6hv6sN7fA'
client_secret = '84Sa2URTsDJVBC3LYzI1tgEfYt9UTw'
user_agent = "script:reddit-sentiment-analyzer:v1.0 (by u/capps025)"

# %%
reddit = praw.Reddit(
    client_id=clientid,
    client_secret=client_secret,
    user_agent=user_agent
)

# %%
print(reddit.read_only)
# Output: True

# %%
tar_sub = 'datacenter'

# %%
import pandas as pd

battery_terms = ['battery', 'batteries', 'lithium', 'li-ion', 'sodium', 'lead', 'acid', 'ups']
data = []
seen_comments = set()

for term in battery_terms:
    for submission in reddit.subreddit(tar_sub).search(term, sort="top", time_filter="all"):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if comment.id in seen_comments:
                continue
            text = comment.body.lower()
            matched_terms = [bt for bt in battery_terms if bt in text]
            if matched_terms:
                data.append({
                    'comment_id': comment.id,
                    'author': str(comment.author),
                    'text': text,
                    'matched_terms': matched_terms,
                    'submission_id': submission.id,
                    'submission_title': submission.title
                })
                seen_comments.add(comment.id)

# Convert to DataFrame
df = pd.DataFrame(data)


# %% [markdown]
# # Begin sentiment Analysis

# %% [markdown]
# - We'll use the `VADER` sentiment analyzer to conduct out sentiment analysis
#     - `VADER` will score each word in the comment
#         - good = + 1.9
#         - good!!!! = +2.3
#         - bad = - 2.5
#         - Extremely Bad = -3
#         - awesome = +3.1
#     - Output from `VADER` will have the proprtion of posivtive and negative sentment in the text and produced a normalized value `compund`
#     - We'll apply the label marking it as net positive or net negative response based on the `compound` score

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


# %%
# Apply VADER to each comment's text
df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x))


# %%
df = pd.concat([df.drop('sentiment', axis=1), df['sentiment'].apply(pd.Series)], axis=1)


# %%
def get_label(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment_label'] = df['compound'].apply(get_label)


# %%
df.head(3)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Simulate a sample of the df for visualization
data = {
    'matched_terms': df['matched_terms'],
    'sentiment_label': df['sentiment_label']
}

df = pd.DataFrame(data)

# Explode matched_terms to allow grouping by single term
df_exploded = df.explode('matched_terms')

# Count sentiment labels per battery type
sentiment_counts = df_exploded.groupby(['matched_terms', 'sentiment_label']).size().unstack(fill_value=0)

# Plot the sentiment distribution per battery type
sentiment_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution by Battery Type')
plt.xlabel('Battery Type')
plt.ylabel('Number of Comments')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')

plt.show()


