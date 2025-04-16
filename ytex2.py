import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re


# ----------------------------
# CONFIGURATION
# ----------------------------
API_KEY = 'AIzaSyBME_qa8VrwNx_sWxbsHd-TpCbiKl6Oz74'
CHANNEL_ID = 'UCfM3zsQsOnfWNUppiycmBuw'
MAX_RESULTS = 20  # Number of videos to scrape

# ----------------------------
# API Setup
# ----------------------------
youtube = build('youtube', 'v3', developerKey=API_KEY)


# ----------------------------
# Get video IDs from channel
# ----------------------------
def get_video_ids(channel_id, max_results=MAX_RESULTS):
    video_ids = []
    request = youtube.search().list(
        part='id',
        channelId=channel_id,
        maxResults=max_results,
        order='date'
    )
    response = request.execute()

    for item in response['items']:
        if item['id']['kind'] == 'youtube#video':
            video_ids.append(item['id']['videoId'])
    return video_ids


# ----------------------------
# Get video data
# ----------------------------
def get_video_details(video_ids):
    videos = []
    for video_id in video_ids:
        request = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        )
        response = request.execute()
        for item in response['items']:
            snippet = item['snippet']
            stats = item['statistics']
            videos.append({
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet['description'],
                'published_at': snippet['publishedAt'],
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0))
            })
    return pd.DataFrame(videos)


# ----------------------------
# Clean & preprocess text
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.lower().strip()
    return text


# ----------------------------
# Sentiment analysis
# ----------------------------
def sentiment_score(text):
    if text:
        return TextBlob(text).sentiment.polarity
    return 0


# ----------------------------
# Main analysis pipeline
# ----------------------------
def analyze_channel(channel_id):
    print("Fetching video IDs...")
    video_ids = get_video_ids(channel_id)

    print("Fetching video details...")
    df = get_video_details(video_ids)

    print("Cleaning text...")
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)

    print("Analyzing sentiment...")
    df['title_sentiment'] = df['clean_title'].apply(sentiment_score)
    df['description_sentiment'] = df['clean_description'].apply(sentiment_score)

    print("\n📊 Basic Statistics:")
    print(df[['views', 'likes', 'comments', 'title_sentiment', 'description_sentiment']].describe())

    print("📈 Creating Visualizations...")

    # Views per video
    plt.figure(figsize=(12, 6))
    sns.barplot(x='title', y='views', data=df.sort_values('views', ascending=False))
    plt.xticks(rotation=90)
    plt.title("Views per Video")
    plt.tight_layout()
    plt.show()

    # Likes vs Views scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='likes', y='views', hue='title_sentiment', size='comments', data=df)
    plt.title("Engagement vs Sentiment")
    plt.tight_layout()
    plt.show()

    return df

def save_data(df, filename='youtube_data'):
    df.to_csv(f'{filename}.csv', index=False)
    df.to_excel(f'{filename}.xlsx', index=False)
    print(f" Data saved as {filename}.csv and {filename}.xlsx")



# ----------------------------
# Run Program
# ----------------------------
if __name__ == '__main__':
    final_df = analyze_channel(CHANNEL_ID)
    save_data(final_df, 'youtube_analysis')
