# 🌐 UNIVERSAL SENTIMENT ANALYSIS — Fixed Label Mapping

import pandas as pd
from transformers import pipeline
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import base64
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Pandas settings
pd.set_option('display.max_colwidth', None)

# Step 1: Load data
# ✅ Correct CSV export URL
url = 'https://docs.google.com/spreadsheets/d/1R3Q1JJdlpRYcoaQbNL7i4cgf5ORdioMCX2OUK8iEa9U/export?format=csv'
print("📥 Loading feedback data...")
df_raw = pd.read_csv(url, on_bad_lines='skip')
df_raw.columns = df_raw.columns.str.strip()

col_name = 'feedback_text' if 'feedback_text' in df_raw.columns else df_raw.columns[0]
df = df_raw[[col_name]].dropna().reset_index(drop=True)
df.rename(columns={col_name: 'Comment'}, inplace=True)
df['Comment'] = df['Comment'].astype(str).str.strip()
df = df[(df['Comment'] != '') & (df['Comment'] != 'nan')].reset_index(drop=True)

# Step 2: Load multilingual model
print("🚀 Loading multilingual model...")
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=model_name,
    truncation=True,
    max_length=512
)

# ✅ Universal label mapping — handles all formats
label_map = {
    '0': 'negative', '1': 'neutral', '2': 'positive',
    'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
    'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive'
}

def get_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        key = str(result['label']).strip().upper()  # e.g. 'POSITIVE'
        sentiment = label_map.get(key, 'unknown')
        return sentiment, result['score']
    except Exception as e:
        print(f"Error: {e}")
        return 'unknown', 0.0

# Step 4: Apply sentiment
print("🧠 Analyzing multilingual feedback...")
results = df['Comment'].apply(get_sentiment)
df['Sentiment'], df['Confidence'] = zip(*results)

# Add emoji
emoji_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐', 'unknown': '❓'}
df['Emoji'] = df['Sentiment'].map(emoji_map)

# Preview
df['Preview'] = df['Comment'].str.slice(0, 60) + df['Comment'].str.len().apply(lambda x: '...' if x > 60 else '')

# Step 5: Save results
output_file = 'multilingual_sentiment_results.csv'
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✅ Results saved to '{output_file}'")

# Step 6: Display
print("\n📋 Multilingual Feedback Analysis:")
print("=" * 80)
display(df[['Preview', 'Sentiment', 'Confidence', 'Emoji']])

# Summary
sentiment_counts = df['Sentiment'].value_counts()
total = len(df)
print(f"\n📊 Summary (Total: {total}):")
for sent, count in sentiment_counts.items():
    percent = round((count / total) * 100, 1)
    print(f"{sent.capitalize()}: {count} ({percent}%)")

# ✅ Charts
colors = {'positive': '#5cb85c', 'negative': '#d9534f', 'neutral': '#f0ad4e', 'unknown': '#adb5bd'}
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
ax[0].bar(sentiment_counts.index, sentiment_counts.values,
          color=[colors[s] for s in sentiment_counts.index])
ax[0].set_title("Sentiment Distribution")
ax[0].set_ylabel("Count")
for i, (sent, count) in enumerate(sentiment_counts.items()):
    ax[0].text(i, count + 0.05, str(count), ha='center', va='bottom')

# Pie chart
ax[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
          colors=[colors[s] for s in sentiment_counts.index])
ax[1].set_title("Sentiment Breakdown")

plt.tight_layout()
plt.show()

# ✅ Universal Download
print("\n" + "="*80)
print(" DOWNLOAD RESULTS ")
print("="*80)

try:
    from google.colab import files
    print("👇 Click to download:")
    files.download(output_file)
except:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="text/csv;base64,{b64}" download="{output_file}"><strong>📥 Click to download {output_file}</strong></a>'
    display(HTML(href))