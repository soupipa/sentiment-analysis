import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

print("🚀 Loading sentiment model...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Install with: pip install torch transformers")
    raise


label_map = {
    '0': 'negative', '1': 'neutral', '2': 'positive',
    'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
    'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive'
}


url = 'https://docs.google.com/spreadsheets/d/1Af6viO1UJIGucH4MdQ8ZLGxLZKFHoc0e4SPeu4Dhqks/export?format=csv'
print("📥 Loading feedback data...")
df = pd.read_csv(url)
df.columns = df.columns.str.strip()
col_name = 'feedback_text' if 'feedback_text' in df.columns else df.columns[0]

df = df[[col_name]].dropna().reset_index(drop=True)
df.rename(columns={col_name: 'feedback_text'}, inplace=True)
df['feedback_text'] = df['feedback_text'].astype(str).str.strip()
df = df[(df['feedback_text'] != '') & (df['feedback_text'] != 'nan')].reset_index(drop=True)

def get_sentiment(text):
    text_lower = text.lower().strip()

    
    neutral_indicators = [
        'nothing special',
        'was acceptable',
        'was okay',
        'is acceptable',
        'is okay',
        'everything was okay',
        'just average',
        'not bad but not great',
        'mediocre',
        'it is what it is'
    ]
    if any(phrase in text_lower for phrase in neutral_indicators):
        return 'neutral'

    
    if any(pos in text_lower for pos in ['excellent', 'love', 'amazing', 'fantastic', 'great', 'recommend', 'very satisfied']):
        return 'positive'

    
    if any(neg in text_lower for neg in ['late', 'crashing', 'crash', 'frustrating', 'below expectations', 'quality was below']):
        return 'negative'

    
    if 'but' in text_lower:
        if any(word in text_lower for word in ["didn't", "not", "problem", "solve", "fix"]):
            return 'negative'

    
    try:
        result = sentiment_pipeline(text)
        first = result[0]

        if isinstance(first, dict):
            label = first['label']
        elif isinstance(first, list) and len(first) > 0:
            label = first[0]['label']
        else:
            return 'neutral'

        key = str(label).strip()
        return label_map.get(key, 'neutral')

    except Exception as e:
        print(f"Model error: {e}")
        return 'neutral'


print("🧠 Analyzing sentiment with smart rules...")
df['sentiment'] = df['feedback_text'].apply(get_sentiment)


print("\n📝 Feedback Sentiment Analysis:")
print("=" * 80)
for _, row in df.iterrows():
    print(f"{row['feedback_text']:<60} | {row['sentiment'].upper():>10}")


sentiment_counts = df['sentiment'].value_counts()
total = len(df)

print(f"\n📊 Summary Statistics (Total: {total}):")
print("-" * 50)
for sent, count in sentiment_counts.items():
    percent = round((count / total) * 100, 1)
    print(f"{sent.capitalize():<10}: {count} ({percent}%)")


colors = {'positive': '#5cb85c', 'negative': '#d9534f', 'neutral': '#f0ad4e'}

fig, ax = plt.subplots(1, 2, figsize=(14, 6))


bars = ax[0].bar(sentiment_counts.index, sentiment_counts.values,
                 color=[colors.get(s, '#adb5bd') for s in sentiment_counts.index])
ax[0].set_title("Sentiment Distribution")
ax[0].set_ylabel("Count")
for bar in bars:
    height = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)


ax[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
          colors=[colors.get(s, '#adb5bd') for s in sentiment_counts.index])
ax[1].set_title("Sentiment Breakdown")

plt.tight_layout()
plt.show()