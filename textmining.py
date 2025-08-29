# 🚀 Customer Feedback Sentiment Analysis (Colab Version)
# With Save/Reload, Charts, and Styled Output

import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Pandas settings
pd.set_option('display.max_colwidth', None)

# Step 1: Load raw data
url = 'https://docs.google.com/spreadsheets/d/1Af6viO1UJIGucH4MdQ8ZLGxLZKFHoc0e4SPeu4Dhqks/export?format=csv'
print("📥 Step 1: Loading raw data...")
df_raw = pd.read_csv(url, header=0, on_bad_lines='skip')

print(f"   → Loaded {len(df_raw)} rows initially")

# Step 2: Clean column names
print("🔧 Step 2: Cleaning column names...")
df_raw.columns = df_raw.columns.str.strip().str.lower()
print(f"   → Columns found: {list(df_raw.columns)}")

# Find feedback column
col_candidates = [col for col in df_raw.columns if 'feedback' in col or 'text' in col or 'comment' in col]
col_name = col_candidates[0] if col_candidates else df_raw.columns[0]
print(f"   → Using column: '{col_name}' for analysis")

# Step 3: Extract and clean
df = df_raw[[col_name]].copy()
df.rename(columns={col_name: 'Comment'}, inplace=True)
df['Comment'] = df['Comment'].astype(str).str.strip()

# Track valid rows
df['is_empty'] = df['Comment'] == ''
df['is_nan'] = df['Comment'].isna()
df['is_valid'] = ~(df['is_empty'] | df['is_nan']) & (df['Comment'] != 'nan')

print(f"📊 Before cleaning: {len(df)} comments")
print(f"   → Valid: {df['is_valid'].sum()}")
print(f"   → Empty: {df['is_empty'].sum()}")
print(f"   → NaN: {df['is_nan'].sum()}")

df = df[df['is_valid']].reset_index(drop=True)
df.drop(columns=['is_empty', 'is_nan', 'is_valid'], inplace=True)
print(f"✅ After cleaning: {len(df)} comments")

# Step 4: Load model
print("\n🚀 Step 3: Loading sentiment model...")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure to run: !pip install torch transformers")
    raise

# Label mapping
label_map = {
    '0': 'negative', '1': 'neutral', '2': 'positive',
    'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'
}

# Step 5: Sentiment function
def get_sentiment(text):
    text_lower = text.lower()

    # Rule 1: Neutral
    if any(phrase in text_lower for phrase in ['nothing special', 'was okay', 'was acceptable', 'everything was okay']):
        return 'NEUTRAL'

    # Rule 2: Positive
    if any(word in text_lower for word in ['excellent', 'love', 'great', 'amazing', 'recommend']):
        return 'POSITIVE'

    # Rule 3: Negative
    if any(word in text_lower for word in ['late', 'crashing', 'frustrating', 'below expectations']):
        return 'NEGATIVE'

    # Rule 4: Mixed feedback with negative outcome
    if 'but' in text_lower and any(word in text_lower for word in ["didn't", "not", "problem"]):
        return 'NEGATIVE'

    # Fallback to model
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label'] if isinstance(result, dict) else result[0]['label']
        return label_map.get(str(label).strip(), 'UNKNOWN').upper()
    except Exception as e:
        return 'UNKNOWN'

# Step 6: Apply sentiment
print("🧠 Step 4: Applying sentiment analysis...")
df['Sentiment'] = df['Comment'].apply(get_sentiment)

# Step 7: Save to CSV
output_file = 'sentiment_results.csv'
df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n✅ Results saved to '{output_file}'")

# Step 8: Read back from CSV
print(f"🔁 Step 5: Reading from '{output_file}'...")
df_reloaded = pd.read_csv(output_file)


# ✅ BONUS: Add Short Comment Preview
df_reloaded['Preview'] = df_reloaded['Comment'].str.slice(0, 60) + df_reloaded['Comment'].str.len().apply(lambda x: '...' if x > 60 else '')

# ✅ BONUS: Reorder columns
df_display = df_reloaded[['Preview', 'Sentiment']]

# Step 9: Display in Colab (Beautifully!)
print("\n📋 Final Feedback Analysis (Styled):")
print("=" * 80)
display(df_display)

# Also show full DataFrame if you want
# display(df_reloaded)

# Summary
sentiment_counts = df_reloaded['Sentiment'].value_counts()
total = len(df_reloaded)
print(f"\n📊 Summary Statistics (Total: {total}):")
print("-" * 50)
for sent, count in sentiment_counts.items():
    percent = round((count / total) * 100, 1)
    emoji = emoji_map.get(sent, '')
    print(f"{sent} {emoji:<2} : {count} ({percent}%)")

# ✅ Visualization: Colored Charts
colors = {
    'POSITIVE': '#5cb85c',   # Green
    'NEGATIVE': '#d9534f',   # Red
    'NEUTRAL':  '#f0ad4e'    # Orange
}

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
bars = ax[0].bar(sentiment_counts.index, sentiment_counts.values,
                 color=[colors[s] for s in sentiment_counts.index])
ax[0].set_title("Sentiment Distribution", fontsize=14)
ax[0].set_ylabel("Count")
for bar in bars:
    height = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)

# Pie chart
ax[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
          colors=[colors[s] for s in sentiment_counts.index], startangle=90)
ax[1].set_title("Sentiment Breakdown", fontsize=14)

plt.tight_layout()
plt.show()

# 🔽 UNIVERSAL DOWNLOAD: Works in Colab, IDLE, and everywhere
print("\n" + "="*80)
print(" DOWNLOAD YOUR RESULTS ")
print("="*80)

import os
import pandas as pd
from IPython.display import HTML, display
import base64

output_file = 'sentiment_results.csv'
df_reloaded.to_csv(output_file, index=False, encoding='utf-8')

try:
    # 🌐 Try Colab download (only works in Colab)
    from google.colab import files
    print("✅ Running in Google Colab")
    print("👇 Click below to download:")
    files.download(output_file)
except ImportError:
    # 🖥 Not in Colab → show path and/or clickable link
    full_path = os.path.abspath(output_file)
    print(f"✅ Results saved to: '{output_file}'")
    print(f"📄 Full path: {full_path}")
    
    # Optional: Show clickable link in notebook environments
    try:
        # If running in Jupyter/Colab but not Colab (e.g. local Jupyter)
        csv = df_reloaded.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="{output_file}"><strong>📥 Click here to download {output_file}</strong></a>'
        display(HTML(href))
    except:
        print("💡 Tip: Open this folder to find your file")
        # Try to open folder if possible
        try:
            if os.name == 'nt':  # Windows
                os.startfile('.')
            elif os.name == 'posix':
                os.system('open .' if sys.platform == 'darwin' else 'xdg-open .')
        except:
            pass