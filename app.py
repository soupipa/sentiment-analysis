import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="TextMining - Feedback Sentiment Analysis",
    page_icon="🧹",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .positive-metric {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .negative-metric {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
    }
    .neutral-metric {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sentiment_pipeline' not in st.session_state:
    st.session_state.sentiment_pipeline = None
if 'df_results' not in st.session_state:
    st.session_state.df_results = None

@st.cache_resource
def load_light_sentiment_model():
    """Load a lighter sentiment analysis model"""
    try:
        # Use a smaller, faster model that downloads quickly
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            truncation=True,
            max_length=512
        )
    except Exception as e:
        st.error(f"Error loading light model: {e}")
        return None

@st.cache_resource
def load_offline_sentiment():
    """Fallback: Rule-based sentiment without downloading models"""
    return "rule-based"

@st.cache_resource
def load_sentiment_model():
    """Load the multilingual sentiment analysis model with timeout handling"""
    import os
    
    # Set longer timeout for model download
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
    
    try:
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
        
        return pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=512
        )
    except Exception as e:
        st.error(f"Error loading full model: {e}")
        st.info("💡 Try using the Light Model or Rule-based option")
        return None

def is_valid_feedback(text):
    """Check if text is valid feedback (not code, HTML, etc.)"""
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 3:
        return False
    if 'http' in text or 'www' in text:
        return False
    if any(kw in text.lower() for kw in [
        'function', 'var ', 'document.', 'window.', '</script>', '<script',
        'copyright', 'license', 'spdx', 'nonce', 'bootstrap', 'docs-chrome',
        '.com', '.js', '.css', '://', 'class="', 'id="', 'href='
    ]):
        return False
    if re.search(r'[{}()\[\]<>;]', text):
        return False
    if text.lower().islower() == False and text.isupper():
        return False
    return True

def get_sentiment(text, sentiment_pipeline):
    """Enhanced sentiment analysis with multilingual rules"""
    if not text or pd.isna(text):
        return 'neutral', 0.0
    
    text_clean = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text_no_space = text_clean.replace(' ', '')
    
    # Label mapping
    label_map = {
        '0': 'negative', '1': 'neutral', '2': 'positive',
        'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
        'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive'
    }
    
    # English positive keywords
    english_positive = [
        'excellent', 'love', 'amazing', 'great', 'best', 'perfect',
        'happy', 'pleased', 'recommended', 'satisfied', 'fantastic',
        'wonderful', 'awesome', 'outstanding', 'brilliant', 'superb'
    ]
    if any(w in text_clean for w in english_positive):
        return 'positive', 0.97
    
    # English negative keywords
    english_negative = [
        'terrible', 'awful', 'hate', 'worst', 'angry', 'frustrated',
        'broken', 'crashing', 'late', 'waste', 'scam', 'disappointed',
        'horrible', 'disgusting', 'pathetic', 'useless', 'garbage'
    ]
    if any(w in text_clean for w in english_negative):
        return 'negative', 0.96
    
    # English neutral keywords
    if any(w in text_clean for w in ['okay', 'fine', 'acceptable', 'nothing special', 'average']):
        return 'neutral', 0.90
    
    # Tunisian/Arabic positive phrases (normalized)
    tunisian_pos_phrases = [
        'المنتوجممتاز', 'جودةعالية', 'مريحوسهلالاستخدام', 'تصميمرائع',
        'تجربةممتازة', 'منتوجمبدع', 'أداءقوي', 'عجبنيبرشة', 'منتوجموثوق',
        'almantoujmumtaz', 'jowdataliya', 'marihwsahilalistikhdam',
        'tasmimrai', 'tajribamumtaza', 'ajabnibarsha'
    ]
    
    # Tunisian/Arabic negative phrases (normalized)
    tunisian_neg_phrases = [
        'المنتوجمخيبللأمل', 'جودةمتوسطة', 'صعيبفيالاستخدام', 'التصميممشعاجبني',
        'تجربةمشمرضية', 'أداءمخيب', 'ماعجبنيش', 'منتوجغيرموثوق',
        'almantoujmkhayyiblilamal', 'jowdatmutawasita', 'majebnish',
        'meshajebni', 'khabech', 'mahoosh'
    ]
    
    # French positive phrases (normalized)
    french_pos_phrases = [
        'leproduitestexcellent', 'qualitesuperieure', 'confortableetfacileautiliser',
        'designmagnifique', 'uneexperienceexceptionnelle', 'produitinnovant',
        'excellent', 'magnifique', 'parfait', 'superbe', 'formidable'
    ]
    
    # French negative phrases (normalized)  
    french_neg_phrases = [
        'leproduitestdecevant', 'qualitemediocre', 'difficileautiliser',
        'ledesignlaisseadesire', 'experienceinsatisfaisante', 'produitpeuinnovant',
        'decevant', 'mediocre', 'mauvais', 'nul', 'terrible'
    ]
    
    # Check for positive phrases
    if (any(phrase in text_no_space for phrase in tunisian_pos_phrases) or
        any(phrase in text_no_space for phrase in french_pos_phrases)):
        return 'positive', 0.99
    
    # Check for negative phrases
    if (any(phrase in text_no_space for phrase in tunisian_neg_phrases) or
        any(phrase in text_no_space for phrase in french_neg_phrases)):
        return 'negative', 0.98
    
    # Enhanced rule-based analysis for mixed languages
    positive_indicators = ['good', 'nice', 'cool', 'top', 'bien', 'bon', 'جميل', 'ممتاز']
    negative_indicators = ['bad', 'poor', 'fail', 'problem', 'issue', 'mauvais', 'pas', 'سيء', 'مشكلة']
    
    pos_score = sum(1 for word in positive_indicators if word in text_clean)
    neg_score = sum(1 for word in negative_indicators if word in text_clean)
    
    if pos_score > neg_score and pos_score > 0:
        return 'positive', 0.80 + (pos_score * 0.05)
    elif neg_score > pos_score and neg_score > 0:
        return 'negative', 0.80 + (neg_score * 0.05)
    
    # Fallback to model if available
    if sentiment_pipeline and sentiment_pipeline != "rule-based":
        try:
            result = sentiment_pipeline(str(text))[0]
            key = str(result['label']).strip().upper()
            sentiment = label_map.get(key, 'neutral')
            score = result['score']
            
            # Reduce overuse of neutral
            if sentiment == 'neutral' and score < 0.75:
                return 'negative', score * 0.8
            return sentiment, score
        except Exception:
            pass
    
    # Final fallback: simple length and punctuation analysis
    if len(text_clean) < 10:
        return 'neutral', 0.60
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    if exclamation_count > 1:
        return 'positive', 0.70
    elif question_count > 1:
        return 'neutral', 0.65
    
    return 'neutral', 0.60

def create_visualizations(df):
    """Create interactive visualizations"""
    sentiment_counts = df['Sentiment'].value_counts()
    colors = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#ffc107'}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map=colors,
            title="Sentiment Distribution",
            labels={'x': 'Sentiment', 'y': 'Count'}
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map=colors,
            title="Sentiment Breakdown"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

def main():
    # Header
    st.markdown('<div class="main-header">🧹 TextMining - Feedback Sentiment Analysis</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Multilingual sentiment analysis for customer feedback**  
    Supports: English, Arabic, French, and Tunisian dialect
    """)
    
    # Sidebar
    st.sidebar.title("📊 Project Settings")
    st.sidebar.markdown("---")
    
    # Model loading
    with st.sidebar:
        model_option = st.radio(
            "🤖 Choose Analysis Method:",
            options=[
                "Rule-based (No Download Required)",
                "Light Model (Small Download)",
                "Full Model (Best Accuracy)"
            ],
            index=0,
            help="Rule-based works offline, Light Model ~130MB, Full Model ~500MB"
        )
        
        if st.button("🚀 Load Sentiment Model", type="primary"):
            with st.spinner("Loading sentiment analysis..."):
                if model_option == "Rule-based (No Download Required)":
                    st.session_state.sentiment_pipeline = load_offline_sentiment()
                    st.success("✅ Rule-based analyzer loaded!")
                elif model_option == "Light Model (Small Download)":
                    st.session_state.sentiment_pipeline = load_light_sentiment_model()
                    if st.session_state.sentiment_pipeline:
                        st.success("✅ Light model loaded successfully!")
                    else:
                        st.error("❌ Light model failed, falling back to rule-based")
                        st.session_state.sentiment_pipeline = load_offline_sentiment()
                else:
                    st.session_state.sentiment_pipeline = load_sentiment_model()
                    if st.session_state.sentiment_pipeline:
                        st.success("✅ Full model loaded successfully!")
                    else:
                        st.error("❌ Full model failed, falling back to rule-based")
                        st.session_state.sentiment_pipeline = load_offline_sentiment()
    
    # File upload
    st.sidebar.markdown("### 📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your feedback data in CSV format"
    )
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            df_raw = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_raw)} rows from {uploaded_file.name}")
            
            # Show raw data preview
            with st.expander("👀 Preview Raw Data", expanded=False):
                st.dataframe(df_raw.head())
            
            # Column selection
            st.markdown("### 🎯 Select Feedback Column")
            feedback_columns = [col for col in df_raw.columns 
                              if any(keyword in col.lower() for keyword in ['feedback', 'comment', 'text', 'review'])]
            
            if not feedback_columns:
                feedback_columns = list(df_raw.columns)
            
            selected_column = st.selectbox(
                "Choose the column containing feedback text:",
                options=feedback_columns,
                index=0
            )
            
            if st.button("🧠 Analyze Sentiment", type="primary"):
                if st.session_state.sentiment_pipeline is None:
                    st.error("❌ Please load the sentiment model first!")
                    return
                
                # Clean data
                with st.spinner("🧹 Cleaning data..."):
                    df = df_raw[[selected_column]].copy()
                    df.rename(columns={selected_column: 'Comment'}, inplace=True)
                    df['Comment'] = df['Comment'].astype(str)
                    
                    # Filter valid feedback
                    df['is_valid'] = df['Comment'].apply(is_valid_feedback)
                    initial_count = len(df)
                    df = df[df['is_valid']].reset_index(drop=True)
                    df.drop(columns=['is_valid'], inplace=True)
                    
                    # Clean text
                    df['Comment'] = df['Comment'].str.strip().replace(r'\s+', ' ', regex=True)
                    
                    st.info(f"📊 Cleaned data: {len(df)} valid comments (removed {initial_count - len(df)} invalid entries)")
                
                if len(df) == 0:
                    st.error("❌ No valid feedback found after cleaning.")
                    return
                
                # Analyze sentiment
                with st.spinner("🧠 Analyzing sentiment..."):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, comment in enumerate(df['Comment']):
                        sentiment, confidence = get_sentiment(comment, st.session_state.sentiment_pipeline)
                        results.append((sentiment, confidence))
                        progress_bar.progress((i + 1) / len(df))
                    
                    df['Sentiment'], df['Confidence'] = zip(*results)
                    
                    # Add emoji
                    emoji_map = {'positive': '😊', 'negative': '😡', 'neutral': '😐', 'unknown': '❓'}
                    df['Emoji'] = df['Sentiment'].map(emoji_map)
                    
                    # Add preview
                    df['Preview'] = df['Comment'].str.slice(0, 80) + df['Comment'].str.len().apply(lambda x: '...' if x > 80 else '')
                    
                    st.session_state.df_results = df
                    progress_bar.empty()
                
                st.success("✅ Analysis complete!")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return
    
    # Display results if available
    if st.session_state.df_results is not None:
        df = st.session_state.df_results
        
        # Metrics
        st.markdown("### 📊 Summary Metrics")
        sentiment_counts = df['Sentiment'].value_counts()
        total = len(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Total Feedback</h3>
                <h2>{total}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pos_count = sentiment_counts.get('positive', 0)
            pos_pct = round((pos_count / total) * 100, 1) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card positive-metric">
                <h3>😊 Positive</h3>
                <h2>{pos_count} ({pos_pct}%)</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            neg_count = sentiment_counts.get('negative', 0)
            neg_pct = round((neg_count / total) * 100, 1) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card negative-metric">
                <h3>😡 Negative</h3>
                <h2>{neg_count} ({neg_pct}%)</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            neu_count = sentiment_counts.get('neutral', 0)
            neu_pct = round((neu_count / total) * 100, 1) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card neutral-metric">
                <h3>😐 Neutral</h3>
                <h2>{neu_count} ({neu_pct}%)</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        create_visualizations(df)
        
        # Detailed results
        st.markdown("### 📋 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by sentiment:",
                options=['All'] + list(sentiment_counts.index),
                index=0
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Minimum confidence:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        # Apply filters
        df_filtered = df.copy()
        if sentiment_filter != 'All':
            df_filtered = df_filtered[df_filtered['Sentiment'] == sentiment_filter]
        df_filtered = df_filtered[df_filtered['Confidence'] >= confidence_threshold]
        
        # Display filtered results
        st.dataframe(
            df_filtered[['Emoji', 'Preview', 'Sentiment', 'Confidence']],
            use_container_width=True,
            hide_index=True
        )
        
        # Download results
        st.markdown("### 💾 Download Results")
        csv = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="feedback_sentiment_analysis.csv",
            mime="text/csv"
        )
        
        # Sample comments by sentiment
        st.markdown("### 🔍 Sample Comments by Sentiment")
        
        tab1, tab2, tab3 = st.tabs(["😊 Positive", "😡 Negative", "😐 Neutral"])
        
        with tab1:
            pos_samples = df[df['Sentiment'] == 'positive'].head(5)
            if not pos_samples.empty:
                for _, row in pos_samples.iterrows():
                    st.info(f"**Confidence: {row['Confidence']:.2f}**\n\n{row['Comment']}")
            else:
                st.write("No positive comments found.")
        
        with tab2:
            neg_samples = df[df['Sentiment'] == 'negative'].head(5)
            if not neg_samples.empty:
                for _, row in neg_samples.iterrows():
                    st.error(f"**Confidence: {row['Confidence']:.2f}**\n\n{row['Comment']}")
            else:
                st.write("No negative comments found.")
        
        with tab3:
            neu_samples = df[df['Sentiment'] == 'neutral'].head(5)
            if not neu_samples.empty:
                for _, row in neu_samples.iterrows():
                    st.warning(f"**Confidence: {row['Confidence']:.2f}**\n\n{row['Comment']}")
            else:
                st.write("No neutral comments found.")
    
    else:
        # Welcome message
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Choose analysis method** in the sidebar:
           - **Rule-based**: Works offline, uses keyword matching
           - **Light Model**: Small download (~130MB), good accuracy
           - **Full Model**: Large download (~500MB), best accuracy
        2. **Load the analyzer** using the button in the sidebar
        3. **Upload your CSV file** containing feedback data
        4. **Select the feedback column** from your data
        5. **Click "Analyze Sentiment"** to process your feedback
        
        ### 🌍 Supported Languages
        - **English**: Standard sentiment keywords
        - **Arabic**: Classical and Tunisian dialect
        - **French**: Standard French expressions
        - **Tunisian**: Mixed Arabic-French expressions
        
        ### 📝 Sample Data Format
        Your CSV should have at least one column containing text feedback:
        
        | feedback | rating |
        |----------|--------|
        | This product is excellent! | 5 |
        | المنتوج ممتاز | 5 |
        | Le produit est magnifique | 4 |
        
        ### 💡 Tips
        - **No Internet?** Use Rule-based analysis - works completely offline
        - **Slow Connection?** Try Light Model first
        - **Best Accuracy?** Use Full Model for multilingual content
        """)

if __name__ == "__main__":
    main()