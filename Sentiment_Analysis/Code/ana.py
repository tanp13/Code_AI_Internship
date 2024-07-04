import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plt
import seaborn as sns

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    .stTextInput, .stTextArea {
        background-color: #f7f7f7;
        border: 1px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        color: #333333;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        font-size: 16px;
        border-radius: 5px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        color: #333333;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset
df = pd.read_csv("Reviews.csv")

# Preprocess the data
review_text = df["Text"]
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment and subjectivity
sentiment_scores = []
blob_subj = []
for review in review_text:
    sentiment_scores.append(analyzer.polarity_scores(review)["compound"])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

# Classify sentiment
sentiment_classes = []
for sentiment_score in sentiment_scores:
    if sentiment_score > 0.8:
        sentiment_classes.append("Highly positive")    
    elif sentiment_score > 0.4:
        sentiment_classes.append("Positive")
    elif -0.4 <= sentiment_score <= 0.4:
        sentiment_classes.append("Neutral")
    elif sentiment_score < -0.4:
        sentiment_classes.append("Negative")
    else:
        sentiment_classes.append("Highly negative")

# Streamlit app layout
st.title("Sentiment Analysis On Customer Feedback 🎉")

# User input section
st.header("Enter Your Feedback")
user_input = st.text_area("Enter the Feedback:")
blob = TextBlob(user_input)

user_sentiment_score = analyzer.polarity_scores(user_input)['compound']
if user_sentiment_score > 0.8:
    user_sentiment_class = "Highly positive 😊"
elif user_sentiment_score > 0.4:
    user_sentiment_class = "Positive 🙂"
elif -0.4 <= user_sentiment_score <= 0.4:
    user_sentiment_class = "Neutral 😐"
elif user_sentiment_score < -0.4:
    user_sentiment_class = "Negative ☹️"
else:
    user_sentiment_class = "Highly negative 😠"

st.write(f"**VADER Sentiment Class:** {user_sentiment_class}")
st.write(f"**VADER Sentiment Score:** {user_sentiment_score}")
st.write(f"**TextBlob Polarity:** {blob.sentiment.polarity}")
st.write(f"**TextBlob Subjectivity:** {blob.sentiment.subjectivity}")

# Clean text section
st.header("Clean Your Text")
pre = st.text_input("Enter text to clean:")
if pre:
    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))
else:
    st.write("No text provided for cleaning.")

# Graphical Representation of Data
st.header("Graphical Representation of Data")
plt.figure(figsize=(10, 6))
sns.histplot(data=sentiment_scores, kde=True, bins=30, color="blue")
plt.xlabel("Sentiment score")
plt.ylabel("Count")
plt.title("Sentiment Score Distribution")
st.pyplot(plt)

# DataFrames with Sentiment Analysis results
df["Sentiment Class"] = sentiment_classes
df["Sentiment Score"] = sentiment_scores
df["Subjectivity"] = blob_subj

new_df = df[["Score", "Text", "Sentiment Score", "Sentiment Class", "Subjectivity"]]
st.header("Input Dataframe")
st.dataframe(new_df.head(30), use_container_width=True)
