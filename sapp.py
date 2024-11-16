import re
import string
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import PyPDF2
import streamlit as st
import io

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    return text


def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def extract_sentences_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences
    return sentences


def analyze_pdf_sentiment(file):
    sentences = extract_sentences_from_pdf(file)
    results = []
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for sentence in sentences:
        cleaned_sentence = preprocess_text(sentence)
        sentiment = get_vader_sentiment(cleaned_sentence)
        results.append({'sentence': sentence, 'sentiment': sentiment})
        sentiment_counts[sentiment] += 1  # Count sentiment for performance metrics

    # Calculate performance metrics
    total = sum(sentiment_counts.values())
    positive_percentage = (sentiment_counts['positive'] / total) * 100
    negative_percentage = (sentiment_counts['negative'] / total) * 100
    neutral_percentage = (sentiment_counts['neutral'] / total) * 100

    # Append performance metrics to results
    results.append({
        'sentence': 'Performance Metrics',
        'sentiment': f"Positive: {positive_percentage:.2f}%, Negative: {negative_percentage:.2f}%, Neutral: {neutral_percentage:.2f}%"
    })

    # Convert results to DataFrame
    return pd.DataFrame(results)


# Streamlit app
st.title("SentimentScopes")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file for sentiment analysis", type="pdf")

if uploaded_file:
    # Perform sentiment analysis and display results
    with st.spinner("Analyzing..."):
        sentiment_results = analyze_pdf_sentiment(uploaded_file)

        # Display results
        st.write("### Sentiment Analysis Results")
        st.dataframe(sentiment_results)

        # Save results to an in-memory Excel file
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
            sentiment_results.to_excel(writer, index=False)

        # Download button for the Excel file
        st.download_button(
            label="Download sentiment results as Excel",
            data=output_buffer.getvalue(),
            file_name="sentiment_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )