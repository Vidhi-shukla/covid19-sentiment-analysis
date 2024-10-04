# Sentiment Analysis on COVID-19 Tweets Using Machine Learning
![OUTPUT](https://github.com/user-attachments/assets/fcf7b792-21d1-433c-9fb8-946b3f1368ce)

## Project Overview

**COVID-19 Sentiment Analysis** is a machine learning-based project designed to analyze the public’s sentiment regarding COVID-19 through Twitter data. By processing large amounts of tweets, the project classifies the sentiment expressed in each tweet as **Positive**, **Negative**, or **Neutral**. This sentiment analysis provides valuable insights into public opinion and reactions during the pandemic, aiding in tracking concerns, misinformation, and overall emotional response.

## Models Included

The project employs various machine learning and deep learning models for sentiment classification, including:
- **BERT**: A transformer-based deep learning model that captures the contextual meaning of tweets to accurately classify sentiment.
- **Support Vector Machine (SVM)**: A traditional machine learning model used to classify sentiments from preprocessed tweet data.
- **User Interface (UI)**: A Flask-based web interface where users can input tweets and receive real-time sentiment predictions.

## Motivation

Sentiment analysis during a global pandemic like COVID-19 is crucial for understanding public reactions, assessing mental health impacts, and tracking the spread of misinformation. This project provides an automated way to classify tweets, giving real-time feedback on public sentiment, with the goals to:
- Track evolving public opinion during the pandemic.
- Identify misinformation or anxiety trends.
- Assist policymakers and researchers in making data-driven decisions.

## How It Works

1. **Data Collection**: The dataset consists of tweets related to COVID-19. The data is filtered by language (English) and location (India) for targeted analysis.
2. **Preprocessing**: The tweets are cleaned to remove irrelevant elements such as URLs, mentions, hashtags, and special characters. The cleaned text is then prepared for model input.
3. **Modeling**: 
   - **BERT** is used to classify the tweets by understanding the context and sentiment.
   - Other models like SVM also perform sentiment classification to compare their effectiveness.
4. **User Interface (UI)**: A web interface allows users to input text or tweets and get real-time sentiment predictions.

## Results

The BERT model, along with other machine learning models, provides accurate sentiment classification for COVID-19 tweets. The project’s success is measured using standard evaluation metrics like accuracy, precision, recall, and F1-score.


