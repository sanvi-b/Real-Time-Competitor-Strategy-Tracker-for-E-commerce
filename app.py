import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from transformers import pipeline

API_KEY = ""  # Groq API Key
SLACK_WEBHOOK = ""  # Slack webhook url

def truncate_text(text, max_length=512):
    return text[:max_length] if text else ""

def load_amazon_data():
    try:
        data = pd.read_csv("amazon_scraped_data.csv")
        data['selling_price'] = pd.to_numeric(data['selling_price'].astype(str).str.replace(',', ''), errors='coerce')
        data['MRP'] = pd.to_numeric(data['MRP'].astype(str).str.replace(',', ''), errors='coerce')
        data['discount'] = pd.to_numeric(data['discount'].astype(str).str.replace('%', ''), errors='coerce')
        data['scrape_datetime'] = pd.to_datetime(data['scrape_datetime'])
        return data
    except Exception as e:
        st.error(f"Error loading Amazon data: {e}")
        return pd.DataFrame()

def load_reviews_data():
    try:
        reviews = pd.read_csv("reviews.csv")
        reviews['scrape_datetime'] = pd.to_datetime(reviews['scrape_datetime'])
        return reviews
    except Exception as e:
        st.error(f"Error loading reviews data: {e}")
        return pd.DataFrame()

def analyze_sentiment(reviews):
    sentiment_pipeline = pipeline("sentiment-analysis")
    truncated_reviews = [truncate_text(str(review)) for review in reviews]
    return sentiment_pipeline(truncated_reviews)

def simple_price_prediction(data, days=5):
    if len(data) < 2:
        return pd.DataFrame()

    data = data.sort_values('scrape_datetime')
    daily_changes = data['discount'].diff().mean()

    last_date = data['scrape_datetime'].max()
    last_discount = data['discount'].iloc[-1]

    future_dates = [last_date + timedelta(days=x) for x in range(1, days + 1)]
    predicted_discounts = [last_discount + (daily_changes * x) for x in range(1, days + 1)]
    predicted_discounts = [min(max(0, d), 100) for d in predicted_discounts]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Discount': predicted_discounts
    })
    forecast_df.set_index('Date', inplace=True)

    return forecast_df

def send_to_slack(data):
    try:
        payload = {"text": data}
        response = requests.post(
            SLACK_WEBHOOK,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error sending to Slack: {e}")
        return False

def generate_strategy_recommendation(product_name, product_data, sentiment):
    date = datetime.now()
    prompt = f"""
    You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

    1. **Product Name**: {product_name}

    2. **Product Data**:
    Current Price: ₹{product_data['selling_price'].iloc[-1] if not product_data.empty else 'N/A'}
    Original Price (MRP): ₹{product_data['MRP'].iloc[-1] if not product_data.empty else 'N/A'}
    Current Discount: {product_data['discount'].iloc[-1] if not product_data.empty else 'N/A'}%
    Availability: {product_data['availability'].iloc[-1] if not product_data.empty else 'N/A'}

    3. **Sentiment Analysis**:
    {sentiment}

    4. **Today's Date**: {str(date)}

    Provide your recommendations in a structured format:
    1. **Pricing Strategy**
    2. **Promotional Campaign Ideas**
    3. **Customer Satisfaction Recommendations**
    """

    try:
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "llama3-8b-8192",  # Or another suitable model
            "temperature": 0,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(data),
            headers=headers
        )
        res = res.json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return "Error generating recommendations"


# Streamlit UI
st.set_page_config(page_title="Amazon Product Analysis Dashboard", layout="wide")
st.title("Amazon Product Analysis Dashboard")

amazon_data = load_amazon_data()
reviews_data = load_reviews_data()

if amazon_data.empty:
    st.error("No Amazon data available. Please run the scraper first.")
else:
    st.sidebar.header("Select a Product")
    products = amazon_data['title'].unique().tolist()
    selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

    product_data = amazon_data[amazon_data['title'] == selected_product].copy()
    product_reviews = reviews_data[reviews_data['title'] == selected_product].copy()

    st.header(f"Analysis for {selected_product}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"₹{product_data['selling_price'].iloc[-1]:,.0f}")
    with col2:
        st.metric("MRP", f"₹{product_data['MRP'].iloc[-1]:,.0f}")
    with col3:
        st.metric("Discount", f"{product_data['discount'].iloc[-1]}%")

    if not product_reviews.empty:
        st.subheader("Review Statistics")
        total_reviews = len(product_reviews)
        review_count = product_reviews['review_count'].iloc[0]
        st.metric("Total Reviews Analyzed", f"{total_reviews} out of {review_count}")

    st.subheader("Price History")

    if not product_data.empty:
        product_data = product_data.sort_values('scrape_datetime')

        time_diff = product_data['scrape_datetime'].max() - product_data['scrape_datetime'].min()

        if time_diff <= timedelta(days=1):
            resample_interval = '1H'
            tick_format = "%H:%M"
        elif time_diff <= timedelta(days=7):
            resample_interval = '12H'
            tick_format = "%Y-%m-%d %H:%M"
        elif time_diff <= timedelta(days=30):
            resample_interval = '1D'
            tick_format = "%Y-%m-%d"
        else:
            resample_interval = '7D'
            tick_format = "%Y-%m-%d"

        product_data = product_data.set_index('scrape_datetime')

        numeric_cols = product_data.select_dtypes(include=['number']).columns
        product_data_resampled = product_data[numeric_cols].resample(resample_interval).mean()
        product_data_resampled = product_data_resampled.reset_index()

        fig_price = px.line(product_data_resampled, x='scrape_datetime', y=['selling_price', 'MRP'], 
                            title=f"Price History ({resample_interval} Intervals)")
        fig_price.update_xaxes(tickformat=tick_format)
        st.plotly_chart(fig_price)

    else:
        st.warning("No price data available for this product.")
    # Sentiment analysis
    if not product_reviews.empty:
        st.subheader("Customer Sentiment Analysis")
        sentiments = analyze_sentiment(product_reviews['review_text'].tolist())
        sentiment_df = pd.DataFrame(sentiments)
        fig_sentiment = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
        st.plotly_chart(fig_sentiment)

        # Display recent reviews
        st.subheader("Recent Reviews")
        st.dataframe(product_reviews[['review_number', 'scrape_datetime', 'review_text']].sort_values('review_number'))
    else:
        st.info("No reviews available for this product.")

    # Simple price prediction
    if len(product_data) >= 2:
        st.subheader("Discount Forecast (Next 5 Days)")
        forecast_df = simple_price_prediction(product_data)
        if not forecast_df.empty:
            fig_forecast = px.line(forecast_df, title="Discount Forecast")
            st.plotly_chart(fig_forecast)

    # Strategic recommendations
    st.subheader("Strategic Recommendations")
    recommendations = generate_strategy_recommendation(
        selected_product,
        product_data,
        sentiments if not product_reviews.empty else "No reviews available"
    )
    st.write(recommendations)

    # Send to Slack if webhook is configured
    if SLACK_WEBHOOK:
        if st.button("Send Recommendations to Slack"):
            if send_to_slack(recommendations):
                st.success("Recommendations sent to Slack!")
            else:
                st.error("Failed to send recommendations to Slack")
