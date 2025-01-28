import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from transformers import pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

API_KEY = "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" 
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

def truncate_text(text, max_length=512):
    return text[:max_length]

def load_amazon_data():
    """Load main Amazon scraped data."""
    try:
        data = pd.read_csv("amazon_scraped_data.csv")
        # Convert price columns to numeric, removing any currency symbols and commas
        data['selling_price'] = pd.to_numeric(data['selling_price'].str.replace(',', ''), errors='coerce')
        data['MRP'] = pd.to_numeric(data['MRP'].str.replace(',', ''), errors='coerce')
        # Convert discount to numeric, removing % symbol
        data['discount'] = pd.to_numeric(data['discount'].str.replace('%', ''), errors='coerce')
        # Convert scrape_datetime to datetime
        data['scrape_datetime'] = pd.to_datetime(data['scrape_datetime'])
        return data
    except Exception as e:
        st.error(f"Error loading Amazon data: {e}")
        return pd.DataFrame()

def load_reviews_data():
    """Load reviews data from the reviews.csv file."""
    try:
        reviews = pd.read_csv("reviews.csv")
        reviews['scrape_datetime'] = pd.to_datetime(reviews['scrape_datetime'])
        return reviews
    except Exception as e:
        st.error(f"Error loading reviews data: {e}")
        return pd.DataFrame()

def analyze_sentiment(reviews):
    """Analyze customer sentiment for reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    truncated_reviews = [truncate_text(str(review)) for review in reviews]
    return sentiment_pipeline(truncated_reviews)

def train_predictive_model(data):
    """Train a predictive model for pricing strategy."""
    try:
        X = data[['selling_price', 'discount']]
        y = data['MRP']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.error(f"Error training predictive model: {e}")
        return None

def forecast_discounts_arima(data, future_days=5):
    """Forecast future discounts using ARIMA."""
    try:
        # Prepare data for ARIMA
        data = data.sort_values('scrape_datetime')
        data.set_index('scrape_datetime', inplace=True)
        
        # Ensure we have numeric discount values
        discount_series = pd.to_numeric(data['discount'], errors='coerce')
        discount_series = discount_series.dropna()
        
        if len(discount_series) < 2:
            raise ValueError("Not enough data points for forecasting")
            
        # Fit ARIMA model
        model = ARIMA(discount_series, order=(5,1,0))
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=future_days)
        future_dates = pd.date_range(
            start=discount_series.index[-1] + pd.Timedelta(days=1),
            periods=future_days
        )
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Discount': forecast
        })
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df
    except Exception as e:
        st.error(f"Error in discount forecasting: {e}")
        return pd.DataFrame()

def send_to_slack(data):
    """Send data to Slack webhook."""
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
    """Generate strategic recommendations using Groq LLM."""
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
            "model": "llama3-8b-8192",
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

# Load data
amazon_data = load_amazon_data()
reviews_data = load_reviews_data()

if amazon_data.empty:
    st.error("No Amazon data available. Please run the scraper first.")
else:
    # Product selection
    st.sidebar.header("Select a Product")
    products = amazon_data['title'].unique().tolist()
    selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)

    # Filter data for selected product
    product_data = amazon_data[amazon_data['title'] == selected_product]
    product_reviews = reviews_data[reviews_data['title'] == selected_product]

    # Display product information
    st.header(f"Analysis for {selected_product}")
    
    # Product details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"₹{product_data['selling_price'].iloc[-1]:,.0f}")
    with col2:
        st.metric("MRP", f"₹{product_data['MRP'].iloc[-1]:,.0f}")
    with col3:
        st.metric("Discount", f"{product_data['discount'].iloc[-1]}%")

    # Price history
    st.subheader("Price History")
    fig_price = px.line(product_data, x='scrape_datetime', y=['selling_price', 'MRP'],
                       title="Price History Over Time")
    st.plotly_chart(fig_price)

    # Sentiment analysis
    if not product_reviews.empty:
        st.subheader("Customer Sentiment Analysis")
        sentiments = analyze_sentiment(product_reviews['review_text'].tolist())
        sentiment_df = pd.DataFrame(sentiments)
        fig_sentiment = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
        st.plotly_chart(fig_sentiment)

        # Display recent reviews
        st.subheader("Recent Reviews")
        st.dataframe(product_reviews[['scrape_datetime', 'review_text']].tail())
    else:
        st.info("No reviews available for this product.")

    # Discount forecasting
    if len(product_data) >= 2:
        st.subheader("Discount Forecast (Next 5 Days)")
        forecast_df = forecast_discounts_arima(product_data)
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
