# Real-Time-Competitor-Strategy-Tracker-for-E-commerce

**Project Overview** 


This project is designed to deliver real-time competitive intelligence for e-commerce businesses, helping them stay ahead by providing valuable insights into competitor pricing, discount tactics, and customer sentiment. The solution utilizes:

1. Machine Learning: ARIMA-based predictive modeling for forecasting competitor pricing trends.
2. LLMs: Sentiment analysis powered by Hugging Face Transformers and Groq for understanding customer feedback.
3. Integration: Slack notifications for instant updates on competitor movements.

**Key Features:**

1. Competitor Data Tracking: Monitor competitor pricing and promotional strategies.
2. Sentiment Insights: Analyze customer reviews to extract actionable business insights.
3. Forecasting Competitor Moves: Use predictive models to anticipate competitor discounts.
4. Real-Time Alerts: Receive instant notifications via Slack on relevant competitor activity.

**Setup Instructions**

**1. Clone the Repository**

git clone <repository-url>
cd <repository-directory>

**2. Install Dependencies**
Install the required Python libraries listed in requirements.txt:

pip install -r requirements.txt

**3. Configure API Keys**
This project requires the following API keys for functionality:

1. Groq API Key: To generate strategic recommendations.
2. Slack Webhook URL: To send notifications.

Steps:

1. Obtain Groq API Key:

(i) Sign up for a Groq account at groq.com.

(ii) Get your API key from the Groq dashboard.

(iii) Add the API key to the app.py file.

2. Set Up Slack Webhook:

(i) Go to the Slack API.

(ii) Create a new app and enable Incoming Webhooks.

(iii) Add a webhook URL to a channel and copy the generated URL.

(iv) Paste this URL into the app.py file.


**4. Run the Application**
Launch the Streamlit app with the following command:

streamlit run app.py

**Project Files**
1. app.py: Main application script.
   
2. scrape.py: Script for scraping competitor data.
  
3. reviews.csv: Sample reviews data used for sentiment analysis.
  
4. competitor_data.csv: Sample data for competitor analysis.
  
5. requirements.txt: List of dependencies required for the project.

**Usage**
1. Launch the Streamlit app.
 
2. Select a product from the sidebar.
  
3. View competitor analysis, sentiment trends, and discount forecasts.
 
4. Receive strategic recommendations and real-time Slack notifications.
