import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from random import uniform
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_title(soup):
    try:
        title = soup.find("span", attrs={"id": "productTitle"}) or soup.find("h1", attrs={"id": "title"})
        return title.text.strip() if title else ""
    except Exception as e:
        logger.error(f"Error extracting title: {e}")
        return ""

def get_selling_price(soup):
    try:
        price = soup.find("span", attrs={'class': 'a-price-whole'}) or soup.find("span", attrs={'class': 'a-price'})
        if price:
            return price.text.strip().replace(".", "").replace(",", "").replace("₹", "")
        return ""
    except Exception as e:
        logger.error(f"Error extracting selling price: {e}")
        return ""

def get_MRP(soup):
    try:
        selectors = [{'class': 'a-size-small aok-offscreen'}, {'class': 'a-text-price'}]
        for selector in selectors:
            price = soup.find("span", attrs=selector)
            if price:
                return price.text.strip().replace("M.R.P.: ₹", "").replace("₹", "").replace(",", "")
        return ""
    except Exception as e:
        logger.error(f"Error extracting MRP: {e}")
        return ""

def get_discount(soup):
    try:
        discount = soup.find("span", attrs={'class': 'savingsPercentage'})
        return discount.text.strip().replace('-', '').replace('%', '') if discount else ""
    except Exception as e:
        logger.error(f"Error extracting discount: {e}")
        return ""

def get_rating(soup):
    try:
        rating_selectors = [{'class': 'a-icon a-icon-star'}, {'class': 'a-icon-alt'}]
        for selector in rating_selectors:
            rating = soup.find("i", attrs=selector) or soup.find("span", attrs=selector)
            if rating:
                return rating.text.strip().split()[0]
        return ""
    except Exception as e:
        logger.error(f"Error extracting rating: {e}")
        return ""

def get_review_texts(soup, max_reviews=5):
    try:
        reviews = []
        # Try finding reviews in the main product page
        review_elements = soup.find_all("div", attrs={'data-hook': 'review-collapsed'})
        if not review_elements:
            review_elements = soup.find_all("div", attrs={'class': 'a-expander-content reviewText'})
        
        # Get up to max_reviews reviews
        for review in review_elements[:max_reviews]:
            review_text = review.text.strip()
            if review_text:
                reviews.append(review_text)
        
        # If no reviews found, return an empty list
        return reviews if reviews else []
    
    except Exception as e:
        logger.error(f"Error extracting review texts: {e}")
        return []

def scrape_amazon_products(search_url, max_products=50):
    base_url = "https://www.amazon.in"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

    try:
        session = requests.Session()
        response = session.get(search_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a", attrs={'class': 'a-link-normal s-no-outline'})

        if not links:
            logger.warning("No product links found on the search page")
            return None, None

        products_data = []
        reviews_data = []

        for link in links[:max_products]:
            product_url = link.get('href')
            if not product_url:
                continue
            if not product_url.startswith('http'):
                product_url = base_url + product_url

            try:
                logger.info(f"Scraping product: {product_url}")
                time.sleep(uniform(1, 3))
                response = session.get(product_url, headers=headers)
                response.raise_for_status()

                product_soup = BeautifulSoup(response.content, "html.parser")
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                title = get_title(product_soup)

                # Get product data
                product_data = {
                    "title": title,
                    "selling_price": get_selling_price(product_soup),
                    "MRP": get_MRP(product_soup),
                    "discount": get_discount(product_soup),
                    "rating": get_rating(product_soup),
                    "availability": "Available",
                    "url": product_url,
                    "scrape_datetime": current_datetime
                }
                products_data.append(product_data)

                # Get reviews data
                review_texts = get_review_texts(product_soup)
                for i, review_text in enumerate(review_texts, 1):
                    review_data = {
                        "title": title,
                        "review_number": i,
                        "review_count": len(review_texts),
                        "review_text": review_text,
                        "scrape_datetime": current_datetime
                    }
                    reviews_data.append(review_data)

            except Exception as e:
                logger.error(f"Error scraping product {product_url}: {e}")
                continue

        # Create DataFrames
        products_df = pd.DataFrame(products_data)
        reviews_df = pd.DataFrame(reviews_data)

        # Clean data
        products_df['title'] = products_df['title'].replace('', pd.NA)
        products_df = products_df.dropna(subset=['title'])

        return products_df, reviews_df

    except Exception as e:
        logger.error(f"Error in main scraping process: {e}")
        return None, None

if __name__ == '__main__':
    search_url = "https://www.amazon.in/s?k=earphones&crid=23H19CC51YB96&sprefix-earphone%2Caps%2C228&ref=nb_sb_noss_2"
    products_df, reviews_df = scrape_amazon_products(search_url)

    if products_df is not None and not products_df.empty:
        # Save product data
        products_df.to_csv("amazon_scraped_data.csv", header=True, index=False)
        logger.info(f"Successfully scraped {len(products_df)} products")

        # Save reviews data
        if not reviews_df.empty:
            reviews_df.to_csv("reviews.csv", index=False)
            logger.info(f"Successfully saved {len(reviews_df)} reviews")
        else:
            logger.warning("No reviews were found")
    else:
        logger.error("No data was scraped")
