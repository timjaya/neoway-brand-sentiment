import pandas as pd
import numpy as np

# Import Yelp reviews
reviews = pd.read_json('review.json', lines=True)

# Import Yelp description of businesses
business = pd.read_json('business.json', lines=True)

# Filter out businesses that belong to a category containing 'Restaurants'
restaurants = business[business['categories'].str.contains("Restaurants", na=False)]

# Filter out reviews that belong to likely restaurants 
restaurant_reviews = reviews[reviews.business_id.isin(restaurants.business_id)]

# Save results as JSON
restaurants.to_json('restaurant_reviews.json', lines=True, orient='records')