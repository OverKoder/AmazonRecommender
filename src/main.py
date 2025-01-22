import requests
import json
import random

CATEGORIES = ['also_buy', 'also_view', 'asin', 'brand', 'category', 'description','feature', 'image', 'price', 'title', 'main_cat']
def main():

    print("Testing API...")
    url = "http://127.0.0.1:8000/"

    # Select random data
    random_idx = str(random.randint(0, 100))
    with open('src/data/demo_data.json', 'r') as j:
        demo_data = json.loads(j.read())

    true_category = demo_data['main_cat'][random_idx]
    demo_data = {
        'also_buy': demo_data['also_buy'][random_idx],
        'also_view': demo_data['also_view'][random_idx],
        'asin': demo_data['asin'][random_idx],
        'brand': demo_data['brand'][random_idx],
        'category': demo_data['category'][random_idx],
        'description': demo_data['description'][random_idx],
        'feature': demo_data['feature'][random_idx],
        'image': demo_data['image'][random_idx],
        'price': demo_data['price'][random_idx],
        'title': demo_data['title'][random_idx],
    }
    
    print(f"Data selected: {random_idx}")

    # Get prediction
    response = requests.post(url, json=demo_data)
    print(f"Status: {response.status_code}")

    response = json.loads(response.text)
    print(f"True category: {true_category}")
    print(f"Response, prediction: {response['Predicted category']}")

    return

if __name__ == "__main__":
    main()