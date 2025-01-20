import requests

def main():

    print("Testing API...")
    url = "http://127.0.0.1:8000/"

    data = {
        "also_buy": ["B071WSK6R8", "B006K8N5WQ", "B01ASDJLX0", "B00658TPYI"],
        "also_view": [],
        "asin": "B00N31IGPO",
        "brand": "Speed Dealer Customs",
        "category": ["Automotive", "Replacement Parts", "Shocks, Struts & Suspension", "Tie Rod Ends & Parts", "Tie Rod Ends"],
        "description": ["Universal heim joint tie rod weld in tube adapter bung. Made in the USA by Speed Dealer Customs. Tube adapter measurements are as in the title, please contact us about any questions you may have."],
        "feature": ["Completely CNC machined 1045 Steel", "Single RH Tube Adapter", "Thread: 3/4-16", "O.D.: 1-1/4", "Fits 1-1/4\" tube with .120\" wall thickness"],
        "image": [],
        "price": "",
        "title": "3/4-16 RH Weld In Threaded Heim Joint Tube Adapter Bung for 1-1/4&quot; Dia by .120 Wall Tube",
        "main_cat": "Automotive"
        }
    
    response = requests.post(url, json=data)
    print(f"Response:{response.text}")
if __name__ == "__main__":
    main()