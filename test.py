import requests

# Define the API URL
API_URL = "http://127.0.0.1:8000/chatbot/"  # Replace with the appropriate URL if hosted elsewhere

# Sample payload
payload = {
    "user_info": {
        "name": "John Doe",
        "preferences": {"category": "Data Science", "level": "Beginner"}
    },
    "query": "What are the best courses to learn Python for data analysis?",
    "chat_history": [
        {"role": "user", "message": "I'm interested in learning data analysis."},
        {"role": "assistant", "message": "What kind of tools are you familiar with?"}
    ]
}

# Send POST request
try:
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        print("Chatbot Response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
except Exception as e:
    print(f"An error occurred: {e}")
