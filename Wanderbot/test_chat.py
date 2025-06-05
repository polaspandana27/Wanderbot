from chatbot import get_response

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("WanderBot:", get_response(user_input))
