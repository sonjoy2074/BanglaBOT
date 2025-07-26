from flask import Flask, request, jsonify
from chatbot import build_chain

app = Flask(__name__)

# Load chain and initialize chat history
chat_chain = build_chain()
chat_history = []

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" in request body'}), 400

    question = data['question']
    response = chat_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    # Append to history
    chat_history.append((question, response["answer"]))

    return jsonify({
        "answer": response["answer"],
        "history": chat_history
    })

if __name__ == '__main__':
    app.run(debug=True)
