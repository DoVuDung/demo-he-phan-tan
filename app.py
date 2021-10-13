from flask import Flask, render_template, jsonify, request
import torch
from torch import nn
from chatbot.model import BertForClassification, roberta
from chatbot.chat import tokenizer, ans_dict, get_response

app = Flask(__name__)
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot', methods=["POST"])
def chatbot_msg():
    model = BertForClassification(roberta) 
    model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
    if request.method == "POST":
        user_data = request.json
        sentence = user_data['msg']
        output = get_response(sentence, ans_dict, model, tokenizer)
        return jsonify(msg=output)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)        



