import json
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load train and validation dataset
with open('val_content.json', 'r', encoding="utf-8") as c:
    contents = json.load(c)
tags = []
X = []
y = []
for content in contents['intents']:
    tag = content['tag']
    for pattern in content['patterns']:
        X.append(pattern)
        tags.append(tag)

tags_set = sorted(set(tags))
batch_size = 32
hidden_size = 512
num_class = len(tags_set)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load PhoBERT and tokenizer
phobert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Define the PhoBERT_finetuned class if not already defined
class PhoBERT_finetuned(nn.Module):
    def __init__(self, phobert, hidden_size, num_class):
        super(PhoBERT_finetuned, self).__init__()
        self.phobert = phobert
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer1 = nn.Linear(768, hidden_size)
        self.layer2 = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.phobert(sent_id, attention_mask=mask, return_dict=False)
        x = self.layer1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

def Chat(question):
    X_test = [question]

    token_test = tokenizer.batch_encode_plus(
        X_test,
        max_length=128,
        padding='max_length',
        truncation=True
    )
    X_test_mask = torch.tensor(token_test['attention_mask'])
    X_test = torch.tensor(token_test['input_ids'])

# Initialize the model with the architecture that matches the saved weights
    model = PhoBERT_finetuned(phobert, hidden_size=hidden_size, num_class=num_class)

# Load the saved weights with strict=False
    state_dict = torch.load('saved_weights.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False)

# Set the model to evaluation mode
    model.eval()


    with torch.no_grad():
        # Move data to the appropriate device
        X_test = X_test.to(device)
        X_test_mask = X_test_mask.to(device)

        # Make predictions
        preds = model(X_test, X_test_mask)
        preds = preds.detach().cpu().numpy()

    max_conf = float(np.max(preds, axis=1))

    if max_conf < -3.0:
        return "Tôi không rõ vấn đề này"

    preds = np.argmax(preds, axis=1)

    tag_pred = tags_set[int(preds)]

    for content in contents['intents']:
        tag = content['tag']
        if tag == tag_pred:
            res = content['responses']

    return random.choice(res)

while True:
    # Nhập câu hỏi từ người dùng
    user_input = input("Bạn: ")

    # Kiểm tra điều kiện dừng
    if user_input.lower() in ['exit', 'quit', 'stop']:
        print("Chương trình kết thúc.")
        break

    # Gọi hàm Chat để nhận phản hồi từ bot
    bot_response = Chat(user_input)
    
    # In câu trả lời của bot
    print("Bot:", bot_response)