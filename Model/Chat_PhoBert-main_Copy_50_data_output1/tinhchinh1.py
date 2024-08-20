import json
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load content from JSON files
with open('content.json', 'r', encoding="utf-8") as c:
    contents = json.load(c)

# Extract tags, patterns, and responses
tags = []
X = []
y = []
for content in contents['intents']:
    tag = content['tag']
    for pattern in content['patterns']:
        X.append(pattern)
        tags.append(tag)

tags_set = sorted(set(tags))
batch_size = 8
hidden_size = 512
num_class = len(tags_set)

# Load PhoBERT and tokenizer
phobert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Define the PhoBERT_finetuned class if not already defined
class PhoBERT_finetuned(nn.Module):
    def __init__(self, phobert, hidden_size, num_class):
        super(PhoBERT_finetuned, self).__init__()
        self.phobert = phobert
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, cls_hs = self.phobert(input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.fc(cls_hs)
        x = self.softmax(x)
        return x

def Chat(question):
    X_test = [question]

    token_test = tokenizer.batch_encode_plus(
        X_test,
        max_length=13,
        padding='max_length',
        truncation=True
    )
    X_test_mask = torch.tensor(token_test['attention_mask'])
    X_test = torch.tensor(token_test['input_ids'])

    # Initialize the model
    model = PhoBERT_finetuned(phobert, hidden_size=hidden_size, num_class=num_class)

    # Load the saved weights
    model.load_state_dict(torch.load('saved_weights.pth', map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        preds = model(X_test, X_test_mask)
        preds = preds.detach().cpu().numpy()

    max_conf = float(np.max(preds, axis=1))

    if max_conf < -0.2:
        return "Tôi không rõ vấn đề này"

    preds = np.argmax(preds, axis=1)
    tag_pred = tags_set[int(preds)]

    for content in contents['intents']:
        tag = content['tag']
        if tag == tag_pred:
            res = content['responses']

    return random.choice(res)

# Uncomment the line below to use the Chat function
# result = Chat("Câu hỏi của bạn ở đây")
# print(result)
while True:
    user_input = input("Nhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
    
    if user_input.lower() == 'exit':
        print("Thoát chương trình.")
        break

    result = Chat(user_input)
    print(result)
