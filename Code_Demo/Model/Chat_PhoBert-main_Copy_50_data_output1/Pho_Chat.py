from training_phobert import model, tokenizer, tags_set, device
import torch
import json
import random
import numpy as np

def Chat(question):
    X_test = [question]
    with open('test_content.json', 'r', encoding="utf-8") as c:
        contents = json.load(c)

    tags_test = []

    for content in contents['intents']:
        tag = content['tag']
        for pattern in content['patterns']:
            tags_test.append(tag)
        
    token_test = tokenizer.batch_encode_plus(
        X_test,
        max_length=13,
        padding='max_length',
        truncation=True
    )
    X_test_mask = torch.tensor(token_test['attention_mask'])
    X_test = torch.tensor(token_test['input_ids'])

    path = 'saved_weights.pth'
    
    # Load model weights
    model.load_state_dict(torch.load(path, map_location=device))
    
    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        preds = model(X_test.to(device), X_test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    max_conf = float(np.max(preds, axis=1))
    print(max_conf)
    
    if max_conf < -0.2:
        return "Tôi không rõ vấn đề này"

    preds = np.argmax(preds, axis=1)
    print(preds)

    tag_pred = tags_set[int(preds)]
    print(tag_pred)

    for content in contents['intents']:
        tag = content['tag']
        if tag == tag_pred:
            res = content['responses']

    return random.choice(res)
