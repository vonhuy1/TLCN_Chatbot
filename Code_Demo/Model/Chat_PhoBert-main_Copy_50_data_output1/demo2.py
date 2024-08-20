import re
import json

def convert_to_json(chatito_code):
    intents = re.findall(r'%\[\w+\](.*?)%\n', chatito_code, re.DOTALL)

    data = {"intents": []}

    for intent in intents:
        lines = intent.strip().split('\n')
        tag_line = lines[0]
        patterns_line = lines[1]
        responses_line = lines[2]

        tag = re.search(r'%\[(\w+)\]', tag_line).group(1)
        patterns = re.search(r'~patterns\((.*?)\)', patterns_line).group(1).split('|')
        responses = re.search(r'~responses\((.*?)\)', responses_line).group(1).split('|')

        intent_data = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }

        data["intents"].append(intent_data)

    return json.dumps(data, indent=2, ensure_ascii=False)

# Đoạn mã Chatito của bạn
chatito_code = """
%[greeting]('training': '100')
    ~patterns(Xin chào|Bạn khỏe không|Có ai ở đó không?|Xin chào|Yolo|Nàyyyy|Aloha|Có ai trả lời được không)
    ~responses(Xin chào!|Chào :-)|Xin chào, tôi có thể giúp gì cho bạn?|Xin chào, tôi có thể giúp gì?|Xin chào, cảm ơn đã ghé thăm|Rất vui khi gặp lại)

%[goodbye]('training': '100')
    ~patterns(cya|Hẹn gặp lại|Tạm biệt|bye)
    ~responses(Hẹn gặp lại, cảm ơn đã ghé thăm|Thật buồn khi bạn đi :(|Chúc bạn ngày mới tốt lành|Tạm biệt!|Tạm biệt! Hãy Quay Lại Sớm Nhé.)
"""

# Chuyển đổi thành JSON
json_data = convert_to_json(chatito_code)

# In kết quả
print(json_data)
