
import openai

openai.api_key = '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'

# Tạo yêu cầu hoàn thành
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Xin chào"}
    ]
)
print(response['choices'][0]['message']['content'])
