import openai


def main(prompt1,model, maxtoken,key,stop,id,tmp):
    api_key = key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Openai-Organization":id
    }
    openai.api_key = api_key
    response = openai.Completion.create(
        engine=model,
        prompt=prompt1,
        temperature=tmp,
        n=1,
        stop=stop,
        max_tokens=maxtoken,
        headers=headers
    )

    return response['choices'][0]['text']


