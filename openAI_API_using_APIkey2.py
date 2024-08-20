import openai


def generate_chat_completion(apikey,model,messages,temperature,top_p,stream,presence_penalty,max_tokens,frequency_penalty):
    #api_key = "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"
    openai.api_key = apikey
    response = openai.ChatCompletion.create(
        model=model,
        messages=
    [{"role": "user", "content": messages}],
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )
    # Return the generated completion
    return response['choices'][0]['message']['content']


