import openai


def main():
    # Set your OpenAI API key
    api_key = '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'
    openai.api_key = api_key
    # List all models
    models = openai.Model.list()
    # Extract only the "id" values and create a list
    model_ids = [d["id"] for d in models.data if d["id"].startswith("ft:gpt")]
    return model_ids



