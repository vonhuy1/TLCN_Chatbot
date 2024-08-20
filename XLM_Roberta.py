from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def get_answer(context, question):
    # Replace '/path/to/your/model' with the actual path where you saved your model
    model_path = r"./XLM_Roberta/Model"
    # Load the model and tokenizer
    QA_model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline('question-answering', model=QA_model, tokenizer=tokenizer)
    res = pipe(question=question, context=context)
    answer = res['answer']
    drop_char = [',', ';', '/', '//']
    if answer[-1] in drop_char:
        answer = answer[:-1]
    return answer
