from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LLMGenerative:
    def __init__(self):
        self.tokenizer = AutoTokenizer(READER_MODEL_NAME)
        self.internal_model = AutoModelForCasualLM(READER_MODEL_NAME)
        self.model = pipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.1,
                return_full_text=False,
                max_new_tokens=500,
                )

    def prompt_creation(query: String, context: Array[]):
        prompt_in_chat_format = [
                {
                    "role": "system",
                    "content": """Using the information contained in the context,
                    give a comprehensive answer to the question.
                    Respond only to the question asked, response should be concise and relevant to the question.
                    Provide the number of the source document when relevant.
                    If the answer cannot be deduced from the context, do not give an answer.""",
                    },
                {
                    "role": "user",
                    "content": """Context:
                    {context}
                    ---
                    Now here is the question you need to answer.

                    Question: {query}""",
                    },
                ]
        return self.tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

    def tokenize_query(complete_query: String):
        return self.tokenizer.apply_chat_template(
                complete_query, tokenize=False, add_generation_prompt=True
                )

    def get_answer_to_prompt(prompt: String):
        return self.model(prompt)[0]["generated_text"]

    def get_response_based_on_context(query: String, context: String[]):
        prompt = prompt_creation(query, context)
        tokenized_prompt = tokenize_query(prompt)
        return get_answer_to_prompt(tokenized_prompt)


