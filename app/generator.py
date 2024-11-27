from openai import OpenAI

class LLMClient:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        self.client = OpenAI(
            base_url="https://api-inference.huggingface.co/v1/",
            api_key=api_key
        )
        self.model_name = model_name

    def get_response_from_model(self, context, user_query):
        """
        Generate a response using the LLM and provided context.
        """
        prompt_text = f"""
        Answer the following question based on the context provided:

        Context:
        {context}

        Question:
        {user_query}
        """
        messages = [{"role": "user", "content": prompt_text}]
        completion = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages, 
            max_tokens=500
        )
        return completion.choices[0].message.content
