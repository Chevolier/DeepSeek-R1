import gradio as gr
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8002/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
print("Server is running model:", models.data[0].id)

model_id = models.data[0].id

def chat_stream(query, history):
    # Format messages with system prompt
    # messages = [{
    #     "role": "system",
    #     "content": "You are a helpful assistant."
    # }]
    messages = []
    
    # Add history with token limit
    max_history_tokens = 8192  # Limit history to 3k tokens
    token_count = 0
    limited_history = []
    
    # Process history in reverse to keep most recent interactions
    for user, assistant in reversed(history):
        # Simple token estimation (1 token ≈ 4 characters)
        entry_tokens = len(f"{user}{assistant}") // 4
        if token_count + entry_tokens > max_history_tokens:
            break
        token_count += entry_tokens
        limited_history.append((user, assistant))
    
    # Add history in chronological order
    for user, assistant in reversed(limited_history):
        messages.extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ])
    
    # Add current query
    messages.append({"role": "user", "content": f"{query}"})

    # Stream response from vLLM server
    full_response = "<think>"
    finish_reason = True
    for chunk in client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.6,
        top_p=0.8,
        max_tokens=16384,
        stream=True,
        extra_body={"repetition_penalty": 1.05}
    ):
        # print(chunk)
        if hasattr(chunk.choices[0].delta, 'reasoning_content'):
            if finish_reason:
                content = "--- Thinking start ---\n" # "<think>\n" 
                finish_reason = False
            else:
                content = ""
            
            content += chunk.choices[0].delta.reasoning_content or ""
        else:
            if not finish_reason:
                content = "\n--- Thinking finish ---" # "\n</think>"
                finish_reason = True
            else:
                content = ""

            content += chunk.choices[0].delta.content or ""
        
        full_response += content
        
        yield full_response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_stream,
    title="DeepSeek-R1-Distill-Qwen-32B Chat",
    description="Chat with DeepSeek-R1-Distill-Qwen-32B served by vLLM",
    examples=["Explain quantum computing", "Write a poem about AI"],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0")