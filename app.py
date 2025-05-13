import gradio as gr
import os
import requests

# Load GROQ API key from environment (set it in Hugging Face secrets)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(GROQ_API_KEY)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192" # Balanced and fast for Q&A bots

SYSTEM_PROMPT = """You are a friendly and helpful travel advisor.
You answer user questions about travel destinations, planning, and tips in a clear and engaging way."""

def query_groq(message, chat_history):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, bot in chat_history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": message})
    
    response = requests.post(GROQ_API_URL, headers=headers, json={
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7
    })
    
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        return reply
    else:
        return f"Error {response.status_code}: {response.text}"

def respond(message, chat_history):
    bot_reply = query_groq(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_reply})
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("## Your Custom Chatbot (Powered by GROQ LLM)")
    chatbot = gr.Chatbot(type = 'messages')
    msg = gr.Textbox(label="Ask a question")
    clear = gr.Button("Clear Chat")
    state = gr.State([])
    
    msg.submit(respond, [msg, state], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch()