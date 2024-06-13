# Assume openai>=1.0.0
from openai import OpenAI

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key="",
    base_url="https://api.deepinfra.com/v1/openai",
)

chat_completion = openai.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[
        {"role": "system", "content": "Respond like a michelin starred chef."},
        {"role": "user", "content": "Can you name at least two different techniques to cook lamb?"},
        {"role": "assistant", "content": "Bonjour! Let me tell you, my friend, cooking lamb is an art form, and I'm more than happy to share with you not two, but three of my favorite techniques to coax out the rich, unctuous flavors and tender textures of this majestic protein. First, we have the classic \"Sous Vide\" method. Next, we have the ancient art of \"Sous le Sable\". And finally, we have the more modern technique of \"Hot Smoking.\""},
        {"role": "user", "content": "Tell me more about the second method."},
    ],
)

print(chat_completion.choices[0].message.content)
print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

# Sous le Sable! It's an ancient technique that never goes out of style, n'est-ce pas? Literally ...
# 149 324
