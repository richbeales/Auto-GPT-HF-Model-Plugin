# Auto-GPT-HF-Model-Plugin

An attempt to replace the chat completion in Auto-GPT with a model hosted on Hugging Face

Currently very experimental (read: broken)

In this case I'm playing with stablelm-tuned-alpha-3b.

Same as Auto-GPT look in .env.template and copy to .env

There's three classes included (you have to change the one you use in the code at the top of __init__.py

    1. one using the (free) inference API on Hugging Face
       https://huggingface.co/stabilityai/stablelm-base-alpha-3b

    2. one using the (hosted - you need to spin up an instance) inference API on Hugging Face,
       https://ui.endpoints.huggingface.co/endpoints
    
    3. other using their local model (downloads the model and uses CUDA - doesn't work on my machine yet (CUDA issue) this is just their sample code)

