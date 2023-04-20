# Auto-GPT-HF-Model-Plugin

An attempt to replace the chat completion in Auto-GPT with a model hosted on Hugging Face

In this case I'm playing with stablelm-base-alpha-3b

Same as Auto-GPT look in .env.template and copy to .env

There's two classes, one using the inference API on Hugging Face, the other using their local model (downloads the model and uses CUDA)

