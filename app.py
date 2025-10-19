import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Define the path to the saved model and tokenizer directory
save_directory = "./urdu_roman_translator"

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(save_directory)

# Load the model
model = GPT2LMHeadModel.from_pretrained(save_directory)

st.title("Urdu to Roman Translation")

def translate_urdu_to_roman(urdu_line):
    input_text = f"<|startoftext|>Urdu: {urdu_line} Roman:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    # Move the input tensor to the same device as the model
    inputs = inputs.to(model.device)
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
urdu_input = st.text_input("Enter Urdu text here:")
translate_button = st.button("Translate")

if translate_button:
    # Translation logic will go here in the next step
    pass
