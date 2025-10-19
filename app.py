import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_save_path = "./urdu_roman_translator"

@st.cache_resource
def load_model_and_tokenizer(path):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'additional_special_tokens': ['<|startoftext|>', '<|endoftext|>']})
    model = GPT2LMHeadModel.from_pretrained(path)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_save_path)

def translate_urdu_to_roman(urdu_line, model, tokenizer):
    input_text = f"<|startoftext|>Urdu: {urdu_line} Roman:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    # Decode the output and remove the input text and special tokens
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Find the index of "Roman:" and extract the text after it
    roman_start_index = decoded_output.find("Roman:")
    if roman_start_index != -1:
        roman_text = decoded_output[roman_start_index + len("Roman:"):].strip()
        # Remove the endoftext token if present
        roman_text = roman_text.replace("<|endoftext|>", "").strip()
        return roman_text
    else:
        return "Translation failed."


st.title("Urdu to Roman Poetry Translator")

urdu_input = st.text_area("Enter Urdu text here:")

if st.button("Translate"):
    if urdu_input:
        roman_output = translate_urdu_to_roman(urdu_input, model, tokenizer)
        st.write("Romanized Text:")
        st.write(roman_output)
    else:
        st.write("Please enter some Urdu text.")
