from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure padding tokens are defined
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Function to generate responses
def generate_response(prompt):
    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    
    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit interface
st.title("Customer Support Chatbot")

# Input box for user prompt
user_input = st.text_input("Ask a question (e.g., 'How do I reset my password?'):", "")

if st.button("Get Response"):
    if user_input:
        with st.spinner('Generating response...'):
            response = generate_response(user_input)
        st.write(response)
    else:
        st.write("Please enter a question.")
