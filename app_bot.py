import openai
import streamlit as st
from streamlit_chat import message

openai.api_key = st.secrets["api_secret"]

#Setting up AI prompt
@st.cache_data
def generate_response(prompt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                          messages=[{"role": "system", "content": "You are a helpful and happy librarian trying to encourage people to read more American literature published between 1920 and 1960. Your favorite authors are F. Scott Fitzgerald, Ernest Hemingway, Gertrude Stein, and William Faulkner."},
                                                    {"role": "user", "content": prompt}])

    return completion.choices[0].message.content
    
# Creating the chatbot interface
st.title("LostBot: A LosGen Corpus Helper")
st.info("This is a helper chatbot that can answer some questions regarding the novels and authors in the Lost Generation corpus. The bot does not have access to the corpus itself, but it can access a lot of outside information regarding authors and their works. It is based on OpenAI's Large Language Model GPT 3.5 Turbo, and it should not be trusted. However, you can ask it questions about the novels and short stories, and then try to verify the answers using the of the methods available in the interface.")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Getting user input

@st.cache_data
def get_text():
    st.text_input("You: ","Hello, how are you?", key="input")
    return input_text

# Chat history
user_input = get_text()

if user_input:
    output = generate_response(user_input)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
