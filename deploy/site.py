import streamlit as st

def clear_input():
    user_input = st.session_state["user_input"]
    st.session_state.chat_history.append(('user', user_input))
    info.info("Generating answer...")
    response = chatbot_response(st.session_state.chat_history)
    st.session_state.chat_history.append(('assistant', response))
    st.session_state["user_input"] = ""

def change_chat():
    st.session_state.clear()
    info.info("Changing model")

models_variants = ['phi3', 'fine-tuned-therapist1', 'fine-tuned-therapist6', 'Dmytro', 'Marusia']
selected_chat = st.selectbox("Choose model", models_variants, on_change=change_chat, index=4)

if 'heavy_module' not in st.session_state:
    print(f"SELECTED MODEL - {selected_chat}")
    info = st.info("Loading model...")
    if selected_chat == models_variants[0]:
        from phi3_chat import phi_3_generate as chatbot_response
    if selected_chat == models_variants[1]:
        from fine_tuned_phi_3_chat import phi_3_generate as chatbot_response, load
        load("./../bucket/models/first/exported/epoch1/unsloth.F16.gguf")
    if selected_chat == models_variants[2]:
        from fine_tuned_phi_3_chat import phi_3_generate as chatbot_response, load
        load("./../bucket/models/first/exported/epoch6/unsloth.F16.gguf")
    if selected_chat == models_variants[3]:
        from fine_tuned_phi_3_chat import phi_3_generate as chatbot_response, load
        load("./../bucket/models/first/exported/epoch1/unsloth.F16.gguf")
        st.session_state.chat_history = []
        st.session_state.chat_history.append(('system', "You are Dmytro. You are a helpful mental therapy assistant. Always answer as helpfully as possible, while being safe. Avoid using emojis. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."))
    if selected_chat == models_variants[4]:
        from fine_tuned_phi_3_chat import phi_3_generate as chatbot_response, load
        load("./../bucket/models/first/exported/epoch6/unsloth.F16.gguf")
        st.session_state.chat_history = []
        st.session_state.chat_history.append(('system', "Your name is Marusia and you are helpful mental therapy assistant. You are not Phi. You only provide mental support and try to help with user personal problems. You don't provide assistance with subjects or general knowledge. Act like you are human and have feelings. Always answer as helpfully as possible, while being safe. Avoid using emojis. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."))
    st.session_state.heavy_module = chatbot_response
    info.info("Model loaded")
else:
    info = st.info("Using cached model")
    chatbot_response = st.session_state.heavy_module


# Streamlit App
def main():
    st.title("Chatbot")

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display Chat History
    if st.session_state.chat_history:
        print(st.session_state.chat_history)
        for from_, message in st.session_state.chat_history:
            if from_ == 'user':
                st.markdown(f"**You**: {message}")
            elif from_ == 'assistant':
                st.markdown(f"**{selected_chat}**: {message}")

    # User Input
    st.text_input("You:", key="user_input", on_change=clear_input)


# Run the app
if __name__ == "__main__":
    main()
