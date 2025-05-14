#-------STREAMLIT UI------------

import traceback
from graph import app
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.title("ABAP LLM")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# to display the convo
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# take user's query
user_query = st.chat_input("Type your message here...")
if user_query:
    # append user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # display user message
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.status(" Processing... Please wait!", expanded=True) as status:
        try:
            response = app.invoke({
                "query": user_query,
                "chat_history": [msg for msg in st.session_state.chat_history]
            })
            
            print(f"🔍 [DEBUG] Response from app.invoke(): {response}")     
            bot_reply = response.get("generation", "I am unable to process your request.")

        except Exception as e:
            tb_str = traceback.format_exc()  
            bot_reply = f" Error: {str(e)}"
            print(f" [ERROR] Exception occurred: {str(e)}")
            print(f" [TRACEBACK]\n{tb_str}")  
            
            # display the traceback in ui for debugging etc
            with st.expander("See error details"):
                st.code(tb_str, language="python")

    
        status.update(label=" Response Ready!", state="complete", expanded=False)

    # display the response of the ai
    with st.chat_message("AI"):
        st.markdown(bot_reply)

    # append the ai response to the chant history
    st.session_state.chat_history.append(AIMessage(content=bot_reply))

   
