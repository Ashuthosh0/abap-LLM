import traceback
from graph import app
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.title("ABAP LLM")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.status("🤖 Processing... Please wait!", expanded=True) as status:
        try:
            response = app.invoke({
                "query": user_query,
                "chat_history": [msg for msg in st.session_state.chat_history]
            })
            
            print(f"🔍 [DEBUG] Response from app.invoke(): {response}")     
            bot_reply = response.get("generation", "I am unable to process your request.")

        except Exception as e:
            tb_str = traceback.format_exc()  # Get full traceback as a string
            bot_reply = f"⚠️ Error: {str(e)}"
            print(f"❌ [ERROR] Exception occurred: {str(e)}")
            print(f"📜 [TRACEBACK]\n{tb_str}")  
            
            # Display traceback in UI for debugging
            with st.expander("See error details"):
                st.code(tb_str, language="python")

        # Update status message
        status.update(label="✅ Response Ready!", state="complete", expanded=False)

    # Display AI response
    with st.chat_message("AI"):
        st.markdown(bot_reply)

    # Append AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=bot_reply))

    # # Invoke model with updated chat history
    # try:
    #     with st.spinner("Processing..."):
    #         response = app.invoke({
    #             "query": user_query,
    #             "chat_history": [msg for msg in st.session_state.chat_history]
    #         })
            
    #     print(f"🔍 [DEBUG] Response from app.invoke(): {response}")     
    #     bot_reply = response.get("generation", "I am unable to process your request.")

    # except Exception as e:
    #     tb_str = traceback.format_exc()  # Get full traceback as a string
    #     bot_reply = f"Error: {str(e)}"
    #     print(f"❌ [ERROR] Exception occurred: {str(e)}")
    #     print(f"📜 [TRACEBACK]\n{tb_str}")  # Print full traceback
        

    # # Display AI response
    # with st.chat_message("AI"):
    #     st.markdown(bot_reply)

    # # Append AI response to chat history
    # st.session_state.chat_history.append(AIMessage(content=bot_reply))
