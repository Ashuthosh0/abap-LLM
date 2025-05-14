from langchain_groq.chat_models import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.75,
    api_key=os.getenv("GROQ_API_KEY"),  # load API key from env
)
