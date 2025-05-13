# ABAP LLM Project

This project leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) techniques to provide ABAP coding guidance and best practices. 

##  Features
- ABAP code retrieval and suggestions
- Fine-tuned ABAP best practices
- Integration with vector databases for efficient search
- API-based interaction with LLMs
- Web interface for user interaction

## Setup Instructions

### 1 Clone the Repository
```sh
git clone https://github.com/Ashuthosh0/abap-LLM.git
cd abapLLM
```

### 2️ Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️ Set Up API Keys
This project requires API keys for:
- **LLM Inference (Groq)**
- **Tavily API for web search**

#### Add API Keys to `.env` File
Create a `.env` file in the project root and add:
```sh
CHAT_GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5️ Run the Application
```sh
streamlit run app.py
```

##  Development & Contribution
Feel free to fork this repository, submit pull requests, and improve the model. 

### Future Enhancements
- Improved retrieval techniques
- More fine-tuned ABAP best practices
- Enhanced UI for better user experience


