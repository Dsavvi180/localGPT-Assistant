## Local AI assistant

This project makes use of LangChain.py framework as an interface between the LLMs and user.
The LLM's used are sourced from gpt4all

## Use case:
This project was created to leverage the power of AI locally, for two main reasons:
1. Locally running LLM's is a workaround privacy issues relating to personal data and private information, for example processing work data that involves private data and is protected under GDPR.
2. It provides a means of 'teaching' the LLM a specific use case. A specific use case context can be passed through to the LLM which can answer questions in a specified format. In addition, relevant
data can be passed through to the LLM automatically to be included in its information source to learn from and gather information to answer the proposed questions and tasks.

## Process
1. The LLM is downloaded and its path is referenced.
2. Relevant data is gathered regarding the specific tasks the AI is to be configured for completing.
3. The data is then read from a pdf and is embedded into a vector data base by an embedding NLP model.
4. The vector data base is configured using ChromaDB
5. The question for the GPT is embedded as well, then a Vector Similarity search is made in the data base to return relevant data that is similar to the question's context.
6. The Search result return natural language based on IDs that correspond to the relevant vectors. The IDs contain the natural language data and is associated with each vector in the Chroma DB.
7. A template is created using LangChain.py
8. The search result is passed through into the template, along with the specific question and predefined instructions.
9. The LLM is configured using the documentation on gpt4all.
10. The template is passed through to the LLM.
11. The LLM returns an answer in a specified format that can be altered in length by sopecifying the tokens.

