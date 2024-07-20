# TrendSpotterAI

TrendSpotterAI is a recommendation bot designed to suggest fashion products based on user queries. Leveraging LangChain and ChromaDB, this project performs efficient retrieval and ranking of fashion products from a pre-existing dataset.

## Project Overview

### Objectives
- Develop a recommendation bot to suggest fashion products based on user queries.
- Provide detailed product information including price, description, and average rating.
### Key Features
- Efficient Retrieval: Retrieve and rank relevant fashion products.
- Comprehensive Details: Provide detailed product information in response to user queries.

###Model and Data

#### Model Used
- Embedding Model: paraphrase-MiniLM-L6-v2 from Sentence Transformers for generating embeddings.
- RAG Framework: Utilizes LangChain’s capabilities for retrieval-augmented generation.

### Data Sources
- Fashion Dataset: A CSV file with the following columns:
- p_id: Unique product identifier
- name: Product name
- products: Product category or type
- price: Price of the product
- colour: Color of the product
- brand: Brand of the product
- img: Image URL of the product
- ratingCount: Number of ratings the product has received
- avg_rating: Average rating of the product
- description: Detailed description of the product
- p_attributes: Additional attributes of the product

Sample Data
<img width="1157" alt="Screenshot 2024-07-20 at 8 26 58 PM" src="https://github.com/user-attachments/assets/60162934-995f-4bb8-b59f-722007e0cd56">

### System Design
<img width="760" alt="Screenshot 2024-07-20 at 11 12 58 PM" src="https://github.com/user-attachments/assets/089a3f91-3b17-4b79-ba2e-2218c3ab9df5">

## Setup
### Installation
> To get started, install the required libraries using the following command:
```python
!pip install openai langchain langchain_openai langchain_community tiktoken chromadb sentence-transformers langchain-chroma
```

### Configuration
1. Prepare Your Dataset:
   - Ensure you have your fashion dataset in CSV format.
   - Update the code to reference your dataset file.
2.	Set Up API Keys:
   - For OpenAI models, ensure you have an API key and configure it in your environment.

#### Usage
1. Initialize the Model and Database:
   - Configure ChromaDB and LangChain to work with your dataset.
   - Clean and process the dataset, then generate embeddings and store them in ChromaDB.
2.	Run the Bot:
   - Start the interactive query system to test the recommendation bot.
```python
from your_module import interactive_query_system
interactive_query_system()
```
### Example Interaction

```python
# Load your dataset and clean the text
import pandas as pd
import re
import chromadb
from sentence_transformers import CrossEncoder
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Read and clean the dataset
df = pd.read_csv('FashionDatasetv2.csv')
def clean_text(text):
    cleaned_text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    cleaned_text = cleaned_text.lower().strip()
    return cleaned_text

df['description'] = df['description'].apply(clean_text)
# Initialize ChromaDB
client = chromadb.PersistentClient('path_to_chromadb')
fashion_prod_collection = client.get_or_create_collection(name='Fashion_docs')
metadata_list = df[['p_id', 'products', 'price', 'avg_rating', 'description']].to_dict(orient='records')
docu_list = df['description'].tolist()

# Add documents to ChromaDB
client.delete_collection(name='Fashion_docs')
fashion_prod_collection.add(documents=docu_list, metadatas=metadata_list, ids=[str(i) for i in range(len(docu_list))])

# Perform a query
query = input("Enter your query: ")
cache_results = cache_collection.query(query_texts=query, n_results=10)

# Search the collection
results = fashion_prod_collection.query(query_texts=[query], n_results=10, include=['documents', 'metadatas'])

# Process results and rank them
results_df = pd.DataFrame(results)
cross_encoder = CrossEncoder('path_to_model')
cross_inputs = [[query, response] for response in results_df['documents']]
cross_rerank_scores = cross_encoder.predict(cross_inputs)
results_df['reranked_scores'] = cross_rerank_scores

# Generate response using LangChain
prompt_template = """
You are a helpful assistant in the fashion product domain who can effectively answer user queries about products, average rating, and description. You strictly follow the below-provided instructions.
You have a question asked by the user in '{query}' and you have some search results from a corpus of products documents in the dataframe '{top_3_RAG}'. These search results are essentially product details that may be relevant to the user query.
The column 'documents' inside this dataframe contains the actual text from the product document and the column 'metadata' contains the name, p_id, price, avg_rating, descriptions. The text inside the document may also contain JSON format.
Use the documents in '{top_3_RAG}' to answer the query '{query}'. Frame an informative answer and also, use the dataframe to return the relevant product name, avg_rating, price, description.
"""

prompt = PromptTemplate(
    input_variables=["query", "top_3_RAG"],
    template=prompt_template
)
chat = ChatOpenAI(model_name="gpt-3.5-turbo")
chain = LLMChain(llm=chat, prompt=prompt)
response = chain.run(query=query, top_3_RAG=results_df.head(3))
print(response)
```
### Troubleshooting
- Error Connecting to API: Ensure your API keys are correctly set and that your internet connection is stable.
- No Results Found: Check if the dataset is properly loaded and if queries match the available data.

### Contributing
- Contributions are welcome! Please fork the repository and submit a pull request with your changes.
### License
This project is licensed under the MIT License. See the [LICENSE file](./MIT%20License.pdf) file for details.
