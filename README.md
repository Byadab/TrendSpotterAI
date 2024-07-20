# TrendSpotterAI

TrendSpotterAI is a recommendation bot that suggests fashion products based on user queries. This project leverages LangChain and ChromaDB to perform efficient retrieval and ranking of fashion products from a pre-existing dataset.

## Project Overview

### Objectives
- Build a recommendation bot to suggest fashion products based on user queries.
- Provide detailed product information including price, description, and average rating.

### Key Features
- Retrieve and rank relevant fashion products.
- Provide comprehensive product details in response to user queries.

## Model and Data

### Model Used
- **Embedding Model:** `paraphrase-MiniLM-L6-v2` from Sentence Transformers for generating embeddings.
- **RAG Framework:** Utilizing LangChain's capabilities for retrieval-augmented generation.

### Data Sources
- **Fashion Dataset:** CSV file with columns: `p_id`, `name`, `products`, `price`, `colour`, `brand`, `img`, `ratingCount`, `avg_rating`, `description`, `p_attributes`, `desc_len`.

### Sample Data
- `p_id`: Unique product identifier.
- `name`: Product name.
- `price`: Price of the product.
- `avg_rating`: Average rating of the product.
- `description`: Detailed description of the product.

## Setup

### Installation
Install the required libraries using the following commands:

```bash
pip install langchain transformers langchain_community openai tiktoken chromadb
