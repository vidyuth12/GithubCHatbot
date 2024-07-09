import os
import re
import subprocess
from typing import Optional, Tuple

import nest_asyncio
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import (
    Settings,
    PromptTemplate,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder

class GitHubSummarizer:
    def __init__(self, openai_api_key: str, model: str = 'gpt-3.5-turbo-0125'):
        self.openai_api_key = openai_api_key
        self.model = model
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embedding_model = self._load_embedding_model()
        self.llm = self._setup_llm()
        nest_asyncio.apply()

    def _load_embedding_model(self):
        model_name = "BAAI/bge-large-en-v1.5"
        encode_kwargs = {"normalize_embeddings": True}
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs,
        )
        return LangchainEmbedding(embedding_model)

    def _setup_llm(self):
        class OpenAIModel:
            def __init__(self, client, model, temperature=0.7, max_tokens=150):
                self.client = client
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens
            
            def generate(self, prompt):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()

        return OpenAIModel(self.client, self.model)

    @staticmethod
    def parse_github_url(url: str) -> Tuple[Optional[str], Optional[str]]:
        pattern = r"https://github\.com/([^/]+)/([^/]+)"
        match = re.match(pattern, url)
        return match.groups() if match else (None, None)

    @staticmethod
    def clone_github_repo(repo_url: str) -> None:
        try:
            print('Cloning the repo ...')
            subprocess.run(["git", "clone", repo_url], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")

    def setup_query_engine(self, github_url: str):
        owner, repo = self.parse_github_url(github_url)
        if not owner or not repo:
            print('Invalid GitHub repo URL, try again!')
            return None

        input_dir_path = f"C:/Users/Vidyuth/OneDrive/Desktop/RAG_githubcodes/{repo}"
        if not os.path.exists(input_dir_path):
            self.clone_github_repo(github_url)

        try:
            loader = SimpleDirectoryReader(
                input_dir=input_dir_path,
                required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
                recursive=True
            )
            docs = loader.load_data()

            if not docs:
                print("No data found, check if the repository is not empty!")
                return None

            index = VectorStoreIndex.from_documents(docs, show_progress=True, embed_model=self.embedding_model)
            query_engine = index.as_query_engine()

            qa_prompt_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
            )

            print("Data loaded successfully!!")
            print("Ready to chat!!")
            return query_engine

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def query(self, github_url: str, question: str) -> str:
        query_engine = self.setup_query_engine(github_url)
        if query_engine:
            response = query_engine.query(question)
            return str(response)
        return "Failed to set up query engine."

# Usage
if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    summarizer = GitHubSummarizer(openai_api_key)
    github_url = "https://github.com/balsa-project/balsa"
    question = "Can you explain how the optimizer works?"
    
    response = summarizer.query(github_url, question)
    print(response)