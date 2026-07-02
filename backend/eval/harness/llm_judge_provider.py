# eval/harness/llm_judge_provider.py
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from eval.config import JUDGE_MODEL, EMBEDDING_MODEL

def get_ragas_llm():
    return LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0))

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL))