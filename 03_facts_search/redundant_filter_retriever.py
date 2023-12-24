from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever
from langchain.vectorstores.chroma import Chroma


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string or input-string. In this example, we pass an instance of openAi embeddings.
        emb = self.embeddings.embed_query(query)
        # take embeddings and feed them into that max_marginal_relevance_search_by_vector
        # to remove the duplicates for us automatically. This is given by chromadb.
        # lambda_mult :- 0 to 1. Higher values allow for simillar docs
        result = self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb, lambda_mult=0.8
        )
        return result

    async def aget_relevant_documents(self):
        return []
