import pprint
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)


if __name__ == "__main__":
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    # Settings.llm = None  # type: ignore

    storage_context = StorageContext.from_defaults(persist_dir="./storage_contexts/")
    index = load_index_from_storage(storage_context)

    question = "Recipe with cucumber"
    print(f'The question asked to retriver:  "{question}"')

    retriver = index.as_retriever()
    pprint.pp(retriver.retrieve(question))
