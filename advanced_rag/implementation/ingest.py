from pathlib import Path
from multiprocessing import Pool

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from tenacity import retry, wait_exponential


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"

DB_NAME = str(
    Path(__file__).parent.parent / "preprocessed_db"
)

COLLECTION_NAME = "docs"

EMBEDDING_MODEL = "text-embedding-3-large"

KNOWLEDGE_BASE_PATH = (
    Path(__file__).parent.parent / "knowledge-base"
)

AVERAGE_CHUNK_SIZE = 100

WORKERS = 3

wait = wait_exponential(
    multiplier=1,
    min=10,
    max=240,
)

openai = OpenAI()


# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------

class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):

    headline: str = Field(
        description=(
            "A brief heading for this chunk, typically a few words, "
            "that is most likely to be surfaced in a query"
        )
    )

    summary: str = Field(
        description=(
            "A few sentences summarizing the content of this chunk "
            "to answer common questions"
        )
    )

    original_text: str = Field(
        description=(
            "The original text of this chunk from the provided "
            "document, exactly as is, not changed in any way"
        )
    )

    def as_result(self, document):

        metadata = {
            "source": document["source"],
            "type": document["type"],
        }

        return Result(
            page_content=(
                self.headline
                + "\n\n"
                + self.summary
                + "\n\n"
                + self.original_text
            ),
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


# ---------------------------------------------------------
# STEP 1: LOAD DOCUMENTS
# ---------------------------------------------------------

def fetch_documents():
    """
    Load every Markdown document from the knowledge base.

    Folder names are used as document types.
    """

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():

        doc_type = folder.name

        for file in folder.rglob("*.md"):

            with open(file, "r", encoding="utf-8") as f:

                documents.append(
                    {
                        "type": doc_type,
                        "source": file.as_posix(),
                        "text": f.read(),
                    }
                )

    print(f"Loaded {len(documents)} documents")

    return documents


# ---------------------------------------------------------
# STEP 2: CREATE SEMANTIC CHUNKING PROMPT
# ---------------------------------------------------------

def make_prompt(document):

    how_many = (
        len(document["text"]) // AVERAGE_CHUNK_SIZE
    ) + 1

    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.

The document is of type:
{document["type"]}

The document has been retrieved from:
{document["source"]}

A chatbot will use these chunks to answer questions about the company.

You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.

This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.

There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.

Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):

    return [
        {
            "role": "user",
            "content": make_prompt(document),
        }
    ]


# ---------------------------------------------------------
# STEP 3: USE THE LLM TO SEMANTICALLY CHUNK A DOCUMENT
# ---------------------------------------------------------

@retry(wait=wait)
def process_document(document):

    messages = make_messages(document)

    response = completion(
        model=MODEL,
        messages=messages,
        response_format=Chunks,
    )

    reply = response.choices[0].message.content

    doc_as_chunks = (
        Chunks.model_validate_json(reply).chunks
    )

    return [
        chunk.as_result(document)
        for chunk in doc_as_chunks
    ]


# ---------------------------------------------------------
# STEP 4: PROCESS ALL DOCUMENTS
# ---------------------------------------------------------

def create_chunks(documents):
    """
    Create semantic chunks using multiple workers.

    If API rate limits occur, change WORKERS to 1.
    """

    chunks = []

    with Pool(processes=WORKERS) as pool:

        for result in tqdm(
            pool.imap_unordered(
                process_document,
                documents,
            ),
            total=len(documents),
        ):

            chunks.extend(result)

    return chunks


# ---------------------------------------------------------
# STEP 5: EMBED CHUNKS AND STORE THEM IN CHROMA
# ---------------------------------------------------------

def create_embeddings(chunks):

    chroma = PersistentClient(
        path=DB_NAME
    )

    existing_collections = [
        collection.name
        for collection in chroma.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:

        chroma.delete_collection(
            COLLECTION_NAME
        )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    vectors = [
        item.embedding
        for item in embedding_response.data
    ]

    collection = chroma.get_or_create_collection(
        COLLECTION_NAME
    )

    ids = [
        str(i)
        for i in range(len(chunks))
    ]

    metadata = [
        chunk.metadata
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadata,
    )

    print(
        f"Vectorstore created with "
        f"{collection.count()} documents"
    )


# ---------------------------------------------------------
# RUN INGESTION
# ---------------------------------------------------------

if __name__ == "__main__":

    documents = fetch_documents()

    chunks = create_chunks(documents)

    create_embeddings(chunks)

    print("Ingestion complete")