from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

load_dotenv(override=True)


# Handbook's advanced model:
MODEL = "groq/openai/gpt-oss-120b"

# Alternative from the handbook:
# MODEL = "openai/gpt-4.1-nano"


DB_NAME = str(
    Path(__file__).parent.parent / "preprocessed_db"
)

COLLECTION_NAME = "docs"

EMBEDDING_MODEL = "text-embedding-3-large"


RETRIEVAL_K = 20

FINAL_K = 10


wait = wait_exponential(
    multiplier=1,
    min=10,
    max=240,
)


openai = OpenAI()


chroma = PersistentClient(
    path=DB_NAME
)


collection = chroma.get_or_create_collection(
    COLLECTION_NAME
)


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.

You are chatting with a user about Insurellm.

Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.

If you don't know the answer, say so.

For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:

{context}

With this context, please answer the user's question.

Be accurate, relevant and complete.
"""


# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------

class Result(BaseModel):

    page_content: str

    metadata: dict


class RankOrder(BaseModel):

    order: list[int] = Field(
        description=(
            "The order of relevance of chunks, "
            "from most relevant to least relevant, "
            "by chunk id number"
        )
    )


# ---------------------------------------------------------
# STEP 1: QUERY REWRITING
# ---------------------------------------------------------

@retry(wait=wait)
def rewrite_query(
    question,
    history=[],
):

    """
    Rewrite a conversational question into a short,
    specific Knowledge Base search query.
    """

    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.

You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:

{history}

And this is the user's current question:

{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.

It should be a VERY short specific question most likely to surface content.

Focus on the question details.

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""

    response = completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": message,
            }
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# STEP 2: VECTOR SEARCH
# ---------------------------------------------------------

def fetch_context_unranked(question):

    # Convert question into an embedding.
    embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )

    query_vector = (
        embedding_response.data[0].embedding
    )

    # Search Chroma.
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=RETRIEVAL_K,
    )

    chunks = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):

        chunks.append(
            Result(
                page_content=document,
                metadata=metadata,
            )
        )

    return chunks


# ---------------------------------------------------------
# STEP 3: MERGE + DE-DUPLICATE SEARCH RESULTS
# ---------------------------------------------------------

def merge_chunks(
    original_chunks,
    rewritten_chunks,
):

    merged = original_chunks[:]

    existing = [
        chunk.page_content
        for chunk in original_chunks
    ]

    for chunk in rewritten_chunks:

        if chunk.page_content not in existing:

            merged.append(chunk)

            existing.append(
                chunk.page_content
            )

    return merged


# ---------------------------------------------------------
# STEP 4: LLM RERANKING
# ---------------------------------------------------------

@retry(wait=wait)
def rerank(question, chunks):

    system_prompt = """
You are a document re-ranker.

You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.

The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.

You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.

Reply only with the list of ranked chunk ids, nothing else.

Include all the chunk ids you are provided with, reranked.
"""

    user_prompt = (
        "The user has asked the following question:\n\n"
        f"{question}\n\n"
        "Order all the chunks of text by relevance "
        "to the question, from most relevant to least relevant. "
        "Include all the chunk ids you are provided with, reranked.\n\n"
    )

    user_prompt += "Here are the chunks:\n\n"

    for index, chunk in enumerate(chunks):

        user_prompt += (
            f"# CHUNK ID: {index + 1}:\n\n"
            f"{chunk.page_content}\n\n"
        )

    user_prompt += (
        "Reply only with the list of ranked "
        "chunk ids, nothing else."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = completion(
        model=MODEL,
        messages=messages,
        response_format=RankOrder,
    )

    reply = (
        response
        .choices[0]
        .message
        .content
    )

    order = (
        RankOrder
        .model_validate_json(reply)
        .order
    )

    return [
        chunks[index - 1]
        for index in order
    ]


# ---------------------------------------------------------
# STEP 5: COMPLETE RETRIEVAL PIPELINE
# ---------------------------------------------------------

def fetch_context(
    original_question,
    history=[],
):

    # Rewrite the question.
    rewritten_question = rewrite_query(
        original_question,
        history,
    )

    # Search using the user's original wording.
    original_results = fetch_context_unranked(
        original_question
    )

    # Search using the rewritten question.
    rewritten_results = fetch_context_unranked(
        rewritten_question
    )

    # Merge both result sets and remove duplicates.
    candidates = merge_chunks(
        original_results,
        rewritten_results,
    )

    # Rerank all candidates.
    reranked = rerank(
        original_question,
        candidates,
    )

    # Only keep the best context.
    return reranked[:FINAL_K]


# ---------------------------------------------------------
# STEP 6: BUILD GROUNDED RAG MESSAGES
# ---------------------------------------------------------

def make_rag_messages(
    question,
    history,
    chunks,
):

    context = "\n\n".join(
        (
            f"Extract from {chunk.metadata['source']}:\n"
            f"{chunk.page_content}"
        )
        for chunk in chunks
    )

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    return (
        [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        + history
        + [
            {
                "role": "user",
                "content": question,
            }
        ]
    )


# ---------------------------------------------------------
# STEP 7: GENERATE FINAL ANSWER
# ---------------------------------------------------------

@retry(wait=wait)
def answer_question(
    question: str,
    history: list[dict] = [],
) -> tuple[str, list]:

    # Retrieve the best evidence.
    chunks = fetch_context(
        question,
        history,
    )

    # Build the grounded conversation.
    messages = make_rag_messages(
        question,
        history,
        chunks,
    )

    # Generate the final answer.
    response = completion(
        model=MODEL,
        messages=messages,
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return answer, chunks