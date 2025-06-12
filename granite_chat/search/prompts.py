from datetime import UTC, datetime

from beeai_framework.backend import Message
from langchain_core.documents import Document

from granite_chat.search.types import SearchResult


class SearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def search_system_prompt(docs: list[Document]) -> str:
        doc_str = "".join(
            f"""Title: {d.metadata["title"]}
Content: {d.page_content}\n\n"""
            for i, d in enumerate(docs)
        )

        print(doc_str)

        return f"""You are Granite, developed by IBM.
You are a helpful assistant tasked with generating a comprehensive, informative, and accurate response.
You are provided to a set of documents that may contain relevant information. You can use these documents to help formulate your response.

Your response should:
- Be clear and comprehensive.
- Use plain language without jargon, or explain terms where necessary.
- Stay aligned with the content and facts of the documents whenever possible.
- Avoid making assumptions.
- Not make any reference or mention of "documents" or "the documents", or of their existence in any way.

If a document contains irrelevant information or is incomprehensible, ignore it.
If the information needed is not available, inform the user that the question cannot be answered based on the available data.
Assume the current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Here are the documents:
{doc_str}
"""  # noqa: E501

    @staticmethod
    def generate_search_queries_prompt(messages: list[Message], max_queries: int = 3) -> str:
        conversation: list[str] = []

        for m in messages:
            if m.role == "user":
                conversation.append("User: " + m.text)
            elif m.role == "assistant":
                conversation.append("Assistant: " + m.text)

        conversation_str = "\n".join(conversation)
        dynamic_example = ", ".join([f'"query {i + 1}"' for i in range(max_queries)])

        return f"""
Assume the current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Given the following conversation between a user and an assistant, analyze the user's last message and generate {max_queries} search engine queries that reflect the user's intent.
The search queries should be clear, concise, and suitable for use in a web search. Include variations to cover possible angles or phrasings.

Tips:
- Do not assume or introduce information that is not directly mentioned in the conversation.
- Use the date to augment queries if the user is asking of recent or latest information.

Here is an example:
Conversation:
User: I'm getting a weird error when deploying my React app to Vercel.
Assistant: What does the error say?
User: It says “Module not found: Can't resolve './App'”.

Search Queries:
["React Vercel deployment error Module not found: Can't resolve './App'"]

Now here is your task:

Conversation:
{conversation_str}

Generate exactly {max_queries} search queries that reflect the intent of the user's last message.
You must respond with a list of strings in the following format: [{dynamic_example}].
The response should contain ONLY the list.
"""  # noqa: E501

    @staticmethod
    def generate_standalone_query(messages: list[Message]) -> str:
        conversation: list[str] = []

        for m in messages:
            if m.role == "user":
                conversation.append("User: " + m.text)
            elif m.role == "assistant":
                conversation.append("Assistant: " + m.text)

        conversation_str = "\n".join(conversation)

        return f"""
Given the following conversation between a user and an assistant, analyze the user's last message and generate a single, standalone query that clearly and concisely reflects the user's intent, preserving the necessary context so it can be understood independently.
The query will be used to look up information.

Here is an example:
Conversation:
User: I'm trying to extract keywords from text.
Assistant: What programming language are you using?
User: Python.
Assistant: You can use libraries like spaCy or sklearn.
User: I want the output in a JSON format with relevance scores.

Standalone Query:
How to extract keywords from text using Python and output the results in a JSON format with relevance scores.

Now here is your task:

Conversation:
{conversation_str}

Generate a standalone query that clearly and concisely reflects the user's intent. Output only the standalone query.
"""  # noqa: E501

    @staticmethod
    def filter_search_result_prompt(query: str, search_result: SearchResult) -> str:
        return f"""
Given a user query and a search result, determine whether the web page linked in the search result is likely to provide information relevant to the user's query.

Here is the user's query: {query}

Here is the search result:
- URL: {search_result.url}
- Title: {search_result.title}
- A snippet from the page: {search_result.body}

Respond with one of the following labels only:
- RELEVANT: if the page likely contains meaningful information answering or directly related to the query.
- IRRELEVANT: if the page is unlikely to contain information useful for the query.
"""  # noqa: E501

    @staticmethod
    def filter_doc_prompt(intent: str, doc: Document) -> str:
        url = doc.metadata["url"]
        title = doc.metadata["title"]
        content = doc.page_content

        return f"""
You are a classifier evaluating the relevance of a given document to a statement of user intent.

Given a statement of user intent and a document, decide whether the document contains information that is relevant to the user intent.
Use the url, title and content of the document to make a decision, but focus primarily on content.
Reject documents if they are too short, incoherent or not interesting.

Example:
Statement of user intent: Learn how to create a basic website using HTML and CSS.

Document:
Title: Getting Started with HTML and CSS
Content: This tutorial provides a step-by-step guide to creating a simple website using HTML and CSS...

Relevance classification:
RELEVANT

Now here is the statement of user intent: {intent}

Here is the document:
URL: {url}
Title: {title}
Content: {content}

Respond with one of the following labels only:
- RELEVANT: if the document is coherent, contains relevant information answering or directly related to the query.
- IRRELEVANT: if the document is incoherent or does not contain information useful for the query.
"""  # noqa: E501
