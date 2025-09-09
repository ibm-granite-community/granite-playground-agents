import json
from datetime import UTC, datetime

from beeai_framework.backend import Message
from langchain_core.documents import Document

from granite_chat import get_logger
from granite_chat.chat.prompts import ChatPrompts
from granite_chat.search.types import SearchResult

logger = get_logger(__name__)


class SearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def search_system_prompt(docs: list[Document]) -> str:
        json_docs = [
            {"doc_id": str(i), "title": d.metadata["title"], "content": d.page_content} for i, d in enumerate(docs)
        ]

        doc_str = json.dumps(json_docs, indent=4)

        logger.debug(doc_str)

        return f"""You are Granite, developed by IBM.
Provide a comprehensive, informative, and accurate response to the user.

You are provided with a set of documents that contain relevant information.
- Use these documents to help formulate your response.
- Your response should stay aligned with the content and facts of the documents when possible.
- If the information needed is not available, inform the user that the question cannot be answered based on the available data.
- Not all documents will be relevant, ignore irrelevant or low quality documents.
- Draw on multiple documents to create a more diverse and informed response.

<documents>
{doc_str}
</documents>

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your response.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
You have access to realtime data, you do not have a knowledge cutoff.
{ChatPrompts.chat_core_guidelines()}"""  # noqa: E501

    @staticmethod
    def generate_search_queries_prompt(messages: list[Message], max_queries: int = 3) -> str:
        conversation: list[str] = []

        for m in messages:
            if m.role == "user":
                conversation.append("User: " + m.text)
            elif m.role == "assistant":
                conversation.append("Assistant: " + m.text)

        conversation_str = "\n".join(conversation)

        return f"""
Assume the current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Given the following conversation between a user and an assistant, analyze the user's last message and generate {max_queries} search engine queries that reflect the user's intent.
The search queries should be clear, concise, and suitable for use in a web search. Include variations to cover possible angles or phrasings. Include all important keywords.
If the user intent is mult-faceted then incorporate each facet into a query.

Tips:
- Do not assume or introduce information that is not directly mentioned in the conversation.
- Use the date to augment queries if the user is asking of recent or latest information but be very precise.

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

Generate {max_queries} search queries that satisfy the intent of the user's last message.
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
The query will be used to look up information. Include relevant keywords.

Here is an example:
Conversation:
User: I'm trying to extract keywords from text.
Assistant: What programming language are you using?
User: Python.
Assistant: You can use libraries like spaCy or sklearn.
User: I want the output in a JSON format with relevance scores.

Standalone Query:
Extract keywords from text using Python and output the results in a JSON format with relevance scores.

Now here is your task:

Conversation:
{conversation_str}

Generate a standalone query that clearly and concisely reflects the user's intent.
"""  # noqa: E501

    @staticmethod
    def filter_search_result_prompt(query: str, search_result: SearchResult) -> str:
        return f"""
You are given a topic and a search result (URL, title and snippet of page content ). Your task is to determine whether the web page linked by the search result is likely to contain useful information that is directly relevant to the topic.

A relevant (True) search result has the following properties:
- The url, title or page snippet indicate that the linked page contains content that is relevant to the topic.
- It is likely that the page is specific, accurate, and up-to-date.
- It is likely that the page contributes an interesting angle or theme.

If a search result looks like it may contain or promote violent, hateful or pornographic material it should be automatically marked as irrelevant.

Here is the topic: {query}

Here is the search result:
- URL: {search_result.url}
- Title: {search_result.title}
- A snippet from the page: {search_result.body}

Return True if the result is likely relevant to the topic; otherwise, return False.

Output format:
{{
    "is_relevant": True|False,
}}
"""  # noqa: E501
