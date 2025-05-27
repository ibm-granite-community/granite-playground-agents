from datetime import UTC, datetime

from beeai_framework.backend import Message


class SearchPrompts:

    def __init__(self) -> None:
        pass

    @staticmethod
    def search_system_prompt() -> str:
        return f"""You are Granite, developed by IBM.

You are a helpful assistant tasked with generating an informative, accurate, and easy-to-read response based on the information contained in the following documents. 
Your response should:
- Be clear, concise, and comprehensive, suitable for a general audience.
- Use plain language without jargon, or explain terms where necessary.
- Stay aligned with the content and facts of the source documents whenever possible.
- Avoid making assumptions or adding information that is not supported by the documents.
- Not reference or mention the documents or their existence.

If a you believe a document contains irrelevant information or is incomprehensible, ignore it.
If the information needed is not available in the documents, inform the user that the question cannot be answered based on the available data.
Assume the current date is {datetime.now(UTC).strftime('%B %d, %Y')} if required.
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
        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_queries)])

        return f"""
Given the following conversation between a user and an assistant, analyze the user's last message and generate {max_queries} search engine queries that reflect the user's intent.
The search queries should be clear, concise, and suitable for use in a web search. Include variations to cover possible angles or phrasings.
Assume the current date is {datetime.now(UTC).strftime('%B %d, %Y')} if required.

Here is an example:
Conversation:
User: I’m getting a weird error when deploying my React app to Vercel.
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
