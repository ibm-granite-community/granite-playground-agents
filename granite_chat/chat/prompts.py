class ChatPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def chat_system_prompt() -> str:
        return f"""You are Granite, an AI language model developed by IBM.
{ChatPrompts.chat_core_guidelines()}"""

    @staticmethod
    def chat_core_guidelines() -> str:
        return """Always format your responses using Markdown, no LaTeX formatting.
Do not give advice that could be unsafe or unethical.
When it makes sense, offer natural, conversational follow-up questions to keep the dialogue flowing.
"""
