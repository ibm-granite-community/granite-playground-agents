class ThinkingPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def thinking_system_prompt() -> str:
        return """
Write down your thoughts and reasoning process as if you're thinking out loud before responding.
Do not provide a response, you will do that later. For not just think through the response.
"""

    @staticmethod
    def responding_system_prompt(thoughts: str) -> str:
        return f"""
Respond in a comprehensive and detailed way.
Use your thoughts to guide and inform your response.

Here are your thoughts:
{thoughts}


"""
