class ThinkingPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def thinking_system_prompt() -> str:
        return """
Write down your thoughts and reasoning process as if you're thinking out loud before responding. Think step by step.
Do NOT provide a response, you will do that later.
"""

    @staticmethod
    def responding_system_prompt(thoughts: str) -> str:
        return f"""
You are a helpful assistant tasked with generating a comprehensive, informative, accurate, and easy-to-read response.
IMPORTANT: Use your previous thoughts to guide and inform your response.

Here are your thoughts:
{thoughts}

If you do not incorporate your thoughts, the response will be considered incomplete.
"""
