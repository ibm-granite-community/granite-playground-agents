import inspect

from granite_core.chat.prompts import ChatPrompts


def test_chat_prompts() -> None:
    # get all static methods of ChatPrompts
    methods = [func for func in dir(ChatPrompts) if isinstance(inspect.getattr_static(ChatPrompts, func), staticmethod)]

    for method in methods:
        func = getattr(ChatPrompts, method)
        s = func()
        assert isinstance(s, str)
        assert len(s) > 0
