"""Loading the prompt text kept in `prompts/*.md`.

Prompts live in files, not string literals in the code, so they can be read and
edited as prose. Placeholders are `${name}` (`string.Template`, not `str.format`)
so a prompt can contain literal braces — JSON Schema, code samples — without
escaping. Only the template is scanned for placeholders, so a `$` inside a
substituted value is left alone.
"""
from string import Template

from .paths import PROMPTS_DIR


def render(name: str, **values: object) -> str:
    """Fill in `prompts/<name>.md` and return the finished prompt.

    Args:
        name: Prompt file stem — "judge" reads `prompts/judge.md`.
        **values: One keyword argument per `${placeholder}` in the file.

    Returns:
        The rendered prompt, stripped of surrounding whitespace.

    Raises:
        KeyError: The file has a placeholder the caller gave no value for. This
            is deliberately fatal: a silently empty slot would send a malformed
            prompt to a model and surface much later as a bad result.
        FileNotFoundError: No prompt file by that name.
    """
    template = Template((PROMPTS_DIR / f"{name}.md").read_text())
    return template.substitute(**values).strip()
