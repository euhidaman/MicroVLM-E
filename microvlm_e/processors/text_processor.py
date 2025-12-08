"""
Text processors for MicroVLM-E.
"""

import re
from typing import Optional

from microvlm_e.common.registry import registry


class BaseTextProcessor:
    """Base class for text processors."""

    def __init__(self, max_length: Optional[int] = None):
        self.max_length = max_length

    def __call__(self, text: str) -> str:
        return text

    @classmethod
    def from_config(cls, cfg):
        return cls(
            max_length=cfg.get("max_length", None),
        )


@registry.register_processor("blip_caption")
class CaptionProcessor(BaseTextProcessor):
    """Text processor for captions."""

    def __init__(
        self,
        max_length: Optional[int] = None,
        min_length: int = 1,
    ):
        super().__init__(max_length)
        self.min_length = min_length

    def __call__(self, text: str) -> str:
        # Clean up text
        text = self.clean_text(text)

        # Truncate if needed
        if self.max_length is not None:
            words = text.split()
            if len(words) > self.max_length:
                text = " ".join(words[:self.max_length])

        return text

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @classmethod
    def from_config(cls, cfg):
        return cls(
            max_length=cfg.get("max_length", None),
            min_length=cfg.get("min_length", 1),
        )


@registry.register_processor("instruction")
class InstructionProcessor(BaseTextProcessor):
    """Text processor for instruction tuning."""

    def __init__(
        self,
        max_length: Optional[int] = None,
        prompt_template: str = "",
    ):
        super().__init__(max_length)
        self.prompt_template = prompt_template

    def __call__(self, text: str) -> str:
        # Apply prompt template
        if self.prompt_template:
            text = self.prompt_template.format(text)

        # Truncate if needed
        if self.max_length is not None:
            words = text.split()
            if len(words) > self.max_length:
                text = " ".join(words[:self.max_length])

        return text

    @classmethod
    def from_config(cls, cfg):
        return cls(
            max_length=cfg.get("max_length", None),
            prompt_template=cfg.get("prompt_template", ""),
        )


@registry.register_processor("qa")
class QAProcessor(BaseTextProcessor):
    """Text processor for question answering."""

    def __init__(
        self,
        max_question_length: Optional[int] = None,
        max_answer_length: Optional[int] = None,
    ):
        super().__init__(max_question_length)
        self.max_answer_length = max_answer_length

    def process_question(self, question: str) -> str:
        """Process a question."""
        question = question.strip()

        # Ensure question mark
        if not question.endswith("?"):
            question += "?"

        # Truncate if needed
        if self.max_length is not None:
            words = question.split()
            if len(words) > self.max_length:
                question = " ".join(words[:self.max_length])

        return question

    def process_answer(self, answer: str) -> str:
        """Process an answer."""
        answer = answer.strip()

        # Truncate if needed
        if self.max_answer_length is not None:
            words = answer.split()
            if len(words) > self.max_answer_length:
                answer = " ".join(words[:self.max_answer_length])

        return answer

    def __call__(self, text: str) -> str:
        return self.process_question(text)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            max_question_length=cfg.get("max_question_length", None),
            max_answer_length=cfg.get("max_answer_length", None),
        )

