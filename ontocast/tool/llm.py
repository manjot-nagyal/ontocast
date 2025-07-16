"""Language Model (LLM) integration tool for OntoCast.

This module provides integration with various language models through LangChain,
supporting OpenAI, Google Gemini, and Ollama providers. It enables text generation and
structured data extraction capabilities.
"""

import asyncio
from typing import Any, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ontocast.dspy_output_parser import DspyOutputParser

from .onto import Tool

T = TypeVar("T", bound=BaseModel)


class LLMTool(Tool):
    """Tool for interacting with language models.

    This class provides a unified interface for working with different language model
    providers (OpenAI, Google Gemini, Ollama) through LangChain. It supports both synchronous and
    asynchronous operations.

    Attributes:
        provider: The LLM provider to use (default: "openai").
        model: The specific model to use (default: "gpt-4o-mini").
        api_key: Optional API key for the provider.
        base_url: Optional base URL for the provider.
        temperature: Temperature parameter for generation (default: 0.1).
    """

    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1

    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the LLM tool.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._llm = None

    @classmethod
    def create(cls, **kwargs):
        """Create a new LLM tool instance synchronously.

        Args:
            **kwargs: Keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        return asyncio.run(cls.acreate(**kwargs))

    @classmethod
    async def acreate(cls, **kwargs):
        """Create a new LLM tool instance asynchronously.

        Args:
            **kwargs: Keyword arguments for initialization.

        Returns:
            LLMTool: A new instance of the LLM tool.
        """
        self = cls.__new__(cls)
        self.__init__(**kwargs)
        await self.setup()
        return self

    async def setup(self):
        """Set up the language model based on the configured provider.

        Raises:
            ValueError: If the provider is not supported.
        """
        if self.provider == "openai":
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
            )
        elif self.provider == "ollama":
            self._llm = ChatOllama(
                model=self.model, base_url=self.base_url, temperature=self.temperature
            )
        elif self.provider == "google":
            self._llm = ChatGoogleGenerativeAI(
                model=self.model, temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call the language model directly.

        Args:
            *args: Positional arguments passed to the LLM.
            **kwds: Keyword arguments passed to the LLM.

        Returns:
            Any: The LLM's response.
        """
        return self.llm.invoke(*args, **kwds)

    @property
    def llm(self) -> BaseChatModel:
        """Get the underlying language model instance.

        Returns:
            BaseChatModel: The configured language model.

        Raises:
            RuntimeError: If the LLM has not been properly initialized.
        """
        if self._llm is None:
            raise RuntimeError(
                "LLM resource not properly initialized. Call setup() first."
            )
        return self._llm

    async def complete(self, prompt: str, **kwargs) -> Any:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt for generation.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            Any: The generated completion.
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

    async def extract(self, prompt: str, output_schema: Type[T], **kwargs) -> T:
        """Extract structured data from the prompt according to a schema.

        Args:
            prompt: The input prompt for extraction.
            output_schema: The Pydantic model class defining the output structure.
            **kwargs: Additional keyword arguments for extraction.

        Returns:
            T: The extracted data conforming to the output schema.
        """
        parser = DspyOutputParser(pydantic_object=output_schema)
        format_instructions = parser.get_format_instructions()

        full_prompt = f"{prompt}\n\n{format_instructions}"
        response = await self.llm.ainvoke(full_prompt)

        return parser.parse(response.content)
