"""
Robust implementation of the OpenRouter chat model with enhanced error handling.
"""
from typing import Any, Dict, List, Mapping, Optional, ClassVar
import requests
import json
import logging
from urllib.parse import urlparse

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

class ChatOpenRouter(BaseChatModel):
    """OpenRouter chat model with Pydantic 2.x compatibility and robust response handling."""
    
    # Add ClassVar annotation to fix the Pydantic error
    URL: ClassVar[str] = "https://openrouter.ai/api/v1/chat/completions"
    
    # API config
    openrouter_api_key: str
    model: str = "openai/gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    
    # HTTP config
    timeout: Optional[int] = 120
    http_referer: str = "https://your-app-domain.com/"
    x_title: Optional[str] = "Document Summarizer"
    
    # Response format
    streaming: bool = False
    
    # Debugging options
    debug_mode: bool = True
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openrouter"
    
    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Create message dicts for OpenRouter."""
        message_dicts = []
        for message in messages:
            if isinstance(message, HumanMessage):
                message_dicts.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                message_dicts.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                message_dicts.append({"role": "system", "content": message.content})
            elif isinstance(message, ChatMessage):
                message_dicts.append({"role": message.role, "content": message.content})
            else:
                raise ValueError(f"Got unknown message type: {message}")
        return message_dicts
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to OpenRouter's completion endpoint with robust error handling."""
        # Ensure HTTP-Referer has a valid format (must include protocol)
        referer = self.http_referer
        if referer and not (referer.startswith('http://') or referer.startswith('https://')):
            referer = f"https://{referer}"
        
        # For testing, provide a default referer if none exists
        if not referer:
            referer = "https://document-summarizer.example.com/"
            
        # Validate referer with urlparse - this avoids malformed URLs
        parsed_referer = urlparse(referer)
        if not all([parsed_referer.scheme, parsed_referer.netloc]):
            logger.warning(f"Invalid referer URL: {referer}, using default")
            referer = "https://document-summarizer.example.com/"
            
        # Prepare headers with proper formatting
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": referer,  # Required by OpenRouter
            "X-Title": self.x_title or "Document Summarizer",
            "Content-Type": "application/json",
        }
        
        # Log full headers for debugging (except API key)
        debug_headers = headers.copy()
        debug_headers["Authorization"] = "Bearer [REDACTED]"
        logger.info(f"Request headers: {debug_headers}")
        logger.info(f"Request URL: {self.URL}")
        
        message_dicts = self._create_message_dicts(messages)
        
        # Log what we're sending to OpenRouter for debugging
        logger.info(f"Using OpenRouter model: {self.model}")
        
        data = {
            "model": self.model,
            "messages": message_dicts,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        
        if self.top_p is not None:
            data["top_p"] = self.top_p
        
        if stop:
            data["stop"] = stop
        
        request_kwargs = {
            "headers": headers,
            "timeout": self.timeout,
            "json": data,
        }

        try:
            logger.info(f"Sending request to OpenRouter: {self.URL}")
            response = requests.post(self.URL, **request_kwargs)
            
            # Always log the raw response in debug mode
            if self.debug_mode:
                logger.info(f"Raw response status: {response.status_code}")
                logger.info(f"Raw response headers: {response.headers}")
                try:
                    # Try to log the response JSON, but don't fail if it's not valid JSON
                    logger.info(f"Raw response JSON: {json.dumps(response.json(), indent=2)}")
                except:
                    logger.info(f"Raw response text: {response.text}")
            
            # Log the status code and response text if there's an error
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                logger.error(f"Request data: {json.dumps(data)}")
                raise ValueError(f"OpenRouter API returned status code {response.status_code}: {response.text}")
                
            response.raise_for_status()
            
            try:
                parsed_response = response.json()
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {response.text}")
                raise ValueError(f"OpenRouter returned invalid JSON: {response.text}")
            
            logger.info("Successfully received response from OpenRouter")
            
            # Check for expected fields in the response
            if "choices" not in parsed_response:
                logger.error(f"Response missing 'choices' field: {parsed_response}")
                
                # If the response has an error field, use that for the error message
                if "error" in parsed_response:
                    error_msg = parsed_response.get("error", {}).get("message", "Unknown error")
                    raise ValueError(f"OpenRouter response error: {error_msg}")
                
                # If it's missing choices but doesn't have an error, create a synthetic response
                # This allows us to return something rather than failing completely
                logger.warning("Creating synthetic response due to missing 'choices' field")
                
                # Create a synthetic response with a message about the issue
                synthetic_msg = "I encountered an issue with the API response. Please try again or try a different model."
                
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=synthetic_msg),
                            generation_info={"synthetic_response": True}
                        )
                    ],
                    llm_output={"raw_response": parsed_response}
                )
            
            return self._create_chat_result(parsed_response)
            
        except requests.RequestException as e:
            logger.error(f"Request to OpenRouter failed: {str(e)}")
            raise ValueError(f"Error communicating with OpenRouter API: {str(e)}")
    
    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        """Create chat result from response with safer parsing."""
        generations = []
        
        # Safely extract choices, providing fallback for missing data
        choices = response.get("choices", [])
        
        if not choices:
            logger.warning("No choices in response, creating fallback response")
            fallback_msg = "The model did not return any response. Please try again or try a different model."
            
            generations.append(
                ChatGeneration(
                    message=AIMessage(content=fallback_msg),
                    generation_info={"fallback_response": True}
                )
            )
        else:
            for choice in choices:
                # Safely extract message content
                message = choice.get("message", {})
                message_content = message.get("content", "")
                
                if not message_content:
                    logger.warning(f"Empty content in choice: {choice}")
                    message_content = "[No content returned by model]"
                
                generation_info = {
                    "finish_reason": choice.get("finish_reason"),
                    "index": choice.get("index"),
                }
                
                generations.append(
                    ChatGeneration(
                        message=AIMessage(content=message_content),
                        generation_info=generation_info,
                    )
                )
        
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage}
        
        return ChatResult(generations=generations, llm_output=llm_output)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying params."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }