import os
import time
import textwrap
import json
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
import argparse

# LangChain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

# Model-specific imports
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Import OpenRouter - use a direct import for our custom implementation
# No try/except to avoid import errors - we know this file exists because we created it
import sys
import os.path

# Add the current directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import our custom OpenRouter implementation
from openrouter_patch import ChatOpenRouter

class EnhancedSummarizer:
    """Enhanced document summarization using multiple LLM providers."""
    
    # Model provider constants
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"  # Add new provider constant
    
    # Default prompts
    DEFAULT_PROMPT = """
    You are a professional summarizer with expertise in creating clear, concise, and comprehensive summaries.
    
    Please summarize the following text:
    
    {text}
    
    Focus on the main points, key arguments, and important details while maintaining the original meaning.
    Structure your summary with a logical flow. Include an introduction, body, and conclusion.
    """
    
    def __init__(
    self,
    provider: str = OPENAI,
    model_name: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    chain_type: str = "refine",
    http_referer: str = "https://document-summarizer.example.com"  # Add this parameter
    ):
        """
        Initialize the summarizer with customizable parameters.
        
        Args:
            provider: Model provider ("openai", "anthropic", "google", or "openrouter")
            model_name: Name of the model to use (provider-specific)
            temperature: Temperature parameter for generation (0.0-1.0)
            max_tokens: Maximum tokens in the response
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            chain_type: LangChain summarization chain type ('map_reduce' or 'refine')
            http_referer: HTTP referer header for OpenRouter API calls
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chain_type = chain_type
        self.http_referer = http_referer
        
        # Initialize the LLM based on provider
        if self.provider == self.OPENAI:
            # Get API key from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables (OPENAI_API_KEY)")
            
            # Set default model if not specified
            if model_name is None:
                model_name = "gpt-4"
                
            self.model_name = model_name
            self.llm = ChatOpenAI(
                temperature=temperature,
                model_name=model_name,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
            
        elif self.provider == self.ANTHROPIC:
            # Get API key from environment
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found in environment variables (ANTHROPIC_API_KEY)")
            
            # Set default model if not specified
            if model_name is None:
                model_name = "claude-3-sonnet-20240229"
                
            self.model_name = model_name
            self.llm = ChatAnthropic(
                temperature=temperature,
                model_name=model_name,
                max_tokens=max_tokens,
                anthropic_api_key=api_key
            )
            
        elif self.provider == self.GOOGLE:
            # Get API key from environment
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not found in environment variables (GOOGLE_API_KEY)")
            
            # Set default model if not specified
            if model_name is None:
                model_name = "gemini-1.5-pro"
                
            self.model_name = model_name
            self.llm = ChatGoogleGenerativeAI(
                temperature=temperature,
                model=model_name,
                max_output_tokens=max_tokens,
                google_api_key=api_key
            )
            
        elif self.provider == self.OPENROUTER:
            # Get API key from environment
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key not found in environment variables (OPENROUTER_API_KEY)")
            
            # Set default model if not specified
            if model_name is None:
                model_name = "openai/gpt-3.5-turbo"
                
            self.model_name = model_name
            self.llm = ChatOpenRouter(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                openrouter_api_key=api_key,
                http_referer=self.http_referer  # Pass the HTTP referer
            )
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'anthropic', 'google', or 'openrouter'.")
    
    @staticmethod
    def get_available_models() -> Dict[str, List[Dict[str, Any]]]:
        """Return a dictionary of available models by provider with their details."""
        return {
            "openai": [
                {"name": "gpt-4", "description": "Most powerful OpenAI model", "cost": "High"},
                {"name": "gpt-4-turbo", "description": "Faster version of GPT-4", "cost": "Medium-High"},
                {"name": "gpt-3.5-turbo", "description": "Fast and cost-effective", "cost": "Low"}
            ],
            "anthropic": [
                {"name": "claude-3-opus-20240229", "description": "Most powerful Claude model", "cost": "High"},
                {"name": "claude-3-sonnet-20240229", "description": "Balanced performance", "cost": "Medium"},
                {"name": "claude-3-haiku-20240307", "description": "Fast and efficient", "cost": "Low"}
            ],
            "google": [
                {"name": "gemini-1.5-pro", "description": "Most powerful Google model", "cost": "Medium"},
                {"name": "gemini-1.0-pro", "description": "Standard performance model", "cost": "Low"},
            ],
            "openrouter": [
                {"name": "openai/gpt-4-turbo", "description": "OpenAI GPT-4 via OpenRouter", "cost": "Medium-High"},
                {"name": "anthropic/claude-3-opus", "description": "Anthropic Claude 3 Opus via OpenRouter", "cost": "High"},
                {"name": "anthropic/claude-3-sonnet", "description": "Anthropic Claude 3 Sonnet via OpenRouter", "cost": "Medium"},
                {"name": "google/gemini-pro", "description": "Google Gemini Pro via OpenRouter", "cost": "Medium"},
                {"name": "mistralai/mistral-large", "description": "Mistral Large via OpenRouter", "cost": "Medium"},
                {"name": "meta-llama/llama-3-70b-instruct", "description": "Meta Llama 3 70B via OpenRouter", "cost": "Medium"},
                {"name": "cohere/command-r", "description": "Cohere Command-R via OpenRouter", "cost": "Low"},
                # Add more OpenRouter models as needed
            ]
        }
    
    @staticmethod
    def get_openrouter_models() -> List[Dict[str, Any]]:
        """Fetch available models from OpenRouter API"""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return []
            
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://document-summarizer.example.com"  # Required by OpenRouter
            }
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
            
            if response.status_code == 200:
                models_data = response.json()
                # Format the data to match our structure
                models = []
                for model in models_data["data"]:
                    models.append({
                        "name": model["id"],
                        "description": model.get("description", "No description available"),
                        "cost": f"{model.get('pricing', {}).get('prompt', '?')}/{model.get('pricing', {}).get('completion', '?')} per tok",
                        "context_length": model.get("context_length", "Unknown")
                    })
                return models
            else:
                print(f"Error from OpenRouter API: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            print(f"Error fetching OpenRouter models: {e}")
            return []
    
    def read_document(self, file_path: str) -> str:
        """Read document from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text
        except Exception as e:
            raise ValueError(f"Error reading file '{file_path}': {e}")
    
    def chunk_document(self, text: str) -> List[Document]:
        """Split document into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(text)
        return [Document(page_content=t) for t in texts]
    
    def prepare_chain(self, custom_prompt: Optional[str] = None) -> Any:
        """Prepare the summarization chain with optional custom prompt."""
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = self.DEFAULT_PROMPT
            
        if self.chain_type == "refine":
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["text"]
            )
            refine_prompt = PromptTemplate(
                template="""
                You are refining an existing summary with new information.
                
                Existing summary: {existing_answer}
                
                New text to consider: {text}
                
                Please refine the summary by incorporating relevant information from the new text.
                Maintain a cohesive, well-structured summary that flows naturally.
                """,
                input_variables=["existing_answer", "text"]
            )
            chain = load_summarize_chain(
                self.llm,
                chain_type="refine",
                question_prompt=prompt,
                refine_prompt=refine_prompt,
                return_intermediate_steps=True
            )
        else:  # map_reduce
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["text"]
            )
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=prompt,
                combine_prompt=prompt,
                return_intermediate_steps=True
            )
        
        return chain
    
    def summarize(
        self, 
        input_source: str, 
        is_file_path: bool = True,
        custom_prompt: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Summarize a document with detailed metrics.
        
        Args:
            input_source: Path to document file OR text content
            is_file_path: If True, input_source is a file path; if False, it's text content
            custom_prompt: Optional custom prompt template
            verbose: If True, print detailed information during processing
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        
        if verbose and is_file_path:
            print(f"Reading document: {input_source}")
        
        # Get text content
        if is_file_path:
            text = self.read_document(input_source)
            doc_path = input_source
        else:
            text = input_source
            doc_path = "text_input"
            
        doc_length = len(text)
        
        if verbose:
            print(f"Document length: {doc_length} characters")
            print(f"Chunking document (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        # Chunk document
        docs = self.chunk_document(text)
        num_chunks = len(docs)
        
        if verbose:
            print(f"Document split into {num_chunks} chunks")
            print(f"Preparing {self.chain_type} chain with model: {self.model_name}")
        
        # Prepare chain
        chain = self.prepare_chain(custom_prompt)
        
        if verbose:
            print("Running summarization...")
        
        # Track token usage and run chain
        token_usage = {}
        if self.provider == self.OPENAI:
            # OpenAI provides detailed token usage metrics
            with get_openai_callback() as cb:
                result = chain({"input_documents": docs})
                token_usage = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "total_cost_usd": cb.total_cost
                }
        else:
            # For other providers, we don't get detailed token metrics through LangChain
            result = chain({"input_documents": docs})
            # Estimate tokens based on a rough 4 chars per token
            estimated_tokens = len(text) // 4
            token_usage = {
                "estimated_total_tokens": estimated_tokens,
                "total_cost_usd": "Not available"
            }
        
        # Collect summary 
        summary = result.get("output_text", "")
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Prepare response
        response = {
            "summary": summary,
            "metadata": {
                "document_path": doc_path,
                "document_length": doc_length,
                "compression_ratio": round(len(summary) / doc_length, 3) if doc_length > 0 else 0,
                "num_chunks": num_chunks,
                "provider": self.provider,
                "model": self.model_name,
                "chain_type": self.chain_type,
                "token_usage": token_usage,
                "processing_time_seconds": processing_time
            }
        }
        
        if verbose:
            print(f"Summarization complete - {processing_time} seconds")
            print(f"Compression ratio: {response['metadata']['compression_ratio']}")
            if 'total_tokens' in token_usage:
                print(f"Token usage: {token_usage['total_tokens']} tokens")
        
        return response
    
    def save_results(self, results: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
        """Save summary and metadata to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base filename from input document
        base_name = os.path.basename(results["metadata"]["document_path"])
        base_name = os.path.splitext(base_name)[0]
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(results["summary"])
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results["metadata"], f, indent=2)
            
        return summary_path, metadata_path


def main():
    """Command-line interface for the enhanced summarizer."""
    parser = argparse.ArgumentParser(
        description="Enhanced document summarization using multiple LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
            # Basic usage with OpenAI
            python enhanced_summarizer.py document.pdf --output summaries --provider openai
            
            # Using Claude
            python enhanced_summarizer.py document.pdf --output summaries --provider anthropic --model claude-3-sonnet-20240229
            
            # Using Google Gemini
            python enhanced_summarizer.py document.pdf --output summaries --provider google --model gemini-1.5-pro
            
            # Using OpenRouter
            python enhanced_summarizer.py document.pdf --output summaries --provider openrouter --model anthropic/claude-3-sonnet
        """)
    )
    
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--output", default="./summaries", help="Output directory for results")
    parser.add_argument("--provider", default="openai", 
                        choices=["openai", "anthropic", "google", "openrouter"], 
                        help="LLM provider to use")
    parser.add_argument("--model", help="Model name (provider-specific)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature parameter (0.0-1.0)")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum tokens in the response")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--chain_type", choices=["map_reduce", "refine"], default="refine", 
                        help="Summarization chain type")
    parser.add_argument("--prompt", help="Custom prompt template (include {text} placeholder)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed processing information")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--list-openrouter-models", action="store_true", 
                        help="Query OpenRouter API for all available models and exit")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        models = EnhancedSummarizer.get_available_models()
        print("\nAvailable models by provider:\n")
        for provider, provider_models in models.items():
            print(f"=== {provider.upper()} ===")
            for model in provider_models:
                print(f"  • {model['name']}")
                print(f"    Description: {model['description']}")
                print(f"    Cost: {model['cost']}")
                print()
        return 0
    
    # List OpenRouter models if requested
    if args.list_openrouter_models:
        print("\nFetching available models from OpenRouter API...\n")
        models = EnhancedSummarizer.get_openrouter_models()
        if not models:
            print("Failed to fetch models or no OPENROUTER_API_KEY environment variable found.")
            return 1
            
        print(f"Found {len(models)} models on OpenRouter:\n")
        for model in models:
            print(f"  • {model['name']}")
            print(f"    Description: {model['description']}")
            print(f"    Context Length: {model['context_length']}")
            print(f"    Cost: {model['cost']}")
            print()
        return 0
    
    try:
        # Initialize summarizer
    # Initialize summarizer
        summarizer = EnhancedSummarizer(
            provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chain_type=args.chain_type,
            http_referer="https://document-summarizer.example.com"
        )
        # Summarize document
        results = summarizer.summarize(
            args.file_path,
            is_file_path=True,
            custom_prompt=args.prompt,
            verbose=args.verbose
        )
        
        # Save results
        summary_path, metadata_path = summarizer.save_results(results, args.output)
        
        print(f"\nSummary saved to: {summary_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print("=" * 50)
        print(results["summary"])
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())