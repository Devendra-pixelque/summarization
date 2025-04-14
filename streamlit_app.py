import streamlit as st
import os
import tempfile
import time
import base64
import json
import logging
import requests
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import functools
import pandas as pd
# Import the EnhancedSummarizer class and DocumentProcessor
from enhanced_summarizer import EnhancedSummarizer
from document_processor import DocumentProcessor

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Default to dark theme
    
# Initialize session state for caching
if 'last_openrouter_fetch' not in st.session_state:
    st.session_state.last_openrouter_fetch = 0

# Initialize session state for search
if 'show_search_results' not in st.session_state:
    st.session_state.show_search_results = False

# Page configuration
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme dictionary
THEMES = {
    'dark': {
        'bg': '#121212',
        'sidebar_bg': '#1E1E1E',
        'text': '#FFFFFF',
        'accent': '#BB86FC',
        'secondary': '#03DAC6',
        'warning': '#FFB74D',
        'error': '#CF6679',
        'success': '#81C784',
        'provider_colors': {
            'openai': '#10A37F',
            'anthropic': '#BE2EDD',
            'google': '#FF9800',
            'openrouter': '#2196F3',
        }
    },
    'light': {
        'bg': '#FFFFFF',
        'sidebar_bg': '#F5F5F5',
        'text': '#121212',
        'accent': '#6200EE',
        'secondary': '#03DAC6',
        'warning': '#FFA000',
        'error': '#B00020',
        'success': '#4CAF50',
        'provider_colors': {
            'openai': '#4CAF50',
            'anthropic': '#9C27B0',
            'google': '#FF9800',
            'openrouter': '#2196F3',
        }
    },
    'high_contrast': {
        'bg': '#000000',
        'sidebar_bg': '#0D0D0D',
        'text': '#FFFFFF',
        'accent': '#FFFF00',
        'secondary': '#00FFFF',
        'warning': '#FF8000',
        'error': '#FF0000',
        'success': '#00FF00',
        'provider_colors': {
            'openai': '#00FF00',
            'anthropic': '#FF00FF',
            'google': '#FFFF00',
            'openrouter': '#00FFFF',
        }
    }
}

# Get current theme colors
theme_colors = THEMES[st.session_state.theme]

# Custom CSS with dynamic theme colors
def get_css():
    return f"""
    <style>
        .stApp {{
            background-color: {theme_colors['bg']};
            color: {theme_colors['text']};
        }}
        
        .stSidebar {{
            background-color: {theme_colors['sidebar_bg']};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {theme_colors['text']} !important;
        }}
        
        .stButton>button {{
            background-color: {theme_colors['accent']};
            color: #FFFFFF;
        }}
        
        .main-header {{
            font-size: 2.5rem;
            color: {theme_colors['accent']};
            margin-bottom: 1rem;
            font-weight: bold;
        }}
        
        .sub-header {{
            font-size: 1.5rem;
            color: {theme_colors['text']};
            margin-bottom: 1rem;
            opacity: 0.8;
        }}
        
        .info-box {{
            background-color: {theme_colors['sidebar_bg']};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid {theme_colors['accent']};
            margin-bottom: 1rem;
            color: {theme_colors['text']};
        }}
        
        .warning-box {{
            background-color: {theme_colors['sidebar_bg']};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid {theme_colors['warning']};
            margin-bottom: 1rem;
            color: {theme_colors['text']};
        }}
        
        .error-box {{
            background-color: {theme_colors['sidebar_bg']};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid {theme_colors['error']};
            margin-bottom: 1rem;
            color: {theme_colors['text']};
        }}
        
        .result-box {{
            background-color: {theme_colors['sidebar_bg']};
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: {theme_colors['text']};
        }}
        
        .provider-openai {{
            border-left: 0.5rem solid {theme_colors['provider_colors']['openai']};
        }}
        
        .provider-anthropic {{
            border-left: 0.5rem solid {theme_colors['provider_colors']['anthropic']};
        }}
        
        .provider-google {{
            border-left: 0.5rem solid {theme_colors['provider_colors']['google']};
        }}
        
        .provider-openrouter {{
            border-left: 0.5rem solid {theme_colors['provider_colors']['openrouter']};
        }}
        
        .footer {{
            margin-top: 3rem;
            text-align: center;
            color: {theme_colors['text']};
            opacity: 0.6;
            font-size: 0.8rem;
        }}
        
        /* Enhanced styling for select boxes */
        .stSelectbox label, .stSlider label {{
            color: {theme_colors['text']} !important;
        }}
        
        /* Model card styling */
        .model-card {{
            background-color: {theme_colors['sidebar_bg']};
            padding: 0.7rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .model-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        .model-card.selected {{
            border: 2px solid {theme_colors['accent']};
            background-color: rgba(187, 134, 252, 0.1);
        }}
        
        .model-card h4 {{
            margin: 0;
            color: {theme_colors['accent']};
            font-size: 1rem;
        }}
        
        .model-card p {{
            margin: 0.5rem 0;
            font-size: 0.9rem;
            color: {theme_colors['text']};
            opacity: 0.8;
        }}
        
        /* Search styling */
        .search-box {{
            margin-bottom: 1rem;
        }}
        
        /* Improve visibility of streamlit components */
        .stTextInput>div>div>input {{
            background-color: {theme_colors['sidebar_bg']};
            color: {theme_colors['text']};
        }}
        
        .stTextArea>div>div>textarea {{
            background-color: {theme_colors['sidebar_bg']};
            color: {theme_colors['text']};
        }}
        
        div[data-baseweb="select"] {{
            background-color: {theme_colors['sidebar_bg']};
        }}
        
        div[data-baseweb="select"] > div {{
            background-color: {theme_colors['sidebar_bg']};
            color: {theme_colors['text']};
        }}
        
        div[data-testid="stFileUploader"] {{
            background-color: {theme_colors['sidebar_bg']};
            border: 1px dashed rgba(255, 255, 255, 0.3);
        }}
        
        div[data-testid="stFileUploader"] > div > span > button {{
            background-color: {theme_colors['accent']};
            color: white;
        }}
        
        /* Model list styling */
        .model-list {{
            max-height: 300px;
            overflow-y: auto;
            margin-top: 1rem;
            padding: 0.5rem;
            background-color: {theme_colors['sidebar_bg']};
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        /* Hide select model buttons */
        .hide-button {{
            display: none;
        }}
    </style>
    """

# Apply CSS
st.markdown(get_css(), unsafe_allow_html=True)

def create_download_link(content, filename, text):
    """Generate a download link for text content"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="color: {theme_colors["accent"]}; text-decoration: none; border: 1px solid {theme_colors["accent"]}; padding: 0.5rem 1rem; border-radius: 0.3rem;">{text}</a>'
    return href

# Cache the API key check to reduce lag
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_keys():
    """Check if API keys are set in environment variables."""
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY")
    }
    
    available_providers = []
    missing_providers = []
    
    for provider, key in api_keys.items():
        if key:
            provider_name = provider.split("_")[0].lower()
            available_providers.append(provider_name)
        else:
            provider_name = provider.split("_")[0].lower()
            missing_providers.append(provider_name)
    
    return available_providers, missing_providers

# Cache OpenRouter models to reduce lag
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_openrouter_models():
    """Get available OpenRouter models if API key is set."""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            models = EnhancedSummarizer.get_openrouter_models()
            st.session_state.last_openrouter_fetch = time.time()
            return models
        except Exception as e:
            logger.error(f"Error fetching OpenRouter models: {e}")
            return []
    return []

def process_file(uploaded_file, provider, model, temperature, max_tokens, chunk_size, chunk_overlap, chain_type, custom_prompt):
    """Process uploaded file with the selected model."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process document
        doc_processor = DocumentProcessor(verbose=True)
        text, metadata = doc_processor.process_document(tmp_file_path)
        
        st.write(f"**Document Metadata:**")
        st.json(metadata)
        
        # Initialize summarizer
        summarizer = EnhancedSummarizer(
            provider=provider,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chain_type=chain_type,
            http_referer="https://your-app-domain.com/"  # Fix for OpenRouter
        )
        
        # Generate summary
        start_time = time.time()
        result = summarizer.summarize(
            text,
            is_file_path=False,
            custom_prompt=custom_prompt,
            verbose=True
        )
        end_time = time.time()
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return result, end_time - start_time
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logging.error(f"Error processing file: {str(e)}", exc_info=True)
        return None, 0

def filter_models(models, search_term):
    """Filter models based on search term."""
    if not search_term:
        return []
    
    search_term = search_term.lower()
    if isinstance(models[0], dict):
        return [m for m in models if search_term in m['name'].lower() or search_term in m.get('description', '').lower()]
    else:
        return [m for m in models if search_term in m.lower()]

# Function to select a model
def select_model(model_name):
    st.session_state.selected_model = model_name
    st.query_params.model = model_name
    st.session_state.show_search_results = False
    st.rerun()

def main():
    """Main Streamlit application."""
    # Initialize session state for selected model if it doesn't exist
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = ''
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = ''
        
    st.markdown('<h1 class="main-header">Document Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by multiple AI providers</p>', unsafe_allow_html=True)
    
    # Theme selector in the sidebar
    with st.sidebar:
        # Theme selection
        theme_options = {
            'dark': '🌙 Dark Mode',
            'light': '☀️ Light Mode',
            'high_contrast': '🔆 High Contrast'
        }
        
        selected_theme = st.selectbox(
            "Choose Theme",
            options=list(theme_options.keys()),
            format_func=lambda x: theme_options[x],
            index=list(theme_options.keys()).index(st.session_state.theme),
            help="Select a theme for better visibility"
        )
        
        # If theme changed, update session state and rerun
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        st.header("Settings")
        
        # Check for available API keys (cached)
        available_providers, missing_providers = check_api_keys()
        
        if not available_providers:
            st.error("No API keys found. Please add at least one API key to use the application.")
            st.stop()
        
        # Provider selection with enhanced UI
        provider_options = []
        if "openai" in available_providers:
            provider_options.append(("OpenAI", "openai"))
        if "anthropic" in available_providers:
            provider_options.append(("Anthropic", "anthropic"))
        if "google" in available_providers:
            provider_options.append(("Google", "google"))
        if "openrouter" in available_providers:
            provider_options.append(("OpenRouter", "openrouter"))
        
        # Get URL params for provider/model if any
        query_params = st.query_params
        url_provider = query_params.get('provider', '')
        default_provider = next((p[1] for p in provider_options if p[1] == url_provider), provider_options[0][1])
        
        # If provider changed in URL, update session state
        if url_provider and url_provider != st.session_state.selected_provider:
            st.session_state.selected_provider = url_provider
        
        # Use radio buttons for provider selection
        selected_provider_name = st.radio(
            "Select Provider",
            [p[0] for p in provider_options],
            index=[p[1] for p in provider_options].index(default_provider) if default_provider in [p[1] for p in provider_options] else 0,
            horizontal=True,
            key="provider_selector"
        )
        
        # Get provider key from selected name
        provider_key = next((p[1] for p in provider_options if p[0] == selected_provider_name), "")
        
        # Update session state and URL when provider changes
        if provider_key != st.session_state.selected_provider:
            st.session_state.selected_provider = provider_key
            st.session_state.selected_model = ''  # Reset selected model when provider changes
            st.query_params.provider = provider_key
            if 'model' in st.query_params:
                del st.query_params.model
            st.rerun()  # Force a rerun to update UI
        
        # Add a divider
        st.markdown("---")
        
        # Model selection section title
        st.subheader("Model Selection")
        
        # Get default model from URL if present
        url_model = query_params.get('model', '')
        
        # If model changed in URL, update session state
        if url_model and url_model != st.session_state.selected_model:
            st.session_state.selected_model = url_model
        
        # Get models based on provider with enhanced UI
        if provider_key == "openai":
            model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
            
            # Simple radio selection for small number of models
            selected_model_index = 0
            if st.session_state.selected_model in model_options:
                selected_model_index = model_options.index(st.session_state.selected_model)
                
            selected_model = st.radio(
                "Select Model",
                model_options,
                index=selected_model_index,
                horizontal=True
            )
            
            # Update if changed
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.query_params.model = selected_model
                
        elif provider_key == "anthropic":
            model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            
            # Simple radio selection for small number of models
            display_names = ["Claude 3 Opus", "Claude 3 Sonnet", "Claude 3 Haiku"]
            
            selected_model_index = 0
            if st.session_state.selected_model in model_options:
                selected_model_index = model_options.index(st.session_state.selected_model)
                
            selected_display_name = st.radio(
                "Select Model",
                display_names,
                index=selected_model_index,
                horizontal=True
            )
            
            selected_model = model_options[display_names.index(selected_display_name)]
            
            # Update if changed
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.query_params.model = selected_model
                
        elif provider_key == "google":
            model_options = ["gemini-1.5-pro", "gemini-1.0-pro"]
            
            # Simple radio selection for small number of models
            selected_model_index = 0
            if st.session_state.selected_model in model_options:
                selected_model_index = model_options.index(st.session_state.selected_model)
                
            selected_model = st.radio(
                "Select Model",
                model_options,
                index=selected_model_index,
                horizontal=True
            )
            
            # Update if changed
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.query_params.model = selected_model
                
        elif provider_key == "openrouter":
            # Get models for OpenRouter
            if 'openrouter_models' not in st.session_state:
                st.session_state.openrouter_models = get_openrouter_models()
            
            openrouter_models = st.session_state.openrouter_models
            static_models = [
                "openai/gpt-4-turbo",
                "anthropic/claude-3-opus",
                "anthropic/claude-3-sonnet",
                "google/gemini-pro",
                "mistralai/mistral-large",
                "meta-llama/llama-3-70b-instruct",
                "cohere/command-r"
            ]
            
            # Show refresh button
            refresh_col, time_col = st.columns([1, 2])
            with refresh_col:
                if st.button("🔄 Refresh Models"):
                    # Clear the cache for this function to force a refresh
                    get_openrouter_models.clear()
                    st.session_state.openrouter_models = get_openrouter_models()
                    st.rerun()
            
            with time_col:
                last_fetch_time = st.session_state.get('last_openrouter_fetch', 0)
                if last_fetch_time > 0:
                    st.text(f"Last updated: {time.strftime('%H:%M:%S', time.localtime(last_fetch_time))}")
            
            # Search box for models - using table view
            search_term = st.text_input("🔍 Search models by name or capability", 
                            placeholder="e.g., 'gpt' or 'llama' or 'mistral'")

            # Get search results
            if search_term:
                st.session_state.show_search_results = True
                
                if openrouter_models:
                    filtered_models = filter_models(openrouter_models, search_term)
                else:
                    # Fallback to static list
                    filtered_models = [model for model in static_models if search_term.lower() in model.lower()]
                    filtered_models = [{"name": model, "description": "No description available"} for model in filtered_models]
                
                # Show search results
                if filtered_models:
                    st.markdown(f"**Found {len(filtered_models)} matching models:**")
                    
                    # Create a table layout for all search results
                    model_data = []
                    for model in filtered_models:
                        model_name = model['name'] if isinstance(model, dict) else model
                        display_name = model_name.split('/')[-1] if '/' in model_name else model_name
                        provider = model_name.split('/')[0] if '/' in model_name else "Unknown"
                        model_data.append({"Model": display_name, "Provider": provider, "Name": model_name})
                    
                    # Show results in a dataframe
                    df = pd.DataFrame(model_data)
                    st.dataframe(df[["Model", "Provider"]], use_container_width=True)
                    
                    # Allow selection by model name
                    selected_model_name = st.selectbox(
                        "Select a model from the results above:", 
                        options=df["Name"].tolist(),
                        format_func=lambda x: f"{x.split('/')[-1]} ({x.split('/')[0]})" if '/' in x else x
                    )
                    
                    if st.button("✅ Use Selected Model"):
                        select_model(selected_model_name)
                else:
                    st.info("No models found matching your search. Try a different term.")
            elif st.session_state.selected_model:
                st.success(f"Using model: {st.session_state.selected_model}")
            else:
                # Show suggestions for popular searches
                st.info("Please search for models above. Try searching for: 'gpt', 'claude', 'llama', etc.")
                
                # Offer quick selection buttons for popular models
                st.markdown("### Popular Models")
                popular_models = [
                    ("OpenAI GPT-4", "openai/gpt-4-turbo"),
                    ("Claude 3 Opus", "anthropic/claude-3-opus"),
                    ("Llama 3", "meta-llama/llama-3-70b-instruct"),
                    ("Mistral Large", "mistralai/mistral-large")
                ]
                
                # Create a 2x2 grid of quick-select buttons
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                cols = [col1, col2, col3, col4]
                for i, (display_name, model_name) in enumerate(popular_models):
                    with cols[i]:
                        if st.button(display_name):
                            select_model(model_name)
            
            # If a model is already selected, show it
            if st.session_state.selected_model:
                selected_model = st.session_state.selected_model
            else:
                # Default to first model if none selected
                if openrouter_models and search_term and filtered_models:
                    selected_model = filtered_models[0]['name'] if isinstance(filtered_models[0], dict) else filtered_models[0]
                    st.session_state.selected_model = selected_model
                    st.query_params.model = selected_model
                elif not search_term:
                    selected_model = ""
                else:
                    selected_model = static_models[0]
                    st.session_state.selected_model = selected_model
                    st.query_params.model = selected_model
        
        # Display selected model
        if st.session_state.selected_model:
            st.markdown(f"**Selected Model:** {st.session_state.selected_model}")
            
            # Update URL with selection
            st.query_params.provider = provider_key
            st.query_params.model = st.session_state.selected_model
        
        # Advanced settings in a collapsible section
        with st.expander("Advanced Settings", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative)"
            )
            
            chunk_size = st.slider(
                "Chunk Size",
                min_value=1000,
                max_value=8000,
                value=4000,
                step=500,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between chunks to maintain context"
            )
            
            chain_type = st.radio(
                "Summarization Method",
                ["refine", "map_reduce"],
                index=0,
                help="Chain type for summarization"
            )
        
        # Summary configuration section
        st.subheader("Summary Configuration")

        # Summary length selector
        summary_length = st.select_slider(
            "Summary Length",
            options=["Very Brief", "Brief", "Moderate", "Detailed", "Comprehensive"],
            value="Moderate",
            help="Control how long or detailed the summary should be"
        )

        # Convert the summary length to appropriate max_tokens value
        if summary_length == "Very Brief":
            max_tokens = 250
        elif summary_length == "Brief":
            max_tokens = 500
        elif summary_length == "Moderate":
            max_tokens = 1000
        elif summary_length == "Detailed":
            max_tokens = 2000
        else:  # Comprehensive
            max_tokens = 3000

        # Focus areas for summary
        st.write("**Focus Areas**")
        col1, col2 = st.columns(2)

        with col1:
            focus_key_points = st.checkbox("Key Points", value=True, 
                                        help="Include the main arguments and key points")
            focus_methodology = st.checkbox("Methodology", value=False,
                                            help="Include details about methods and procedures")
            focus_examples = st.checkbox("Examples & Evidence", value=False, 
                                        help="Include examples, data, and supporting evidence")

        with col2:
            focus_conclusions = st.checkbox("Conclusions", value=True,
                                        help="Emphasize findings and conclusions")
            focus_implications = st.checkbox("Implications", value=False,
                                            help="Discuss implications and applications")
            focus_technical = st.checkbox("Technical Details", value=False,
                                        help="Include technical details and specifications")

        # Build the focus instruction for the prompt
        focus_instructions = []
        if focus_key_points:
            focus_instructions.append("key points and main arguments")
        if focus_methodology:
            focus_instructions.append("methodology and procedures")
        if focus_examples:
            focus_instructions.append("examples and supporting evidence")
        if focus_conclusions:
            focus_instructions.append("conclusions and findings")
        if focus_implications:
            focus_instructions.append("implications and applications")
        if focus_technical:
            focus_instructions.append("technical details and specifications")

        # Format the focus areas into a readable string
        if focus_instructions:
            focus_text = ", ".join(focus_instructions[:-1])
            if len(focus_instructions) > 1:
                focus_text += f", and {focus_instructions[-1]}"
            else:
                focus_text = focus_instructions[0]
        else:
            focus_text = "all important aspects"

        # Default prompt that will be updated based on user selections
        default_prompt = f"""
        You are a professional summarizer with expertise in creating clear, concise, and comprehensive summaries.

        Please summarize the following text:

        {{text}}

        Your summary should be {summary_length.lower()} in length and focus primarily on {focus_text}.
        Structure your summary with a logical flow. Include an introduction, body, and conclusion if appropriate for the length.
        """

        # Custom prompt toggle
        use_custom_prompt = st.checkbox("Use Custom Prompt", value=False, 
                                    help="Enable this to customize the summarization prompt")

        # Only show the custom prompt editor if the toggle is enabled
        if use_custom_prompt:
            with st.expander("Custom Prompt Editor", expanded=True):
                custom_prompt = st.text_area(
                    "Enter your custom prompt template",
                    value=default_prompt,
                    height=200,
                    help="Include {text} placeholder where you want the document content to be inserted."
                )
                
                if "{text}" not in custom_prompt:
                    st.warning("Custom prompt must include {text} placeholder.")
                    custom_prompt = default_prompt
        else:
            # If custom prompt is not enabled, use the default prompt with user configurations
            custom_prompt = default_prompt
            
        # Display selected model details
        if st.session_state.selected_model:
            with st.expander("Selected Model Details", expanded=True):
                selected_model = st.session_state.selected_model
                
                if provider_key == "openrouter" and 'openrouter_models' in st.session_state:
                    # Find the selected model details
                    model_details = None
                    for model in st.session_state.openrouter_models:
                        if model['name'] == selected_model:
                            model_details = model
                            break
                    
                    if model_details:
                        st.markdown(f"### {selected_model.split('/')[-1]}")
                        st.markdown(f"**Provider:** {selected_model.split('/')[0] if '/' in selected_model else 'Unknown'}")
                        st.markdown(f"**Description:** {model_details.get('description', 'No description available')}")
                        st.markdown(f"**Context Length:** {model_details.get('context_length', 'Unknown')}")
                        st.markdown(f"**Cost:** {model_details.get('cost', 'Variable')}")
                    else:
                        st.markdown(f"### {selected_model.split('/')[-1]}")
                        st.markdown("Detailed information not available.")
                elif provider_key == "openai":
                    st.markdown(f"### {selected_model}")
                    if selected_model == "gpt-4":
                        st.markdown("**Description:** Most powerful OpenAI model with strong reasoning capabilities")
                        st.markdown("**Context Length:** 8,192 tokens")
                        st.markdown("**Cost:** High ($0.03/1K input tokens, $0.06/1K output tokens)")
                    elif selected_model == "gpt-4-turbo":
                        st.markdown("**Description:** Faster version of GPT-4 with more recent knowledge")
                        st.markdown("**Context Length:** 128,000 tokens")
                        st.markdown("**Cost:** Medium-High ($0.01/1K input tokens, $0.03/1K output tokens)")
                    elif selected_model == "gpt-3.5-turbo":
                        st.markdown("**Description:** Fast and cost-effective model with good capabilities")
                        st.markdown("**Context Length:** 16,385 tokens")
                        st.markdown("**Cost:** Low ($0.0015/1K input tokens, $0.002/1K output tokens)")
                elif provider_key == "anthropic":
                    st.markdown(f"### {selected_model}")
                    if "opus" in selected_model:
                        st.markdown("**Description:** Most powerful Claude model with exceptional reasoning and comprehension")
                        st.markdown("**Context Length:** 200,000 tokens")
                        st.markdown("**Cost:** High ($15/1M input tokens, $75/1M output tokens)")
                    elif "sonnet" in selected_model:
                        st.markdown("**Description:** Balanced performance Claude model with strong capabilities")
                        st.markdown("**Context Length:** 200,000 tokens")
                        st.markdown("**Cost:** Medium ($3/1M input tokens, $15/1M output tokens)")
                    elif "haiku" in selected_model:
                        st.markdown("**Description:** Fast and efficient Claude model for simpler tasks")
                        st.markdown("**Context Length:** 200,000 tokens")
                        st.markdown("**Cost:** Low ($0.25/1M input tokens, $1.25/1M output tokens)")
                elif provider_key == "google":
                    st.markdown(f"### {selected_model}")
                    if selected_model == "gemini-1.5-pro":
                        st.markdown("**Description:** Most powerful Google model with multimodal capabilities")
                        st.markdown("**Context Length:** 1,000,000 tokens")
                        st.markdown("**Cost:** Medium ($0.0025/1K input tokens, $0.0075/1K output tokens)")
                    elif selected_model == "gemini-1.0-pro":
                        st.markdown("**Description:** Standard performance model from Google")
                        st.markdown("**Context Length:** 32,768 tokens")
                        st.markdown("**Cost:** Low ($0.00125/1K input tokens, $0.00375/1K output tokens)")
    
    # Main area for file upload and results
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=["pdf", "docx", "txt", "md", "html"],
            help="Upload a document file to summarize."
        )
        
        if uploaded_file:
            st.markdown(f"""
            <div class="info-box">
                <strong>File:</strong> {uploaded_file.name}<br>
                <strong>Size:</strong> {uploaded_file.size / 1024:.2f} KB
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Summary Settings")
        
        if not uploaded_file:
            st.markdown(
                '<div class="info-box">Please upload a document to generate a summary.</div>',
                unsafe_allow_html=True
            )
        else:
            if st.session_state.selected_model:
                st.markdown(
                    f'<div class="info-box">Ready to summarize <b>{uploaded_file.name}</b> using <b>{selected_provider_name}</b> - <b>{st.session_state.selected_model}</b>.</div>',
                    unsafe_allow_html=True
                )
                
                # Generate summary button
                if st.button("📝 Generate Summary", type="primary"):
                    with st.spinner("Processing document and generating summary..."):
                        result, processing_time = process_file(
                            uploaded_file,
                            provider_key,
                            st.session_state.selected_model,
                            temperature,
                            max_tokens,
                            chunk_size,
                            chunk_overlap,
                            chain_type,
                            custom_prompt
                        )
                    
                    if result:
                        # Display summary
                        st.subheader("Summary")
                        
                        provider_class = f"provider-{provider_key}"
                        st.markdown(
                            f'<div class="result-box {provider_class}">{result["summary"]}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Create download link
                        download_link = create_download_link(
                            result["summary"],
                            f"{uploaded_file.name.split('.')[0]}_summary.txt",
                            "📥 Download Summary"
                        )
                        st.markdown(download_link, unsafe_allow_html=True)
                        
                        # Display metadata
                        with st.expander("Summary Metadata"):
                            st.json(result["metadata"])
                            
                            # Download metadata
                            metadata_json = json.dumps(result["metadata"], indent=2)
                            metadata_link = create_download_link(
                                metadata_json,
                                f"{uploaded_file.name.split('.')[0]}_metadata.json",
                                "📥 Download Metadata"
                            )
                            st.markdown(metadata_link, unsafe_allow_html=True)
                        
                        st.success(f"Summary generated in {processing_time:.2f} seconds!")
            else:
                st.warning("Please select a model before generating a summary.")
    
    # Footer
    st.markdown(
        '<div class="footer">Document Summarizer - Powered by OpenRouter and LangChain</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()