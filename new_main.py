import argparse
import os
import json
import sys
import time
import logging
from typing import Dict, List, Any, Optional
import requests
from dotenv import load_dotenv
from datetime import datetime
from new_evaluator import MultipleChoiceEvaluator as Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class ModelInterface:
    """Base class for model interfaces"""
    def generate_response(self, prompt: str) -> str:
        """Generate a response for the given prompt"""
        raise NotImplementedError("Subclasses must implement this method")


class GoogleModel(ModelInterface):
    """Interface for Google AI Studio models"""
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize with model name and API key
        
        Args:
            model_name: Name of the Google model
            api_key: API key for Google AI Studio
        """
        self.model_name = model_name
        self.api_key = api_key
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Google AI API with retry logic"""
        max_retries = 3
        retry_count = 0
        base_delay = 10  # seconds
        
        while retry_count <= max_retries:
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=self.api_key)
                
                generation_config = {
                    "temperature": 0.01,
                    "top_p": 1,
                    "max_output_tokens": 4000,
                }
                
                logger.info(f"Sending request to Google AI for model {self.model_name}")
                
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=generation_config
                )
                
                response = model.generate_content(prompt)
                
                if hasattr(response, 'text'):
                    logger.info(f"Received successful response, length: {len(response.text)} chars")
                    return response.text
                else:
                    logger.info(f"Received successful response")
                    return str(response)
                
            except Exception as e:
                if "429" in str(e) and retry_count < max_retries:
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)  # Exponential backoff
                    
                    # Try to extract the retry delay from Google's error message
                    retry_seconds = 60  # Default
                    if "retry_delay" in str(e) and "seconds" in str(e):
                        try:
                            # Extract the seconds value from the error message
                            import re
                            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(e))
                            if match:
                                retry_seconds = int(match.group(1))
                                wait_time = max(wait_time, retry_seconds + 5)  # Add a buffer
                        except:
                            pass
                    
                    logger.warning(f"Rate limited (429). Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                elif "GoogleGenerativeAI" not in str(e) and retry_count < max_retries:
                    # Retry for network errors, but not for package errors
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)
                    logger.warning(f"Request error. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    logger.warning(f"Error details: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating response: {e}")
                    return "Error generating response"
        
        # If we've exhausted all retries
        return "Error: Maximum retries exceeded due to rate limiting"



class OpenAIModel(ModelInterface):
    """
    Interface for OpenAI chat models
    (gpt‑4o, gpt‑4‑turbo, gpt‑3.5‑turbo, …)
    """
    def __init__(self, model_name: str, api_key: str):
        """
        Args:
            model_name : e.g. 'gpt-4o'  'gpt-4-turbo'  'gpt-3.5-turbo'
            api_key    : your OPENAI_API_KEY
        """
        self.model_name = model_name
        self.api_key    = api_key

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response with the modern OpenAI Python SDK (v1.x).
        Handles 429s via exponential back‑off.
        """
        import time, openai
        from openai import OpenAI          # <-- v1.x client

        client = OpenAI(api_key=self.api_key)

        max_retries  = 3
        retry_count  = 0
        base_delay_s = 5  # first back‑off = 5 s

        while retry_count <= max_retries:
            try:
                logger.info(f"Sending request to OpenAI model {self.model_name}")
                resp = client.chat.completions.create(
                    model       = self.model_name,
                    messages    = [{"role": "user", "content": prompt}],
                    temperature = 0.01,
                    max_tokens  = 4000
                )
                text = resp.choices[0].message.content.strip()
                logger.info(f"OpenAI response OK, {len(text)} chars")
                return text

            except openai.RateLimitError:
                retry_count += 1
                if retry_count > max_retries:
                    logger.error("OpenAI rate‑limit: maximum retries exceeded")
                    return "Error: Maximum retries exceeded"
                wait = base_delay_s * (2 ** (retry_count - 1))
                logger.warning(f"Rate‑limited — retrying in {wait}s "
                               f"(attempt {retry_count}/{max_retries})")
                time.sleep(wait)

            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                return "Error generating response"

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return "Error generating response"

        return "Error: Maximum retries exceeded"



class DeepSeekModel(ModelInterface):
    """
    Interface for DeepSeek models
    Docs: https://platform.deepseek.com (OpenAI‑compatible chat endpoint)
    """
    def __init__(self, model_name: str, api_key: str):
        """
        Args:
            model_name: e.g. 'deepseek-chat'  or  'deepseek-coder'
            api_key   : your DEEPSEEK_API_KEY
        """
        self.model_name = model_name
        self.api_key    = api_key

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using DeepSeek’s OpenAI‑style REST API
        with manual exponential‑back‑off retry logic.
        """
        import time, requests, json

        max_retries   = 3
        retry_count   = 0
        base_delay_s  = 5      # first back‑off = 5 s

        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type" : "application/json"
        }

        payload = {
            "model"    : self.model_name,
            "messages" : [{"role": "user", "content": prompt}],
            "temperature" : 0.01,
            "max_tokens"  : 4000
        }

        while retry_count <= max_retries:
            try:
                logger.info(f"Sending request to DeepSeek model {self.model_name}")
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

                # ---- success -------------------------------------------------
                if response.status_code == 200:
                    text = response.json()["choices"][0]["message"]["content"].strip()
                    logger.info(f"DeepSeek response OK, {len(text)} chars")
                    return text

                # ---- rate‑limit / retry --------------------------------------
                elif response.status_code == 429 and retry_count < max_retries:
                    retry_count += 1
                    wait = base_delay_s * (2 ** (retry_count - 1))
                    logger.warning(f"DeepSeek 429 rate‑limit — retrying in {wait}s "
                                   f"(attempt {retry_count}/{max_retries})")
                    time.sleep(wait)
                    continue

                # ---- other HTTP error ----------------------------------------
                else:
                    logger.error(f"DeepSeek API error {response.status_code}: {response.text}")
                    return "Error generating response"

            # ---- network / unexpected error ---------------------------------
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    wait = base_delay_s * (2 ** (retry_count - 1))
                    logger.warning(f"DeepSeek request error '{e}' — retrying in {wait}s "
                                   f"(attempt {retry_count}/{max_retries})")
                    time.sleep(wait)
                    continue
                logger.error(f"DeepSeek fatal error: {e}")
                return "Error generating response"

        return "Error: Maximum retries exceeded"

class AnthropicModel(ModelInterface):
    """Interface for Anthropic Claude models"""
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize with model name and API key
        
        Args:
            model_name: Name of the Anthropic model
            api_key: API key for Anthropic
        """
        self.model_name = model_name
        self.api_key = api_key
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the Anthropic API with retry logic"""
        max_retries = 3
        retry_count = 0
        base_delay = 5  # seconds
        
        while retry_count <= max_retries:
            try:
                import anthropic
                
                client = anthropic.Anthropic(api_key=self.api_key)
                
                logger.info(f"Sending request to Anthropic for model {self.model_name}")
                
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.01,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response_text = response.content[0].text
                logger.info(f"Received successful response, length: {len(response_text)} chars")
                return response_text
                
            except Exception as e:
                if "rate limit" in str(e).lower() and retry_count < max_retries:
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)
                    logger.warning(f"Rate limited. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                elif "anthropic" not in str(e).lower() and retry_count < max_retries:
                    # Retry for network errors, but not for package errors
                    retry_count += 1
                    wait_time = base_delay * (2 ** retry_count)
                    logger.warning(f"Request error. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                    logger.warning(f"Error details: {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating response: {e}")
                    return "Error generating response"
        
        # If we've exhausted all retries
        return "Error: Maximum retries exceeded due to rate limiting"

def get_api_key(provider, cli_api_key):
    """Get API key from CLI args or environment variable"""
    if cli_api_key:
        return cli_api_key
    
    env_var_map = {
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":      "OPENAI_API_KEY",
        "deepseek":    "DEEPSEEK_API_KEY"
    }
    
    env_var = env_var_map.get(provider)
    if not env_var:
        logger.error(f"Unknown provider: {provider}")
        return None
    
    api_key = os.environ.get(env_var)
    if not api_key:
        logger.warning(f"No API key found for {provider} in .env file")
    
    return api_key

def create_output_folder(base_dir):
    """Create a timestamped output folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on open-ended questions")
    parser.add_argument("--dataset", type=str, default="output.json",
                      help="Path to the dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="results_openended",
                      help="Directory to store evaluation results")
    parser.add_argument("--provider", type=str, required=False,
                      choices=[ "google", 
                               "anthropic", "openai", "deepseek"],
                      help="API provider to use")
    parser.add_argument("--model_name", type=str, required=False,
                      help="Name of the model to use")
    parser.add_argument("--api_key", type=str, default=None,
                      help="API key for the selected provider (can also be set in .env file)")
    parser.add_argument("--subjects", type=str, default="all",
                      help="Comma-separated list of subjects to evaluate")
    parser.add_argument("--compute_accuracy", action="store_true",
                      help="Compute accuracy from existing results")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose output")
    parser.add_argument("--timestamped", action="store_true",
                      help="Create a timestamped subfolder in the output directory")
    parser.add_argument("--num_examples", type=int, default=1,  # Changed from 3 to 1 to reduce token usage
                      help="Number of examples to use for few-shot prompting")
    parser.add_argument("--delay", type=float, default=10.0,
                      help="Delay between API requests in seconds")
    parser.add_argument("--batch_size", type=int, default=5,
                      help="Number of questions to process in each batch")
    parser.add_argument("--batch_delay", type=int, default=60,
                      help="Delay between batches in seconds")
    parser.add_argument("--install_deps", action="store_true",
                      help="Install required dependencies (sympy)")
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        try:
            import pip
            pip.main(['install', 'sympy'])
            logger.info("Successfully installed sympy")
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory if it doesn't exist
    if args.timestamped:
        args.output_dir = create_output_folder(args.output_dir)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Using output directory: {args.output_dir}")
    
    # Create evaluator
    evaluator = Evaluator(args.dataset, args.output_dir, delay_between_requests=args.delay)

    
    # If just computing accuracy, do that and exit
    if args.compute_accuracy:
        evaluator.compute_accuracy(args.output_dir)
        return
    
    # For model evaluation, we need provider and model name
    if not args.provider or not args.model_name:
        logger.error("Provider and model_name are required for model evaluation.")
        parser.print_help()
        sys.exit(1)
    
    # Get API key from CLI args or environment variable
    api_key = get_api_key(args.provider, args.api_key)
    
    if not api_key:
        logger.error(f"No API key provided for {args.provider}. Use --api_key or set in .env file.")
        sys.exit(1)
    
    # Determine subjects to evaluate
    subjects = None
    if args.subjects != "all":
        subjects = args.subjects.split(",")
        logger.info(f"Will evaluate on selected subjects: {subjects}")
    
    # Create model interface based on provider
    model = None
    if   args.provider == "google":
        model = GoogleModel(args.model_name, api_key)
    elif args.provider == "anthropic":
        model = AnthropicModel(args.model_name, api_key)
    elif args.provider == "openai":
        model = OpenAIModel(args.model_name, api_key)
    elif args.provider == "deepseek":
        model = DeepSeekModel(args.model_name, api_key)
    else:
        logger.error(f"Unknown provider: {args.provider}")
        sys.exit(1)
    
    # Save configuration
    config = {
        "provider": args.provider,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "subjects": subjects if subjects else "all",
        "timestamp": datetime.now().isoformat(),
        "num_examples": args.num_examples,
        "delay_between_requests": args.delay,
        "batch_size": args.batch_size,
        "batch_delay": args.batch_delay
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run evaluation
    logger.info(f"Starting evaluation with {args.provider} model: {args.model_name}")
    logger.info(f"Using {args.num_examples} examples per prompt")
    logger.info(f"Using delay of {args.delay}s between requests and {args.batch_delay}s between batches of {args.batch_size}")
    
    start_time = time.time()
    results = evaluator.evaluate_model(
        model=model,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    print("\nEvaluation Results:")
    print(f"accuracy: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")

if __name__ == "__main__":
    main()