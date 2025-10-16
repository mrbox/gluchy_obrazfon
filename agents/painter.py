"""
Painter Agent - Image Generation using Smolagents
"""
import os
import base64
from pathlib import Path
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from smolagents import LiteLLMModel, ToolCallingAgent, tool
import litellm

# Load environment variables
load_dotenv()

# Configure litellm to drop unsupported params for Azure OpenAI
litellm.drop_params = True


@tool
def generate_image(prompt: str, output_dir: str = "generated_images") -> str:
    """
    Generates an image based on a text description using Azure OpenAI DALL-E.
    
    Args:
        prompt: Text description of the image to generate
        output_dir: Directory to save the generated image (default: "generated_images")
    
    Returns:
        Path to the saved image file
    """
    # Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_DALLE_ENDPOINT")
    api_key = os.getenv("AZURE_DALLE_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    dalle_deployment = "dall-e-3"
    
    if not all([azure_endpoint, api_key]):
        raise ValueError(
            "Missing required environment variables. "
            "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
        )
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    # Generate image
    try:
        response = client.images.generate(
            model=dalle_deployment,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url"  # Can also use "b64_json" for base64
        )
        
        image_url = response.data[0].url
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        filepath = output_path / filename
        
        # Download and save the image
        import requests
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(image_response.content)
        
        return str(filepath.absolute())
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate image: {str(e)}")


def create_painter_agent(model: Optional[LiteLLMModel] = None) -> ToolCallingAgent:
    """
    Creates and returns a Painter agent configured for image generation.
    
    Args:
        model: Optional LiteLLM model instance. If not provided, creates one from env vars.
    
    Returns:
        Configured ToolCallingAgent for image generation
    """
    if model is None:
        # Azure OpenAI configuration
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([azure_endpoint, api_key, deployment_name]):
            raise ValueError(
                "Missing required environment variables. "
                "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "and AZURE_OPENAI_DEPLOYMENT_NAME"
            )
        
        # Create LiteLLM model for smolagents
        model = LiteLLMModel(
            model_id=f"azure/{deployment_name}",
            api_key=api_key,
            api_base=azure_endpoint,
            api_version=api_version
        )
    
    # Initialize the agent with image generation tool
    agent = ToolCallingAgent(
        tools=[generate_image],
        model=model,
        max_steps=10,
        verbosity_level=1
    )
    
    return agent


def main():
    """Example usage of the Painter agent"""
    print("=" * 60)
    print("Painter Agent - Image Generation")
    print("=" * 60)
    
    # Create the painter agent
    agent = create_painter_agent()
    
    # Example task
    task = "Generate an image of a serene mountain landscape at sunset with a lake in the foreground"
    print(f"\nTask: {task}")
    print("\nAgent is working...\n")
    
    result = agent.run(task)
    
    print(f"\nResult: {result}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
