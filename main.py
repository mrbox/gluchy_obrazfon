"""
Basic Azure OpenAI + Smolagents Application
"""
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from smolagents import LiteLLMModel, ToolCallingAgent, tool
from agents.painter import create_painter_agent
from agents.describer import describe_image

# Load environment variables
load_dotenv()


def generate_random_initial_prompt() -> str:
    """
    Generate a random creative prompt using Azure OpenAI.
    Each call produces a different imaginative scene description.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_endpoint, api_key, deployment_name]):
        raise ValueError(
            "Missing required environment variables for prompt generation. "
            "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and AZURE_OPENAI_DEPLOYMENT_NAME"
        )
    
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    system_prompt = (
        "You are a prompt generator for image generation. "
        "Generate a single, beautiful and imaginative scene description in up to 3 sentences. "
        "Be creative and varied - include different themes. Make it realistic and beautiful. You are forbidden to generate forrest descriptions. "
        "Keep it concise but descriptive (15-25 words)."
    )
    
    user_prompt = "Generate a unique and creative image prompt. You are free with the theme selection."
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.8,  # High temperature for more creativity and randomness
            max_tokens=200
        )
        
        prompt = response.choices[0].message.content.strip()
        # Remove quotes if the LLM wrapped the prompt in them
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1]
        if prompt.startswith("'") and prompt.endswith("'"):
            prompt = prompt[1:-1]
        
        return prompt
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate initial prompt: {str(e)}")


def main():
    """Main application entry point"""
    
    # Azure OpenAI configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_endpoint, api_key, deployment_name]):
        raise ValueError(
            "Missing required environment variables. "
            "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "and AZURE_OPENAI_DEPLOYMENT_NAME"
        )
    
    # Create painter agent
    painter_agent = create_painter_agent()
    
    # Generate initial sentence for the painter using LLM
    print("Generating random initial prompt...")
    initial_prompt = generate_random_initial_prompt()
    
    print("=" * 60)
    print("Iterative Image Generation Loop")
    print("=" * 60)
    print(f"\nInitial prompt: {initial_prompt}\n")
    # return
    # Current prompt starts with the initial sentence
    current_prompt = initial_prompt
    
    # Loop 10 iterations
    for iteration in range(1, 5):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}/10")
        print(f"{'=' * 60}")
        
        # Painter generates an image
        print(f"\n[PAINTER] Generating image from prompt:")
        print(f"  '{current_prompt}'")
        print("\nPainter is working...\n")
        
        image_path = painter_agent.run(f"Generate an image: {current_prompt}")
        print(f"\n[PAINTER] Image saved to: {image_path}")
        
        # Describer describes the image
        print(f"\n[DESCRIBER] Analyzing the generated image...")
        description = describe_image(image_path)
        print(f"\n[DESCRIBER] Description:")
        print(f"  '{description}'")
        
        # Use the description as the prompt for the next iteration
        current_prompt = description
        
        print(f"\n{'=' * 60}")
    
    print("\n" + "=" * 60)
    print("Loop completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
