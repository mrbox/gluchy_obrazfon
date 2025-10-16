"""
Basic Azure OpenAI + Smolagents Application
"""
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from smolagents import LiteLLMModel, ToolCallingAgent, tool

# Load environment variables
load_dotenv()


# Define a custom tool for the agent
@tool
def calculator(operation: str, x: float, y: float) -> float:
    """
    Performs basic mathematical operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        x: First number
        y: Second number
    
    Returns:
        The result of the operation
    """
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        if y == 0:
            raise ValueError("Cannot divide by zero")
        return x / y
    else:
        raise ValueError(f"Unknown operation: {operation}")


def main():
    """Main application entry point"""
    
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
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    # Create LiteLLM model for smolagents (using Azure OpenAI)
    model = LiteLLMModel(
        model_id=f"azure/{deployment_name}",
        api_key=api_key,
        api_base=azure_endpoint,
        api_version=api_version
    )
    
    # Initialize the agent with tools
    agent = ToolCallingAgent(
        tools=[calculator],
        model=model,
        max_steps=10
    )
    
    # Example usage
    print("=" * 60)
    print("Azure OpenAI + Smolagents Application")
    print("=" * 60)
    
    # Run a simple task
    task = "Calculate the result of 15 multiplied by 7, then add 23 to it"
    print(f"\nTask: {task}")
    print("\nAgent is working...\n")
    
    result = agent.run(task)
    
    print(f"\nResult: {result}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
