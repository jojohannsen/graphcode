#!/usr/bin/env python3
import sys
from langsmith import Client
from pathlib import Path

def get_prompt_content(prompt_name):
    """
    Reads a prompt from the LangSmith hub using the format "johannes/lgcodegen-<prompt_name>".
    
    Args:
        prompt_name (str): The prompt name (e.g., "gen_state_spec")
        
    Returns:
        str: The text content of the prompt template
    """
    client = Client()
    hub_name = f"johannes/lgcodegen-{prompt_name}"
    
    try:
        prompt = client.pull_prompt(hub_name)
        return prompt.messages[0].prompt.template
    except Exception as e:
        raise Exception(f"Failed to fetch prompt '{hub_name}': {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prompt_fetcher.py <prompt_name>")
        print("Example: python prompt_fetcher.py gen_state_spec")
        sys.exit(1)
    
    prompt_name = sys.argv[1]
    
    try:
        # Fetch and print the prompt content
        content = get_prompt_content(prompt_name)
        print(f"Prompt content for 'johannes/lgcodegen-{prompt_name}':")
        print("-" * 50)
        print(content)
        print("-" * 50)
        
        # Create the prompt file
        output_file = f"{prompt_name}_prompt.py"
        with open(output_file, "w") as f:
            f.write(f'"""\nPrompt content for johannes/lgcodegen-{prompt_name}\n"""\n\n')
            f.write(f"{prompt_name.upper()}_PROMPT = '''{content}'''\n")
        
        # Create the text file with just the prompt content
        text_file = f"{prompt_name}.txt"
        with open(text_file, "w") as f:
            f.write(content)
        
        print(f"\nPrompt saved to: {output_file}")
        print(f"Raw prompt text saved to: {text_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)