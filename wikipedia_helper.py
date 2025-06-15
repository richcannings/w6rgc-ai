#!/usr/bin/env python3
# wikipedia_helper.py

import wikipediaapi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Wikipedia API with a custom user agent
# This is required by Wikipedia's API policy
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent="W6RGC-AI-Ham-Radio-Voice-Assistant/1.0"
)

def get_wikipedia_summary(topic: str) -> dict:
    """
    Fetches a summary of a Wikipedia page for a given topic.

    Args:
        topic (str): The topic to search for on Wikipedia.

    Returns:
        dict: A dictionary containing the 'summary' and 'full_text' of the article.
              If the page is not found, the summary will contain an error message.
    """
    if not topic:
        return {"summary": "I need a topic to search for on Wikipedia.", "full_text": ""}

    logging.info(f"Searching Wikipedia for topic: {topic}")
    
    page = wiki_wiki.page(topic)

    if not page.exists():
        logging.warning(f"Wikipedia page for '{topic}' not found.")
        return {"summary": f"Sorry, I could not find a Wikipedia page for '{topic}'. Please try another topic.", "full_text": ""}

    summary = page.summary[:1000] # Limit summary length
    full_text = page.text

    # Add a note to the end of the summary to prompt for more questions
    summary_with_prompt = (
        f"{summary}\n\nI have the full article. Would you like to know more about '{topic}'?"
    )

    logging.info(f"Successfully retrieved summary for '{topic}'.")
    return {"summary": summary_with_prompt, "full_text": full_text}

if __name__ == '__main__':
    # Example usage for testing
    # Test case 1: A topic that exists
    print("--- Testing an existing topic: 'Amateur radio' ---")
    result_existing = get_wikipedia_summary('Amateur radio')
    print(result_existing['summary'])
    
    # Test case 2: A topic that doesn't exist
    print("\n--- Testing a non-existent topic: 'asdfghjkl' ---")
    result_non_existent = get_wikipedia_summary('asdfghjkl')
    print(result_non_existent['summary'])

    # Test case 3: Empty topic
    print("\n--- Testing an empty topic ---")
    result_empty = get_wikipedia_summary('')
    print(result_empty['summary']) 