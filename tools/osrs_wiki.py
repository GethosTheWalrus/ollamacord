import aiohttp
import logging
import socket
import traceback
import ollama
import os
from bs4 import BeautifulSoup
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO level

async def validate_search_term(search_term: str, ollama_client) -> tuple[str, list[str]]:
    """
    Use the OpenSearch API to validate and find the best matching search term.
    Returns a tuple of (validated_term, list_of_urls)
    """
    base_url = "https://oldschool.runescape.wiki/api.php"
    params = {
        "action": "opensearch",
        "search": search_term,
        "format": "json",
        "limit": 20
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error: Received status code {response.status} from OpenSearch API")
                    return search_term, []
                
                data = await response.json()
                if not data or len(data) < 4 or not data[1]:
                    return search_term, []
                
                # Extract the search results
                search_term = data[0]  # Original search term
                suggestions = data[1]  # List of suggestions
                descriptions = data[2]  # List of descriptions (usually empty)
                urls = data[3]  # List of URLs
                
                if len(suggestions) == 1:
                    # If there's only one suggestion, use it
                    return suggestions[0], [urls[0]]  # Return only the matching URL
                elif len(suggestions) > 1:
                    # If there are multiple suggestions, ask the LLM to choose the best one
                    prompt = f"""Given this search term: "{search_term}"
                    And these possible matches from the OSRS Wiki:
                    {', '.join(suggestions)}
                    
                    Which one is the most relevant match? Return ONLY the exact match from the list, nothing else.
                    If none seem relevant, return the original search term: "{search_term}"
                    """
                    
                    response = ""
                    for chunk in ollama_client.chat(
                        model="llama3.2:16384",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    ):
                        response += chunk["message"]["content"]
                    
                    # Clean up the response and find the best match
                    chosen_term = response.strip().strip('"').strip("'")
                    if chosen_term in suggestions:
                        logger.info(f"LLM chose best match: {chosen_term}")
                        # Find the URL for the chosen term
                        chosen_index = suggestions.index(chosen_term)
                        return chosen_term, [urls[chosen_index]]  # Return only the matching URL
                    else:
                        logger.info(f"LLM chose original term: {search_term}")
                        return search_term, []
                
                return search_term, []
                
    except Exception as e:
        logger.error(f"Error validating search term: {str(e)}")
        return search_term, []

async def extract_search_term(query: str, ollama_client) -> tuple[str, list[str]]:
    """
    Use the LLM to extract the most relevant search term from a query.
    Returns a tuple of (search_term, list_of_urls)
    """
    prompt = f"""Given this query about Old School RuneScape: "{query}"
    Extract the most specific and relevant search term that would be used to look up information on the OSRS Wiki.
    The search term should be the name of an item, quest, monster, location, or other game content.
    Return ONLY the search term, nothing else.
    
    Examples:
    Query: "How do I get the Rogue outfit?"
    Search term: "Rogue outfit"
    
    Query: "What is the best way to train Agility?"
    Search term: "Agility training"
    
    Query: "Where can I find the Abyssal demon?"
    Search term: "Abyssal demon"
    
    """
    
    try:
        response = ""
        for chunk in ollama_client.chat(
            model="llama3.2:16384",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            response += chunk["message"]["content"]
        
        # Clean up the response to ensure we only get the search term
        search_term = response.strip().strip('"').strip("'")
        logger.info(f"LLM extracted search term: {search_term}")
        
        # Validate the search term using OpenSearch API
        validated_term, urls = await validate_search_term(search_term, ollama_client)
        if validated_term != search_term:
            logger.info(f"Search term validated: {validated_term}")
        
        return validated_term, urls
    except Exception as e:
        logger.error(f"Error extracting search term: {str(e)}")
        return None, []

def clean_text(text):
    """Clean up text by removing extra whitespace and newlines."""
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_relevant_content(soup):
    """Extract relevant content from the wiki page."""
    content = []
    
    # Get the main content div
    main_content = soup.find('div', {'class': 'mw-parser-output'})
    if not main_content:
        return "No content found"
    
    # Remove unwanted elements
    for element in main_content.find_all(['script', 'style', 'table', 'div']):
        element.decompose()
    
    # Extract text from paragraphs
    for p in main_content.find_all('p'):
        text = clean_text(p.get_text())
        if text and len(text) > 20:  # Only include substantial paragraphs
            content.append(text)
    
    # Extract information from infobox if it exists
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox:
        content.append("\nInfobox Information:")
        for row in infobox.find_all('tr'):
            label = row.find('th')
            value = row.find('td')
            if label and value:
                label_text = clean_text(label.get_text())
                value_text = clean_text(value.get_text())
                if label_text and value_text:
                    content.append(f"{label_text}: {value_text}")
    
    return "\n".join(content)

async def get_osrs_wiki_content(search_term):
    """Fetch content from OSRS Wiki web page."""
    base_url = "https://oldschool.runescape.wiki/w/"
    url = base_url + quote(search_term)
    references = set()  # Use a set to automatically handle duplicates
    
    try:
        # Test DNS resolution first
        try:
            socket.gethostbyname('oldschool.runescape.wiki')
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed: {str(e)}")
            return None

        # Configure the session with a timeout and DNS settings
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(ssl=True, force_close=True)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Error: Received status code {response.status} from OSRS Wiki")
                    return None
                
                # Check for redirects
                if response.history:
                    # If we were redirected, the page exists but might be under a different name
                    # We'll use the final URL's page name as our search term
                    final_url = str(response.url)
                    page_name = final_url.split('/')[-1]
                    if page_name != search_term:
                        return await get_osrs_wiki_content(page_name)
                
                # Parse the HTML content
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check if we got a disambiguation page
                if "may refer to" in html or "disambiguation" in html.lower():
                    # Extract the list of possible pages
                    links = []
                    for link in soup.find_all('a'):
                        href = link.get('href', '')
                        if href.startswith('/w/') and not href.startswith('/w/Special:'):
                            page_name = href.split('/')[-1]
                            if page_name and page_name != search_term:
                                links.append(page_name)
                    if links:
                        return {
                            "type": "disambiguation",
                            "content": f"Multiple pages found for '{search_term}'. Please be more specific. Possible pages: {', '.join(links[:5])}",
                            "references": list(references)  # Convert set back to list
                        }
                
                # Extract relevant content
                content = extract_relevant_content(soup)
                return {
                    "type": "content",
                    "content": content,
                    "references": list(references)  # Convert set back to list
                }
                
    except aiohttp.ClientError as e:
        logger.error(f"Network error while fetching from OSRS Wiki: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching from OSRS Wiki: {str(e)}")
        return None

# Tool definition
TOOL = {
    "description": "Use this tool to look up information about Old School RuneScape items, quests, or other game content.",
    "function": get_osrs_wiki_content,
    "trigger_patterns": [
        r"osrs|runescape|rs3|rs\s+wiki",
        r"item|quest|skill|monster|boss",
        r"stats|requirements|location|guide",
        r"price|value|cost|gp",
        r"drop|loot|reward"
    ],
    "reaction": "🔍",  # Magnifying glass for wiki lookups
    "extract_search_term": extract_search_term  # Add the search term extraction function to the tool
} 