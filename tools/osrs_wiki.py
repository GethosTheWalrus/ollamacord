import aiohttp
import logging
import socket
import traceback
import os
import time
from bs4 import BeautifulSoup
import re
from urllib.parse import quote
from .ollama_client import ollama_client
from .redis_client import redis_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO level

async def validate_search_term(search_term: str) -> tuple[str, list[str]]:
    """
    Use the OpenSearch API to validate and find the best matching search term.
    Returns a tuple of (validated_term, list_of_urls)
    """
    start_time = time.time()
    logger.info(f"Starting search term validation for: {search_term}")
    
    base_url = "https://oldschool.runescape.wiki/api.php"
    params = {
        "action": "opensearch",
        "search": search_term,
        "format": "json",
        "limit": 20
    }
    
    try:
        # Time the OpenSearch API call
        api_start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Error: Received status code {response.status} from OpenSearch API")
                    return search_term, []
                
                data = await response.json()
                api_duration = time.time() - api_start
                logger.info(f"OpenSearch API call took {api_duration:.2f} seconds")
                
                if not data or len(data) < 4 or not data[1]:
                    logger.error("No results found from OpenSearch API")
                    return search_term, []
                
                # Extract the search results
                original_term = search_term
                suggestions = data[1]
                descriptions = data[2]
                urls = data[3]
                
                logger.info(f"OpenSearch results for '{original_term}':")
                logger.info(f"Found {len(suggestions)} suggestions")
                
                if len(suggestions) == 1:
                    # If there's only one suggestion, use it
                    logger.info(f"Single match found: {suggestions[0]} -> {urls[0]}")
                    total_duration = time.time() - start_time
                    logger.info(f"Total validation took {total_duration:.2f} seconds")
                    return suggestions[0], [urls[0]]
                elif len(suggestions) > 1:
                    try:
                        # Time the LLM validation
                        llm_start = time.time()
                        prompt = f"""You are helping to find the most relevant OSRS Wiki page for a search term.
                        
Original search term: "{original_term}"

Possible matches from the OSRS Wiki:
{chr(10).join(f"{i+1}. {suggestion}" for i, suggestion in enumerate(suggestions))}

Instructions:
1. Choose the most relevant match for the original search term
2. Consider the context of Old School RuneScape
3. If none seem relevant, return the original term
4. Return ONLY the exact match from the list, nothing else

Your choice:"""
                        
                        if not ollama_client.is_available():
                            logger.error("Ollama client is not available, falling back to string matching")
                            raise ValueError("Ollama client is not available")
                        
                        response = ollama_client.chat(
                            messages=[{"role": "user", "content": prompt}],
                            stream=False,
                            options={"timeout": 10}
                        )
                        
                        llm_duration = time.time() - llm_start
                        logger.info(f"LLM validation took {llm_duration:.2f} seconds")
                        
                        # Clean up the response and find the best match
                        chosen_term = response["message"]["content"].strip().strip('"').strip("'").strip()
                        logger.info(f"LLM response: {chosen_term}")
                        
                        # Try to find the chosen term in suggestions
                        if chosen_term in suggestions:
                            logger.info(f"LLM chose best match: {chosen_term}")
                            chosen_index = suggestions.index(chosen_term)
                            chosen_url = urls[chosen_index]
                            total_duration = time.time() - start_time
                            logger.info(f"Total validation took {total_duration:.2f} seconds")
                            return chosen_term, [chosen_url]
                        
                        # If the response is a number, use it as an index
                        try:
                            index = int(chosen_term) - 1
                            if 0 <= index < len(suggestions):
                                logger.info(f"LLM chose match by index: {suggestions[index]}")
                                total_duration = time.time() - start_time
                                logger.info(f"Total validation took {total_duration:.2f} seconds")
                                return suggestions[index], [urls[index]]
                        except ValueError:
                            pass
                        
                        logger.warning("LLM selection failed, falling back to string matching")
                        raise ValueError("LLM selection failed")
                        
                    except Exception as e:
                        logger.error(f"Error in LLM validation: {str(e)}, falling back to string matching")
                        # Time the string matching fallback
                        fallback_start = time.time()
                        
                        # Fall back to string matching if LLM fails
                        if original_term in suggestions:
                            index = suggestions.index(original_term)
                            fallback_duration = time.time() - fallback_start
                            logger.info(f"String matching fallback took {fallback_duration:.2f} seconds")
                            total_duration = time.time() - start_time
                            logger.info(f"Total validation took {total_duration:.2f} seconds")
                            return suggestions[index], [urls[index]]
                        
                        # Then try case-insensitive match
                        lower_term = original_term.lower()
                        for i, suggestion in enumerate(suggestions):
                            if suggestion.lower() == lower_term:
                                fallback_duration = time.time() - fallback_start
                                logger.info(f"String matching fallback took {fallback_duration:.2f} seconds")
                                total_duration = time.time() - start_time
                                logger.info(f"Total validation took {total_duration:.2f} seconds")
                                return suggestion, [urls[i]]
                        
                        # If no exact match, try to find the most relevant match
                        for i, suggestion in enumerate(suggestions):
                            if lower_term in suggestion.lower():
                                fallback_duration = time.time() - fallback_start
                                logger.info(f"String matching fallback took {fallback_duration:.2f} seconds")
                                total_duration = time.time() - start_time
                                logger.info(f"Total validation took {total_duration:.2f} seconds")
                                return suggestion, [urls[i]]
                        
                        # If still no match, use the first suggestion
                        fallback_duration = time.time() - fallback_start
                        logger.info(f"String matching fallback took {fallback_duration:.2f} seconds")
                        total_duration = time.time() - start_time
                        logger.info(f"Total validation took {total_duration:.2f} seconds")
                        return suggestions[0], [urls[0]]
                
                total_duration = time.time() - start_time
                logger.info(f"Total validation took {total_duration:.2f} seconds")
                return original_term, []
                
    except Exception as e:
        logger.error(f"Error validating search term: {str(e)}")
        total_duration = time.time() - start_time
        logger.info(f"Total validation took {total_duration:.2f} seconds")
        return search_term, []

def clean_text(text):
    """Clean up text by removing extra whitespace and newlines."""
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_content(content_list, max_words_per_chunk=4000):
    """Break content into manageable chunks based on sections and word count."""
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for item in content_list:
        # If item starts with a newline and section name, it's a new section
        if item.startswith('\n') and ':' in item:
            # If current chunk has content, save it and start new chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0
        
        # Count words in the new item
        item_words = len(item.split())
        
        # If adding this item would exceed the limit, save current chunk and start new one
        if current_word_count + item_words > max_words_per_chunk and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_word_count = 0
        
        # Add item to current chunk
        current_chunk.append(item)
        current_word_count += item_words
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def extract_relevant_content(soup):
    """Extract relevant content from the wiki page."""
    content = []
    related_links = set()  # Use a set to avoid duplicates
    
    # Get the main content div
    main_content = soup.find('div', {'class': 'mw-parser-output'})
    if not main_content:
        return "No content found", []
    
    # Remove unwanted elements
    for element in main_content.find_all(['script', 'style', 'table', 'div']):
        element.decompose()
    
    # Extract text from paragraphs and collect links
    for p in main_content.find_all('p'):
        # Collect links from the paragraph
        for link in p.find_all('a'):
            href = link.get('href', '')
            if href.startswith('/w/') and not href.startswith('/w/Special:'):
                page_name = href.split('/')[-1]
                if page_name and not page_name.startswith('File:'):
                    related_links.add(page_name)
        
        text = clean_text(p.get_text())
        if text and len(text) > 20:  # Only include substantial paragraphs
            content.append(text)
    
    # Extract information from infobox if it exists
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox:
        infobox_content = []
        for row in infobox.find_all('tr'):
            label = row.find('th')
            value = row.find('td')
            if label and value:
                label_text = clean_text(label.get_text())
                value_text = clean_text(value.get_text())
                if label_text and value_text:
                    infobox_content.append(f"{label_text}: {value_text}")
                    
            # Collect links from the infobox
            for link in row.find_all('a'):
                href = link.get('href', '')
                if href.startswith('/w/') and not href.startswith('/w/Special:'):
                    page_name = href.split('/')[-1]
                    if page_name and not page_name.startswith('File:'):
                        related_links.add(page_name)
                        
        if infobox_content:
            content.append("\nInfobox Information:\n" + "\n".join(infobox_content))
    
    # Extract location information
    location_section = None
    for h2 in main_content.find_all('h2'):
        if 'location' in h2.get_text().lower():
            location_section = h2
            break
    
    if location_section:
        location_content = []
        current = location_section.next_sibling
        while current and current.name != 'h2':
            if current.name == 'p':
                text = clean_text(current.get_text())
                if text:
                    location_content.append(text)
            current = current.next_sibling
        if location_content:
            content.append("\nLocation Information:\n" + "\n".join(location_content))
    
    # Extract drops/items information
    drops_section = None
    for h2 in main_content.find_all('h2'):
        if 'drops' in h2.get_text().lower():
            drops_section = h2
            break
    
    if drops_section:
        drops_content = []
        current = drops_section.next_sibling
        while current and current.name != 'h2':
            if current.name == 'p':
                text = clean_text(current.get_text())
                if text:
                    drops_content.append(text)
            current = current.next_sibling
        if drops_content:
            content.append("\nDrops Information:\n" + "\n".join(drops_content))
    
    # Extract combat information
    combat_section = None
    for h2 in main_content.find_all('h2'):
        if 'combat' in h2.get_text().lower():
            combat_section = h2
            break
    
    if combat_section:
        combat_content = []
        current = combat_section.next_sibling
        while current and current.name != 'h2':
            if current.name == 'p':
                text = clean_text(current.get_text())
                if text:
                    combat_content.append(text)
            current = current.next_sibling
        if combat_content:
            content.append("\nCombat Information:\n" + "\n".join(combat_content))
    
    # Extract quest-specific sections
    quest_sections = {
        'requirements': 'Requirements',
        'rewards': 'Rewards',
        'walkthrough': 'Walkthrough',
        'details': 'Details',
        'required for': 'Required for completing'
    }
    
    for section_id, section_name in quest_sections.items():
        section = None
        for h2 in main_content.find_all('h2'):
            if section_id in h2.get_text().lower():
                section = h2
                break
        
        if section:
            section_content = []
            current = section.next_sibling
            while current and current.name != 'h2':
                if current.name == 'p':
                    text = clean_text(current.get_text())
                    if text:
                        section_content.append(text)
                current = current.next_sibling
            if section_content:
                content.append(f"\n{section_name}:\n" + "\n".join(section_content))
    
    # Extract item-specific sections
    item_sections = {
        'creation': 'Creation',
        'money making': 'Money Making',
        'products': 'Products',
        'uses': 'Uses',
        'item sources': 'Item Sources',
        'combat stats': 'Combat Stats',
        'requirements': 'Requirements',
        'cost': 'Cost',
        'materials': 'Materials',
        'skill requirements': 'Skill Requirements',
        'grand exchange': 'Grand Exchange',
        'advanced data': 'Advanced Data'
    }
    
    for section_id, section_name in item_sections.items():
        section = None
        for h2 in main_content.find_all('h2'):
            if section_id in h2.get_text().lower():
                section = h2
                break
        
        if section:
            section_content = []
            current = section.next_sibling
            while current and current.name != 'h2':
                if current.name == 'p':
                    text = clean_text(current.get_text())
                    if text:
                        section_content.append(text)
                current = current.next_sibling
            if section_content:
                content.append(f"\n{section_name}:\n" + "\n".join(section_content))
    
    # Extract NPC/boss-specific sections
    npc_sections = {
        'strategy': 'Strategy',
        'mechanics': 'Mechanics',
        'combat info': 'Combat Info',
        'slayer info': 'Slayer Info',
        'combat stats': 'Combat Stats',
        'aggressive stats': 'Aggressive Stats',
        'defence': 'Defence',
        'immunities': 'Immunities',
        'drops': 'Drops',
        'changes': 'Changes',
        'gallery': 'Gallery',
        'trivia': 'Trivia',
        'references': 'References'
    }
    
    for section_id, section_name in npc_sections.items():
        section = None
        for h2 in main_content.find_all('h2'):
            if section_id in h2.get_text().lower():
                section = h2
                break
        
        if section:
            section_content = []
            current = section.next_sibling
            while current and current.name != 'h2':
                if current.name == 'p':
                    text = clean_text(current.get_text())
                    if text:
                        section_content.append(text)
                current = current.next_sibling
            if section_content:
                content.append(f"\n{section_name}:\n" + "\n".join(section_content))
    
    # Extract shopkeeper-specific sections
    shop_sections = {
        'stock': 'Stock',
        'dialogue': 'Dialogue',
        'involvement in quests': 'Involvement in Quests',
        'involvement in events': 'Involvement in Events',
        'shop': 'Shop',
        'services': 'Services',
        'repair': 'Repair',
        'trade': 'Trade',
        'options': 'Options',
        'examine': 'Examine',
        'changes': 'Changes',
        'gallery': 'Gallery',
        'trivia': 'Trivia',
        'notes': 'Notes',
        'references': 'References'
    }
    
    for section_id, section_name in shop_sections.items():
        section = None
        for h2 in main_content.find_all('h2'):
            if section_id in h2.get_text().lower():
                section = h2
                break
        
        if section:
            section_content = []
            current = section.next_sibling
            while current and current.name != 'h2':
                if current.name == 'p':
                    text = clean_text(current.get_text())
                    if text:
                        section_content.append(text)
                current = current.next_sibling
            if section_content:
                content.append(f"\n{section_name}:\n" + "\n".join(section_content))
    
    # Extract slayer master-specific sections
    slayer_sections = {
        'slayer masters': 'Slayer Masters',
        'slayer points': 'Slayer Points',
        'requirements': 'Requirements',
        'combat level': 'Combat Level',
        'slayer level': 'Slayer Level',
        'task list': 'Task List',
        'location': 'Location',
        'teleport': 'Teleport',
        'equipment': 'Equipment',
        'rewards': 'Rewards',
        'notes': 'Notes',
        'changes': 'Changes',
        'gallery': 'Gallery',
        'trivia': 'Trivia',
        'references': 'References'
    }
    
    for section_id, section_name in slayer_sections.items():
        section = None
        for h2 in main_content.find_all('h2'):
            if section_id in h2.get_text().lower():
                section = h2
                break
        
        if section:
            section_content = []
            current = section.next_sibling
            while current and current.name != 'h2':
                if current.name == 'p':
                    text = clean_text(current.get_text())
                    if text:
                        section_content.append(text)
                current = current.next_sibling
            if section_content:
                content.append(f"\n{section_name}:\n" + "\n".join(section_content))
    
    # If content is too long, chunk it and summarize each chunk
    if len(' '.join(content).split()) > 200:
        try:
            # Break content into chunks
            chunks = chunk_content(content)
            
            # Summarize each chunk
            summarized_chunks = []
            for chunk in chunks:
                if not ollama_client.is_available():
                    raise ValueError("Ollama client is not available")
                    
                summary_prompt = f"""Summarize this section of OSRS Wiki content in 800 words or fewer, focusing on the most important information:

{chunk}

Summary:"""
                
                response = ollama_client.chat(
                    model=ollama_client.summary_model,  # Use the summary model
                    messages=[{"role": "user", "content": summary_prompt}],
                    stream=False,
                    options={"timeout": 30}  # 30 second timeout for summarization
                )
                
                summarized_chunks.append(response["message"]["content"].strip())
            
            # Combine all summaries
            final_summary = "\n\n".join(summarized_chunks)
            
            # If the combined summary is still too long, summarize it one final time
            if len(final_summary.split()) > 200:
                final_prompt = f"""Create a final summary of these OSRS Wiki content summaries in 200 words or fewer:

{final_summary}

Final Summary:"""
                
                response = ollama_client.chat(
                    model=ollama_client.summary_model,  # Use the summary model
                    messages=[{"role": "user", "content": final_prompt}],
                    stream=False,
                    options={"timeout": 30}  # 30 second timeout for final summary
                )
                
                return response["message"]["content"].strip(), list(related_links)
            
            return final_summary, list(related_links)
            
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            # If summarization fails, return a truncated version
            return " ".join(' '.join(content).split()[:200]) + "...", list(related_links)
    
    return "\n".join(content), list(related_links)

async def get_osrs_wiki_content(search_term: str) -> dict:
    """Fetch content from OSRS Wiki web page."""
    start_time = time.time()
    logger.info(f"Starting wiki content fetch for: {search_term}")
    
    if not ollama_client.is_available():
        logger.error("Ollama client is not available, cannot proceed with wiki lookup")
        return {
            "type": "error",
            "content": "Sorry, I'm having trouble processing your request right now. Please try again later.",
            "references": [],
            "search_message": f"Error searching the OSRS wiki for '{search_term}'..."
        }
    
    # Get max retries from environment variable (default: 3)
    max_retries = int(os.getenv("WIKI_SEARCH_RETRIES", "3"))
    retry_count = 0
    search_messages = []
    
    while retry_count <= max_retries:
        # First validate the search term and get the correct URL
        validated_term, urls = await validate_search_term(search_term)
        
        if not urls:
            logger.warning(f"No valid URL found for search term: {search_term} (attempt {retry_count + 1}/{max_retries + 1})")
            
            if retry_count < max_retries:
                # Try to extract a different search term
                try:
                    if not ollama_client.is_available():
                        raise ValueError("Ollama client is not available")
                        
                    retry_prompt = f"""The search term "{search_term}" didn't yield any results on the OSRS Wiki.
                    Please suggest a different, more specific search term that might work better.
                    Consider:
                    1. Using the exact name of an item, quest, monster, or location
                    2. Removing any level requirements or specific details
                    3. Using more general terms
                    
                    Return ONLY the new search term, nothing else.
                    
                    Examples:
                    Original: "slayer master at level 40"
                    Better: "Slayer master"
                    
                    Original: "how to get rune platebody"
                    Better: "Rune platebody"
                    
                    Original: "best way to train agility at level 30"
                    Better: "Agility training"
                    """
                    
                    response = ollama_client.chat(
                        messages=[{"role": "user", "content": retry_prompt}],
                        stream=False,
                        options={"timeout": 10}
                    )
                    
                    new_term = response["message"]["content"].strip().strip('"').strip("'")
                    logger.info(f"Retrying with new search term: {new_term}")
                    
                    # Add message about retrying
                    search_messages.append(f"Searching the OSRS wiki for '{search_term}'...")
                    search_messages.append(f"No results found. Trying '{new_term}' instead...")
                    
                    search_term = new_term
                    retry_count += 1
                    continue
                    
                except Exception as e:
                    logger.error(f"Error generating new search term: {str(e)}")
                    break
            
            # If we've exhausted all retries or hit an error
            logger.error(f"Failed to find valid URL after {retry_count + 1} attempts")
            return {
                "type": "error",
                "content": f"Sorry, I couldn't find any information about '{search_term}' in the OSRS Wiki after {retry_count + 1} attempts.",
                "references": [],
                "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{search_term}'..."
            }
        
        # Use the validated URL from OpenSearch
        url = urls[0]
        references = set()
        
        # Check Redis cache first
        if redis_client.is_available():
            cached_data = redis_client.get_cached_page(url)
            if cached_data:
                logger.info(f"Retrieved page from cache: {url}")
                total_duration = time.time() - start_time
                logger.info(f"Total content fetch took {total_duration:.2f} seconds (from cache)")
                return cached_data
        
        try:
            # Test DNS resolution first
            dns_start = time.time()
            try:
                socket.gethostbyname('oldschool.runescape.wiki')
                dns_duration = time.time() - dns_start
                logger.info(f"DNS resolution took {dns_duration:.2f} seconds")
            except socket.gaierror as e:
                logger.error(f"DNS resolution failed: {str(e)}")
                return {
                    "type": "error",
                    "content": "Sorry, I'm having trouble connecting to the OSRS Wiki right now. Please try again later.",
                    "references": [],
                    "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{validated_term}'..."
                }

            # Configure the session with a timeout and DNS settings
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=True, force_close=True)
            
            # Time the page fetch
            fetch_start = time.time()
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Error: Received status code {response.status} from OSRS Wiki")
                        return {
                            "type": "error",
                            "content": f"Sorry, I couldn't access the OSRS Wiki page for '{validated_term}'. Please try again later.",
                            "references": [],
                            "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{validated_term}'..."
                        }
                    
                    # Parse the HTML content
                    html = await response.text()
                    fetch_duration = time.time() - fetch_start
                    logger.info(f"Page fetch took {fetch_duration:.2f} seconds")
                    
                    # Time the content processing
                    process_start = time.time()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Check if we got a disambiguation page
                    if "may refer to" in html or "disambiguation" in html.lower():
                        # Extract the list of possible pages
                        links = []
                        for link in soup.find_all('a'):
                            href = link.get('href', '')
                            if href.startswith('/w/') and not href.startswith('/w/Special:'):
                                page_name = href.split('/')[-1]
                                if page_name and page_name != validated_term:
                                    links.append(page_name)
                        if links:
                            result = {
                                "type": "disambiguation",
                                "content": f"Multiple pages found for '{validated_term}'. Please be more specific. Possible pages: {', '.join(links[:5])}",
                                "references": list(references),
                                "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{validated_term}'..."
                            }
                            process_duration = time.time() - process_start
                            logger.info(f"Content processing took {process_duration:.2f} seconds")
                            total_duration = time.time() - start_time
                            logger.info(f"Total content fetch took {total_duration:.2f} seconds")
                            return result
                    
                    # Extract and summarize relevant content
                    content, related_links = extract_relevant_content(soup)
                    
                    # Format the response with related links
                    if related_links:
                        content += "\n\nRelated Links:\n" + "\n".join(f"• {link}" for link in sorted(related_links))
                    
                    result = {
                        "type": "content",
                        "content": content,
                        "references": list(references),
                        "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{validated_term}'..."
                    }
                    
                    # Cache the result if Redis is available
                    if redis_client.is_available():
                        redis_client.cache_page(url, result)
                    
                    process_duration = time.time() - process_start
                    logger.info(f"Content processing took {process_duration:.2f} seconds")
                    total_duration = time.time() - start_time
                    logger.info(f"Total content fetch took {total_duration:.2f} seconds")
                    return result
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching from OSRS Wiki: {str(e)}")
            return {
                "type": "error",
                "content": "Sorry, I'm having trouble connecting to the OSRS Wiki right now. Please try again later.",
                "references": [],
                "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{validated_term}'..."
            }
        except Exception as e:
            logger.error(f"Unexpected error while fetching from OSRS Wiki: {str(e)}")
            return {
                "type": "error",
                "content": "Sorry, something went wrong while processing your request. Please try again later.",
                "references": [],
                "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{validated_term}'..."
            }
    
    # If we've exhausted all retries
    return {
        "type": "error",
        "content": f"Sorry, I couldn't find any information about '{search_term}' in the OSRS Wiki after {max_retries + 1} attempts.",
        "references": [],
        "search_message": "\n".join(search_messages) if search_messages else f"Searching the OSRS wiki for '{search_term}'..."
    }

async def extract_search_term(query: str) -> tuple[str, list[str]]:
    """
    Use the LLM to extract the most relevant search term from a query.
    Returns a tuple of (search_term, list_of_urls)
    """
    if not ollama_client.is_available():
        logger.error("Ollama client is not available")
        return None, []
        
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
        response = ollama_client.chat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"timeout": 10}  # 10 second timeout
        )
        
        # Clean up the response to ensure we only get the search term
        search_term = response["message"]["content"].strip().strip('"').strip("'")
        logger.info(f"LLM extracted search term: {search_term}")
        
        # Validate the search term using OpenSearch API
        validated_term, urls = await validate_search_term(search_term)
        if validated_term != search_term:
            logger.info(f"Search term validated: {validated_term}")
        
        return validated_term, urls
    except Exception as e:
        logger.error(f"Error extracting search term: {str(e)}")
        return None, []

# Tool definition
TOOL = {
    "description": "Use this tool to look up information about Old School RuneScape items, quests, or other game content.",
    "function": get_osrs_wiki_content,  # Use the function directly
    "trigger_patterns": [
        r"osrs|runescape|rs3|rs\s+wiki",
        r"item|quest|skill|monster|boss",
        r"stats|requirements|location|guide",
        r"price|value|cost|gp",
        r"drop|loot|reward"
    ],
    "reaction": "🔍",  # Magnifying glass for wiki lookups
    "thinking_reaction": "🤔",  # Thinking emoji while processing
    "success_reaction": "✅",  # Success emoji when done
    "extract_search_term": extract_search_term  # Add the search term extraction function to the tool
} 