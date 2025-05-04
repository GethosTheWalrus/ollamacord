import discord
import queue
import asyncio
import ollama
import datetime
import os
import aiohttp
import json
import re
import logging
import sys
from collections import deque  # Import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set base level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Set third-party libraries to WARNING level to reduce noise
logging.getLogger('discord').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Keep our application logger at INFO level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

intents = discord.Intents.all()
client = discord.Client(command_prefix="!", intents=intents)
ollama_host = os.getenv("OLLAMA_URL", "http://localhost:11434")
logger.info(f"Connecting to Ollama at: {ollama_host}")
ollama_client = ollama.Client(host=ollama_host)

server_id = None
query_queue = queue.Queue()
history = {}


@client.event
async def on_ready():
    logger.info("We have logged in as {0.user}".format(client))
    client.loop.create_task(background_task())  # Start the background task


@client.event
async def on_message(message):
    global server_id

    if message.author == client.user:
        return

    server_id = message.guild.id # noqa

    if message.content.startswith("!test"):
        messageContent = message.content.split(" ")
        query = (
            " ".join(messageContent[1:])
            if messageContent[0] == "!test"
            else ""
        )
        if len(query) == 0:
            return
        else:
            logger.info(f"Received query: {query}")
            query_queue.put({"query": query, "message": message})
            update_conversation_history("user", query, message.author.name)
            await message.add_reaction("🤔")


async def get_osrs_wiki_content(search_term):
    """Fetch content from OSRS Wiki API and handle redirects."""
    logger.info(f"Attempting to fetch OSRS Wiki content for: {search_term}")
    base_url = "https://oldschool.runescape.wiki/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "titles": search_term
    }
    
    try:
        # Test DNS resolution first
        import socket
        try:
            logger.info("Attempting to resolve oldschool.runescape.wiki...")
            socket.gethostbyname('oldschool.runescape.wiki')
            logger.info("DNS resolution successful")
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed: {str(e)}")
            return None

        # Configure the session with a timeout and DNS settings
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(ssl=True, force_close=True)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            logger.info(f"Making request to {base_url} with params: {params}")
            async with session.get(base_url, params=params) as response:
                logger.info(f"Received response with status: {response.status}")
                if response.status != 200:
                    logger.error(f"Error: Received status code {response.status} from OSRS Wiki API")
                    return None
                    
                data = await response.json()
                logger.info("Successfully parsed JSON response")
                
                # Get the first page from the response
                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    logger.warning("No pages found in response")
                    return None
                    
                page = next(iter(pages.values()))
                
                # Check for redirects
                if "revisions" in page:
                    content = page["revisions"][0]["*"]
                    if content.startswith("#REDIRECT"):
                        logger.info(f"Found redirect in content: {content}")
                        # Extract the redirect target
                        redirect_target = content.split("[[")[1].split("]]")[0]
                        logger.info(f"Following redirect to: {redirect_target}")
                        # Recursively call with the redirect target
                        return await get_osrs_wiki_content(redirect_target)
                    else:
                        logger.info("No redirect found, returning page content")
                        return page
                else:
                    logger.warning("No revisions found in page")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"Network error while fetching from OSRS Wiki: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching from OSRS Wiki: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None


# Tool definitions with reactions
TOOLS = {
    "osrs_wiki": {
        "description": "Use this tool to look up information about Old School RuneScape items, quests, or other game content.",
        "function": get_osrs_wiki_content,
        "trigger_patterns": [
            r"osrs|runescape|rs3|rs\s+wiki",
            r"item|quest|skill|monster|boss",
            r"stats|requirements|location|guide",
            r"price|value|cost|gp",
            r"drop|loot|reward"
        ],
        "reaction": "🔍"  # Magnifying glass for wiki lookups
    }
}


async def extract_search_term(query: str) -> str:
    """
    Use the LLM to extract the most relevant search term from a query.
    Returns the most appropriate search term for the OSRS Wiki.
    """
    prompt = f"""Given this query about Old School RuneScape: "{query}"
    Extract the most specific and relevant search term that would be used to look up information on the OSRS Wiki.
    The search term should be the name of an item, quest, monster, location, or other game content.
    Return ONLY the search term, nothing else.
    
    Examples:
    Query: "What are the stats for the Abyssal whip?"
    Search term: "Abyssal whip"
    
    Query: "How do I complete Dragon Slayer quest?"
    Search term: "Dragon Slayer"
    
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
        return search_term
    except Exception as e:
        logger.error(f"Error extracting search term: {str(e)}")
        # Fallback to the original method if LLM fails
        return None


async def determine_tools(query: str) -> list:
    """
    Use the LLM to determine which tools should be used for a given query.
    Returns a list of tool names that are relevant for the query.
    """
    # Create a description of available tools
    tools_description = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])
    
    prompt = f"""Given this query: "{query}"

Available tools:
{tools_description}

Determine which tools, if any, would be helpful in answering this query.
Return ONLY a comma-separated list of tool names, or "none" if no tools are needed.
Do not include any explanation or additional text.

Examples:
Query: "What are the stats for the Abyssal whip?"
Tools: osrs_wiki

Query: "What's the weather like today?"
Tools: none

Query: "How do I complete Dragon Slayer and what items do I need?"
Tools: osrs_wiki
"""
    
    try:
        response = ""
        for chunk in ollama_client.chat(
            model="llama3.2:16384",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            response += chunk["message"]["content"]
        
        # Clean up the response
        tools = response.strip().lower()
        if tools == "none":
            logger.info("LLM determined no tools are needed")
            return []
        
        # Split the comma-separated list and clean up each tool name
        tool_list = [tool.strip() for tool in tools.split(",")]
        logger.info(f"LLM determined tools to use: {tool_list}")
        return tool_list
    except Exception as e:
        logger.error(f"Error determining tools: {str(e)}")
        # Fallback to the original regex method
        return None


async def process_query():
    if query_queue.empty():
        return

    queue_object = query_queue.get()
    query = queue_object["query"]
    message = queue_object["message"]

    logger.info(
        f"Processing message with id {message.id} "
        f"from user {message.author.name}"
    )

    if len(query) > 2000:
        await message.add_reaction("📏")
        await message.reply(
            "I can only respond to prompts with no more than than 2000"
        )
        return

    try:
        # create the initial reply
        reply = await message.reply("*Thinking...*")
        active_tools = set()  # Keep track of which tools we're using

        # First, determine which tools to use
        tools_to_use = await determine_tools(query)
        
        # If LLM determination failed, fall back to regex matching
        if tools_to_use is None:
            logger.info("Falling back to regex-based tool selection")
            tools_to_use = []
            for tool_name, tool_info in TOOLS.items():
                if any(re.search(pattern, query.lower()) for pattern in tool_info["trigger_patterns"]):
                    tools_to_use.append(tool_name)

        # Check if we need to use any tools
        tool_results = {}
        for tool_name in tools_to_use:
            if tool_name in TOOLS:
                tool_info = TOOLS[tool_name]
                logger.info(f"Using tool: {tool_name}")
                
                # Add the tool's reaction
                if tool_info.get("reaction"):
                    await message.add_reaction(tool_info["reaction"])
                    active_tools.add(tool_info["reaction"])
                
                try:
                    # Get search term from the LLM
                    search_term = await extract_search_term(query)
                    
                    if search_term:
                        logger.info(f"Using LLM-extracted search term: {search_term}")
                        result = await tool_info["function"](search_term)
                        if result and "revisions" in result:
                            logger.info(f"Found results for term: {search_term}")
                            tool_results[tool_name] = {
                                "term": search_term,
                                "content": result["revisions"][0]["*"]
                            }
                    else:
                        # Fallback to the original method if LLM extraction fails
                        logger.info("Falling back to rule-based search term extraction")
                        words = query.split()
                        common_words = {'what', 'where', 'when', 'why', 'how', 'is', 'are', 'the', 'a', 'an', 'for', 'to', 'in', 'on', 'at', 'with'}
                        potential_terms = []
                        
                        # First try: Look for multi-word terms (2-3 words)
                        for i in range(len(words) - 1):
                            term = ' '.join(words[i:i+2]).strip('?.,!')
                            if len(term) > 3 and not any(word.lower() in common_words for word in term.split()):
                                potential_terms.append(term)
                        
                        # Second try: Look for single words
                        for word in words:
                            word = word.strip('?.,!')
                            if len(word) > 3 and word.lower() not in common_words:
                                potential_terms.append(word)
                        
                        logger.info(f"Potential search terms (in order): {potential_terms}")
                        
                        for term in potential_terms:
                            logger.info(f"Trying term: {term}")
                            result = await tool_info["function"](term)
                            if result and "revisions" in result:
                                logger.info(f"Found results for term: {term}")
                                tool_results[tool_name] = {
                                    "term": term,
                                    "content": result["revisions"][0]["*"]
                                }
                                break
                except Exception as e:
                    logger.error(f"Error using tool {tool_name}: {str(e)}")
                    import traceback
                    logger.error(f"Full traceback for tool error: {traceback.format_exc()}")
                    # Remove the tool's reaction if it failed
                    if tool_info.get("reaction") in active_tools:
                        await message.remove_reaction(tool_info["reaction"], client.user)
                        active_tools.remove(tool_info["reaction"])

        # Prepare tool results for the prompt
        tool_context = ""
        if tool_results:
            tool_context = "Here is some relevant information from the OSRS Wiki:\n"
            for tool_name, result in tool_results.items():
                tool_context += f"\nFor {result['term']}:\n{result['content'][:500]}...\n"

        # update the reply as it streams from Ollama
        response = ""
        token_count = 0
        for chunk in ask_ollama(query, tool_context):
            token_count += 1
            response += chunk

        if token_count > 0:
            await reply.edit(content=response)

        # update the chat history and reactions
        await message.remove_reaction("🤔", client.user)
        if not active_tools:  # If no tools were used successfully
            await message.add_reaction("💭")  # Thought bubble for pure LLM response
        else:
            await message.add_reaction("✅")  # Checkmark for successful completion
        update_conversation_history("bot", response, "Ollama")
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        import traceback
        logger.error(f"Full traceback for process_query error: {traceback.format_exc()}")
        # Remove all tool reactions
        for reaction in active_tools:
            await message.remove_reaction(reaction, client.user)
        await message.remove_reaction("🤔", client.user)
        await message.add_reaction("❌")
        await message.reply(
            f"I encountered an error while responding to you: "
            f"```{e}```"
        )


def ask_ollama(query, tool_context=""):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that can use tools to gather information.
            When responding to questions about Old School RuneScape, use the provided wiki information
            to give accurate and detailed answers. If the wiki information is provided, prioritize using
            that information in your response."""
        }
    ]

    if tool_context:
        messages.append({
            "role": "system",
            "content": tool_context
        })

    messages.append({
        "role": "user",
        "content": f"""Refer to this conversation history,
                       but do not directly mention it: {history}
                       Respond to this prompt in fewer than
                       2000 characters: {query}."""
    })

    stream = ollama_client.chat(
        model="llama3.2:16384",
        messages=messages,
        stream=True
    )

    final_content = ""
    for chunk in stream:
        final_content += chunk["message"]["content"]
        yield chunk["message"]["content"]


def update_conversation_history(
    role, content, source
):
    global server_id
    global history

    if server_id is None:
        return

    if server_id not in history:
        history[server_id] = deque(
            maxlen=int(os.getenv("MEMORY_MAX_LENGTH", 20))
        )

    history[server_id].append({
        "role": role,
        "content": content,
        "source": source,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    })


async def background_task():
    while True:
        await process_query()
        await asyncio.sleep(2)  # Pause for 2 seconds


client.run(
    os.getenv("DISCORD_TOKEN")
)
