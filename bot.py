import discord
import queue
import asyncio
import datetime
import os
import json
import re
import logging
import sys
import socket
import aiohttp
from collections import deque  # Import deque
from tools import TOOLS  # Import tools from the new module
from tools.ollama_client import ollama_client  # Import the singleton client

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Set all loggers to INFO level
logging.getLogger('discord').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.INFO)
logging.getLogger('httpcore').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)

# Keep our application logger at INFO level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Test network connectivity
async def test_network():
    logger.info("Testing network connectivity...")
    try:
        # Test DNS resolution
        logger.info("Testing DNS resolution...")
        discord_ip = socket.gethostbyname('discord.com')
        logger.info(f"Discord.com resolves to: {discord_ip}")
        
        # Test HTTP connection
        logger.info("Testing HTTP connection to Discord...")
        async with aiohttp.ClientSession() as session:
            async with session.get('https://discord.com/api/v10/users/@me') as response:
                logger.info(f"Discord API response status: {response.status}")
    except Exception as e:
        logger.error(f"Network test failed: {str(e)}")
        raise

# Configure intents explicitly
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
intents.messages = True  # Enable message events
intents.guilds = True  # Enable guild events

client = discord.Client(intents=intents)
server_id = None
query_queue = queue.Queue()
history = {}

@client.event
async def on_ready():
    logger.info("We have logged in as {0.user}".format(client))
    logger.info("Bot is ready to receive messages")
    client.loop.create_task(background_task())  # Start the background task

@client.event
async def on_connect():
    logger.info("Bot connected to Discord")
    await test_network()  # Test network when connected

@client.event
async def on_disconnect():
    logger.warning("Bot disconnected from Discord")

@client.event
async def on_error(event, *args, **kwargs):
    logger.error(f"Discord error in {event}: {args} {kwargs}")

@client.event
async def on_message(message):
    global server_id

    if message.author == client.user:
        return

    server_id = message.guild.id # noqa

    if message.content.startswith("!ai"):
        messageContent = message.content.split(" ")
        query = (
            " ".join(messageContent[1:])
            if messageContent[0] == "!ai"
            else ""
        )
        if len(query) == 0:
            return
        else:
            query_queue.put({"query": query, "message": message})
            update_conversation_history("user", query, message.author.name)
            await message.add_reaction("🤔")
    else:
        return

async def determine_tools(query: str) -> list:
    """
    Use the LLM to determine which tools should be used for a given query.
    Returns a list of tool names that are relevant for the query.
    """
    if not ollama_client.is_available():
        logger.error("Ollama client is not available, falling back to regex matching")
        return None
        
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
        response = ollama_client.chat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"timeout": 10}  # 10 second timeout
        )
        
        # Clean up the response
        tools = response["message"]["content"].strip().lower()
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
        all_references = set()  # Keep track of all references

        # First, determine which tools to use
        tools_to_use = await determine_tools(query)
        
        # If LLM determination failed, fall back to regex matching
        if tools_to_use is None:
            tools_to_use = []
            for tool_name, tool_info in TOOLS.items():
                if any(re.search(pattern, query.lower()) for pattern in tool_info["trigger_patterns"]):
                    tools_to_use.append(tool_name)

        # Check if we need to use any tools
        tool_results = {}
        for tool_name in tools_to_use:
            if tool_name in TOOLS:
                tool_info = TOOLS[tool_name]
                
                # Add the tool's reaction
                if tool_info.get("reaction"):
                    await message.add_reaction(tool_info["reaction"])
                    active_tools.add(tool_info["reaction"])
                
                try:
                    # Get search term from the LLM using the tool's extract_search_term function
                    if "extract_search_term" in tool_info:
                        search_term, urls = await tool_info["extract_search_term"](query)
                        if urls:
                            all_references.update(urls)
                            # If it's the OSRS wiki tool, add a message about searching
                            if tool_name == "osrs_wiki":
                                # Update the reply to include the search message
                                await reply.edit(content=f"*Thinking...*\n*Searching the OSRS wiki for {search_term}...*")
                    else:
                        search_term = None
                    
                    if search_term:
                        result = await tool_info["function"](search_term)
                        if result:
                            if result.get("type") == "disambiguation":
                                # Handle disambiguation pages
                                tool_results[tool_name] = {
                                    "term": search_term,
                                    "content": result["content"]
                                }
                                if "references" in result:
                                    all_references.update(result["references"])
                            else:
                                # Handle regular content
                                tool_results[tool_name] = {
                                    "term": search_term,
                                    "content": result["content"]
                                }
                                if "references" in result:
                                    all_references.update(result["references"])
                    else:
                        # Fallback to the original method if LLM extraction fails
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
                        
                        for term in potential_terms:
                            result = await tool_info["function"](term)
                            if result:
                                if result.get("type") == "disambiguation":
                                    # Handle disambiguation pages
                                    tool_results[tool_name] = {
                                        "term": term,
                                        "content": result["content"]
                                    }
                                    if "references" in result:
                                        all_references.update(result["references"])
                                else:
                                    # Handle regular content
                                    tool_results[tool_name] = {
                                        "term": term,
                                        "content": result["content"]
                                    }
                                    if "references" in result:
                                        all_references.update(result["references"])
                                break
                except Exception as e:
                    logger.error(f"Error using tool {tool_name}: {str(e)}")
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

        # Add references section if we have any
        if all_references:
            response += "\n\n**References:**\n"
            for ref in sorted(all_references):
                response += f"• {ref}\n"

        if token_count > 0:
            await reply.edit(content=response)

        # update the chat history and reactions
        await message.remove_reaction("🤔", client.user)
        await message.add_reaction("✅")
        if active_tools:
            await message.add_reaction("🔨")
        update_conversation_history("bot", response, "Ollama")
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
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
    if not ollama_client.is_available():
        logger.error("Ollama client is not available")
        yield "Sorry, I'm having trouble processing your request right now. Please try again later."
        return
        
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

    try:
        response = ollama_client.chat(
            messages=messages,
            stream=True,
            options={"timeout": 30}  # 30 second timeout
        )
        
        for chunk in response:
            yield chunk["message"]["content"]
    except Exception as e:
        logger.error(f"Error in ask_ollama: {str(e)}")
        yield "Sorry, I encountered an error while processing your request. Please try again later."

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

    # If MEMORY_MAX_LENGTH is 0, only keep a summary
    if history[server_id].maxlen == 0:
        try:
            if not ollama_client.is_available():
                raise ValueError("Ollama client is not available")
                
            # If we already have a summary, update it
            if len(history[server_id]) > 0 and history[server_id][-1]["source"] == "summary":
                current_summary = history[server_id][-1]["content"]
                summary_prompt = f"""Update this conversation summary to include the new message. Keep the total summary under 150 words:

Current summary: {current_summary}

New message to add: {json.dumps({"role": role, "content": content, "source": source}, indent=2)}

Updated summary:"""
            else:
                # Create new summary
                summary_prompt = f"""Create a 150-word summary of this message:

{json.dumps({"role": role, "content": content, "source": source}, indent=2)}

Summary:"""
            
            response = ollama_client.chat(
                model=ollama_client.summary_model,
                messages=[{"role": "user", "content": summary_prompt}],
                stream=False,
                options={"timeout": 30}
            )
            
            # Clear any existing messages and add only the summary
            history[server_id].clear()
            history[server_id].append({
                "role": "system",
                "content": f"Conversation summary: {response['message']['content'].strip()}",
                "source": "summary",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating conversation summary: {str(e)}")
            # If summary fails, don't add anything to history
            return
    else:
        # If we're at max length, the oldest message will roll off
        if len(history[server_id]) >= history[server_id].maxlen:
            # Get the message that's about to roll off
            rolled_message = history[server_id][0]
            
            # Create or update the summary
            if len(history[server_id]) > 0 and history[server_id][-1]["source"] == "summary":
                # Update existing summary
                current_summary = history[server_id][-1]["content"]
                summary_prompt = f"""Update this conversation summary to include the rolled-off message. Keep the total summary under 150 words:

Current summary: {current_summary}

New message to add: {json.dumps(rolled_message, indent=2)}

Updated summary:"""
            else:
                # Create new summary
                summary_prompt = f"""Create a 150-word summary of this message:

{json.dumps(rolled_message, indent=2)}

Summary:"""
            
            try:
                if not ollama_client.is_available():
                    raise ValueError("Ollama client is not available")
                    
                response = ollama_client.chat(
                    model=ollama_client.summary_model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    stream=False,
                    options={"timeout": 30}
                )
                
                # Remove the oldest message and add the updated summary
                history[server_id].popleft()  # Remove the rolled message
                history[server_id].append({
                    "role": "system",
                    "content": f"Previous conversation summary: {response['message']['content'].strip()}",
                    "source": "summary",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Error updating conversation summary: {str(e)}")
                # If summary fails, just remove the oldest message
                history[server_id].popleft()

    # Add the new message (unless we're in summary-only mode)
    if history[server_id].maxlen > 0:
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
