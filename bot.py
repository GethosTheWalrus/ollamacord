import discord
import queue
import asyncio
import ollama
import datetime
import os
from collections import deque  # Import deque

intents = discord.Intents.all()
client = discord.Client(command_prefix="!", intents=intents)
ollama_client = ollama.Client(
    host=os.getenv("OLLAMA_URL", "http://ollama.home:11434"),
)

server_id = None
query_queue = queue.Queue()
history = {}


@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))
    client.loop.create_task(background_task())  # Start the background task


@client.event
async def on_message(message):
    global server_id

    if message.author == client.user:
        return

    server_id = message.guild.id # noqa

    if message.content.startswith("!chat"):
        messageContent = message.content.split(" ")
        query = (
            " ".join(messageContent[1:])
            if messageContent[0] == "!chat"
            else ""
        )
        if len(query) == 0:
            return
        else:
            query_queue.put({"query": query, "message": message})
            update_conversation_history("user", query, message.author.name)
            await message.add_reaction("🤔")


async def process_query():
    if query_queue.empty():
        return

    queue_object = query_queue.get()
    query = queue_object["query"]
    message = queue_object["message"]

    print(
        f"Processing message with id {message.id} "
        f"from user {message.author.name}",
        flush=True
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

        # update the reply as it streams from Ollama
        response = ""
        token_count = 0
        for chunk in ask_ollama(query):
            token_count += 1
            response += chunk

        if token_count > 0:
            await reply.edit(content=response)

        # update the chat history
        await message.remove_reaction("🤔", reply.author)
        await message.add_reaction("✅")
        update_conversation_history("bot", response, "Ollama")
    except Exception as e:
        await message.remove_reaction("🤔", reply.author)
        await message.add_reaction("❌")
        await message.reply(
            f"I encountered an error while responding to you: "
            f"```{e}```"
        )


def ask_ollama(query):
    stream = ollama_client.chat(
        model="llama3.2:16384",
        messages=[
            {
                "role": "user",
                "content": f"""Refer to this converation history,
                               but do not directly mention it: {history}
                               Respond to this prompt in fewer than
                               2000 characters: {query}.""",
            },
        ],
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
