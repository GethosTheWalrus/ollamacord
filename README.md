# Quick Start

1. Install [Ollama](https://ollama.com) on your machine

2. Clone the repository to your machine
```
git clone https://github.com/GethosTheWalrus/ollamacord.git
```

3. Update docker-compose.yml with your [Discord token](https://discord.com/developers/applications)
```
services:
  python:
    build: .
    environment:
      DISCORD_TOKEN: your_token_here
      OLLAMA_URL: your_ollama_url_here
```

4. Start the app
```
docker-compose up
```
