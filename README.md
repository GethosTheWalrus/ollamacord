# Quick Start

1. Clone the repository to your machine
```
git clone https://github.com/GethosTheWalrus/ollamacord.git
```

2. Update docker-compose.yml with your [Discord token](https://discord.com/developers/applications)
```
services:
  python:
    build: .
    environment:
      DISCORD_TOKEN: your_token_here
```

3. Start the app
```
docker-compose up
```
