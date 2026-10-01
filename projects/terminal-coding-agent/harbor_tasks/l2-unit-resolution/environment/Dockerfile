FROM ubuntu:24.04

# python3 + pytest so the agent can run the repo's own test suite offline.
# git + ripgrep are the two tools the agent leans on most.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pytest \
    git \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY repo/ /app/

# Seed a git repo: the agent's `git` tool and the Stop hook's patch.diff need one.
RUN git init -q /app \
    && git -C /app add -A \
    && git -C /app -c user.email=task@example.com -c user.name=task commit -qm "seed"
