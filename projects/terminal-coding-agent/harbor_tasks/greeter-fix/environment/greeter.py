"""Tiny greeter fot the harbor greeter-fix task."""

def greet(name: str) -> str:
    return f"Helo, {name}"

if __name__ == "__main__":
    print(greet("agent"))