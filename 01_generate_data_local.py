"""
Step 1 (alt): Generate synthetic D&D training data using a local vLLM server.

Drop-in replacement for 01_generate_data.py that hits a vLLM instance
via its OpenAI-compatible API (default: http://localhost:8000).

Sends concurrent async requests so vLLM's continuous batching engine
can process multiple generations in parallel.

Generates the same output format: data/dnd_training.jsonl

Start vLLM first:
    vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --max-model-len 4096

Usage:
    python 01_generate_data_local.py
    python 01_generate_data_local.py --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --num 500
    python 01_generate_data_local.py --concurrency 16
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

# --- Configuration ---
DEFAULT_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
VLLM_BASE_URL = "http://localhost:8000/v1"
NUM_EXAMPLES = 2500
CONCURRENCY = 8  # Number of requests in flight at once
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "dnd_training.jsonl"

# --- Prompt categories (same as Claude version) ---
CATEGORIES = {
    "npc_character": {
        "weight": 20,
        "system": "You are a D&D dungeon master's assistant. Generate creative, detailed NPCs.",
        "prompts": [
            "Create a mysterious NPC shopkeeper for a port town.",
            "Generate a morally ambiguous quest-giver NPC.",
            "Create an NPC tavern keeper with a dark secret.",
            "Design a friendly NPC who is secretly a villain.",
            "Create a retired adventurer NPC running an inn.",
            "Generate a street urchin NPC who knows everything happening in the city.",
            "Create an NPC blacksmith who forges magical weapons.",
            "Design an NPC sage who speaks only in riddles.",
            "Create an NPC guard captain torn between duty and corruption.",
            "Generate an NPC traveling merchant from a distant land.",
            "Create a druid NPC living on the outskirts of town.",
            "Design an NPC noble with a gambling problem.",
            "Create an NPC bard who collects dangerous stories.",
            "Generate an NPC healer hiding from their past.",
            "Create a warlock NPC whose patron is demanding something terrible.",
        ],
    },
    "quest_hook": {
        "weight": 20,
        "system": "You are a D&D dungeon master's assistant. Generate compelling quest hooks and outlines.",
        "prompts": [
            "Create a quest hook involving a missing person in a small village.",
            "Generate a dungeon crawl quest for level 3 adventurers.",
            "Design a political intrigue quest in a royal court.",
            "Create a quest involving a cursed item that won't stop following the party.",
            "Generate a rescue mission quest in an underwater temple.",
            "Create a quest where the party must broker peace between warring factions.",
            "Design a heist quest to steal from a dragon's hoard.",
            "Create a quest involving a plague spreading through a city.",
            "Generate a quest where an ancient evil is awakening beneath a mountain.",
            "Create a mystery quest involving murdered merchants on a trade route.",
            "Design a quest where the party escorts a dangerous prisoner.",
            "Create a quest centered around a magical tournament gone wrong.",
            "Generate a quest involving a portal to the Feywild opening in a farmer's field.",
            "Create a quest where a beloved NPC has been replaced by a shapeshifter.",
            "Design a quest involving competing adventuring parties racing for the same artifact.",
        ],
    },
    "dialog": {
        "weight": 25,
        "system": "You are a D&D dungeon master's assistant. Generate immersive, in-character dialog.",
        "prompts": [
            "Write dialog for a villain monologue before a boss fight.",
            "Generate a conversation between a party and a suspicious innkeeper.",
            "Write dialog for an NPC begging the party for help.",
            "Create a negotiation dialog between the party and a bandit leader.",
            "Write dialog for an ancient dragon who finds mortals amusing.",
            "Generate a heated argument between two NPC faction leaders.",
            "Write dialog for a dying NPC revealing a crucial secret.",
            "Create dialog for a trickster fey creature bargaining with the party.",
            "Write dialog for a ghost who doesn't know they're dead.",
            "Generate a conversation with a friendly but unhelpful bureaucrat.",
            "Write dialog for a mentor NPC training the party.",
            "Create dialog for an NPC betraying the party.",
            "Write tavern banter between colorful local NPCs.",
            "Generate dialog for a deity speaking through a shrine.",
            "Write dialog for a captured enemy trying to bargain for their life.",
        ],
    },
    "location": {
        "weight": 15,
        "system": "You are a D&D dungeon master's assistant. Generate vivid, atmospheric location descriptions.",
        "prompts": [
            "Describe a haunted forest clearing at twilight.",
            "Generate a description for an abandoned dwarven mine.",
            "Describe a bustling marketplace in a desert city.",
            "Create a description of a necromancer's tower interior.",
            "Describe a peaceful elven village hidden in ancient trees.",
            "Generate a description of a volcanic forge used by fire giants.",
            "Describe an underwater ruins entrance.",
            "Create a description of a thieves' guild hidden beneath a city.",
            "Describe a cursed battlefield where ghosts still fight.",
            "Generate a description of a wizard's library that shifts and rearranges.",
            "Describe a frozen temple at the top of a mountain.",
            "Create a description of a swamp village built on stilts.",
            "Describe the interior of a living dungeon that breathes.",
            "Generate a description of a celestial observatory floating in the sky.",
            "Describe a crossroads where the material plane is thin.",
        ],
    },
    "encounter": {
        "weight": 20,
        "system": "You are a D&D dungeon master's assistant. Design interesting combat and social encounters.",
        "prompts": [
            "Design a combat encounter with goblins that uses interesting terrain.",
            "Create a social encounter at a noble's masquerade ball.",
            "Design a puzzle encounter involving elemental runes.",
            "Create an ambush encounter on a narrow mountain pass.",
            "Design an encounter with a monster that can't be defeated by combat alone.",
            "Create a trap-filled corridor encounter for a dungeon.",
            "Design an encounter where the party must protect civilians during a raid.",
            "Create an encounter with a creature that feeds on magic.",
            "Design a chase encounter through city rooftops.",
            "Create an encounter in a room where gravity shifts.",
            "Design a stealth encounter infiltrating an enemy camp.",
            "Create an encounter with a friendly monster being hunted unfairly.",
            "Design a boss encounter with multiple phases.",
            "Create an encounter involving environmental hazards (lava, flooding, collapse).",
            "Design an encounter where former allies have turned against the party.",
        ],
    },
}


def build_prompt(category: str, base_prompt: str) -> str:
    """Add variation to base prompts to get diverse outputs."""
    variations = [
        "",
        " Make it suitable for a dark fantasy campaign.",
        " Give it a lighthearted, comedic tone.",
        " Include sensory details (sounds, smells, textures).",
        " Make it suitable for new players.",
        " Add an unexpected twist.",
        " Include hooks for future adventures.",
        " Set it in a wintery, northern setting.",
        " Set it in a tropical, coastal setting.",
        " Make it morally complex with no clear right answer.",
    ]
    variation = random.choice(variations)
    detail_requests = [
        "",
        " Be detailed and specific.",
        " Keep it concise but evocative.",
        " Include stat suggestions where appropriate.",
        " Include a memorable quote or catchphrase.",
    ]
    detail = random.choice(detail_requests)
    return base_prompt + variation + detail


SYSTEM_MESSAGE = (
    "You are a creative Dungeons & Dragons dungeon master's assistant. "
    "You help create memorable characters, compelling quests, immersive dialog, "
    "vivid locations, and exciting encounters for D&D campaigns."
)


async def generate_example(
    client: AsyncOpenAI, model: str, category: str, info: dict
) -> dict:
    """Generate a single training example via the vLLM OpenAI-compatible API."""
    base_prompt = random.choice(info["prompts"])
    user_prompt = build_prompt(category, base_prompt)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": info["system"]},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.8,
    )

    assistant_text = response.choices[0].message.content

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "category": category,
    }


def check_vllm(base_url: str, model: str):
    """Verify vLLM is running and the model is available."""
    sync_client = OpenAI(base_url=base_url, api_key="unused")
    try:
        models = sync_client.models.list()
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to vLLM server. Make sure it's running.\n"
            f"Start it with: vllm serve {model} --max-model-len 4096\n"
            f"Error: {e}"
        )

    available = [m.id for m in models.data]
    if model not in available:
        print(f"WARNING: Model '{model}' not found. Available: {available}")
        if available:
            print(f"Hint: try --model {available[0]}")
        raise RuntimeError(f"Model '{model}' not served by vLLM")


async def worker(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    index: int,
    total: int,
    category: str,
    info: dict,
) -> dict | None:
    """Worker that respects the concurrency semaphore."""
    async with semaphore:
        try:
            example = await generate_example(client, model, category, info)
            print(f"  [{index}/{total}] Generated ({category})")
            return example
        except Exception as e:
            print(f"  Error on example {index}: {e}")
            return None


async def run(args):
    check_vllm(args.url, args.model)
    print("vLLM connection verified.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build weighted category list
    weighted_categories = []
    for cat, info in CATEGORIES.items():
        weighted_categories.extend([(cat, info)] * info["weight"])

    existing = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = sum(1 for _ in f)
        print(f"Found {existing} existing examples, will append new ones...")

    examples_needed = args.num - existing
    if examples_needed <= 0:
        print(f"Already have {existing} examples. Delete {OUTPUT_FILE} to regenerate.")
        return

    print(
        f"Generating {examples_needed} D&D training examples "
        f"({args.concurrency} concurrent requests)..."
    )

    client = AsyncOpenAI(base_url=args.url, api_key="unused")
    semaphore = asyncio.Semaphore(args.concurrency)

    # Build all tasks up front
    tasks = []
    category_counts = {}
    for i in range(examples_needed):
        cat, info = random.choice(weighted_categories)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        tasks.append(worker(semaphore, client, args.model, i + 1, examples_needed, cat, info))

    start = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    generated = 0
    with open(OUTPUT_FILE, "a") as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result) + "\n")
                generated += 1

    print(f"\nDone! Generated {generated} examples in {elapsed:.1f}s.")
    print(f"Category breakdown: {category_counts}")
    print(f"Output: {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate D&D training data using a local vLLM server"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name as served by vLLM (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num", type=int, default=NUM_EXAMPLES,
        help=f"Number of examples (default: {NUM_EXAMPLES})",
    )
    parser.add_argument(
        "--url", default=VLLM_BASE_URL,
        help=f"vLLM base URL (default: {VLLM_BASE_URL})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY,
        help=f"Max concurrent requests to vLLM (default: {CONCURRENCY})",
    )
    args = parser.parse_args()

    print(f"Using vLLM model: {args.model} at {args.url}")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
