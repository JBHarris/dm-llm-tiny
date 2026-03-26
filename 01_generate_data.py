"""
Step 1: Generate synthetic D&D training data using Claude API.

Generates instruction/response pairs for:
- Character creation (NPCs, villains, party members)
- Quest hooks and full quest outlines
- Dialog lines and conversations
- Location/dungeon descriptions
- Encounter design

Output: data/dnd_training.jsonl
"""

import json
import os
import random
import time
from pathlib import Path

from anthropic import Anthropic

# --- Configuration ---
NUM_EXAMPLES = 500  # Total examples to generate (increase for better results)
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "dnd_training.jsonl"
MODEL = "claude-sonnet-4-20250514"

# --- Prompt categories ---
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


def generate_example(client: Anthropic, category: str, info: dict) -> dict:
    """Generate a single training example."""
    base_prompt = random.choice(info["prompts"])
    user_prompt = build_prompt(category, base_prompt)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=info["system"],
        messages=[{"role": "user", "content": user_prompt}],
    )

    assistant_text = response.content[0].text

    # Format as chat for TinyLlama's chat template
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a creative Dungeons & Dragons dungeon master's assistant. "
                    "You help create memorable characters, compelling quests, immersive dialog, "
                    "vivid locations, and exciting encounters for D&D campaigns."
                ),
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ],
        "category": category,
    }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY in your .env file")

    client = Anthropic(api_key=api_key)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build weighted category list
    weighted_categories = []
    for cat, info in CATEGORIES.items():
        weighted_categories.extend([(cat, info)] * info["weight"])

    existing = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing = sum(1 for _ in f)
        print(f"Found {existing} existing examples, generating {NUM_EXAMPLES - existing} more...")

    examples_needed = NUM_EXAMPLES - existing
    if examples_needed <= 0:
        print(f"Already have {existing} examples. Delete {OUTPUT_FILE} to regenerate.")
        return

    print(f"Generating {examples_needed} D&D training examples...")
    category_counts = {}

    with open(OUTPUT_FILE, "a") as f:
        for i in range(examples_needed):
            cat, info = random.choice(weighted_categories)
            category_counts[cat] = category_counts.get(cat, 0) + 1

            try:
                example = generate_example(client, cat, info)
                f.write(json.dumps(example) + "\n")
                f.flush()

                if (i + 1) % 10 == 0:
                    print(f"  [{i + 1}/{examples_needed}] Generated ({cat})")

            except Exception as e:
                print(f"  Error on example {i + 1}: {e}")
                time.sleep(5)
                continue

            # Rate limiting: ~50 requests/min for Sonnet
            time.sleep(1.2)

    print(f"\nDone! Generated {examples_needed} examples.")
    print(f"Category breakdown: {category_counts}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
