from 


GENRE_KEYS = [g.value for g in GenreType]
FORM_KEYS = [f.value for f in FormType]

CLASS_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "old_tag": {"type": "string"},
            "context": {"enum": [e.value for e in ContextType]},
            "genre": {"enum": GENRE_KEYS + [None]},
            "form": {"enum": FORM_KEYS + [None]},
        },
        "required": ["old_tag", "context", "genre", "form"],
        "additionalProperties": False,
    },
    "additionalItems": False,
}

WRAPPER_SCHEMA = {
    "type": "object",
    "properties": {
        "results": CLASS_ARRAY_SCHEMA
    },
    "required": ["results"],
    "additionalProperties": False,
}


# ─────────────────────────── FEW-SHOT EXAMPLES ───────────────────────────

FEW_SHOT = [
    {"old_tag": "science fiction", "context": "objective", "genre": "science_fiction", "form": None},
    {"old_tag": "graphic novel fantasy",  "context": "objective", "genre": "fantasy", "form": "graphic_novel"},
    {"old_tag": "poetry", "context": "objective", "genre": None, "form": "poetry"},
    {"old_tag": "lgbtq romance", "context": "objective", "genre": "lgbtq", "form": None},
    {"old_tag": "dnf", "context": "unclassified","genre": None, "form": None},
]
FEW_SHOT_BLOCK = json.dumps(FEW_SHOT, ensure_ascii=False, indent=2)

# ─────────────────────────── PROMPT TEMPLATE ───────────────────────────

PROMPT_TEMPLATE = """
You are a book-tag classification assistant.
Given a list of raw tags, return a **JSON array** of objects that
**validates** against the <schema>.  Each object must contain:

  1. old_tag  – original string
  2. context  – "objective" | "unclassified"
  3. genre    – key or null (see list)
  4. form     – key or null  (see list)

Allowed genre keys: {genres}
Allowed form  keys: {forms}

<schema>
{schema}
</schema>

<examples>
{examples}
</examples>

Classify these tags:
{tags_block}
""".strip()