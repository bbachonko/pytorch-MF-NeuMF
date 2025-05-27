from openai import OpenAI
from tqdm.auto import tqdm
import json
from pydantic import ValidationError
from dotenv import load_dotenv

from .prompt_templates import WRAPPER_SCHEMA, PROMPT_TEMPLATE, GENRE_KEYS, FORM_KEYS, FEW_SHOT_BLOCK
from .book_type_enums import TagClassification


OPENAI_MODEL = "gpt-4o-mini"  # Most price reasonable and fastest
load_dotenv()
client = OpenAI()


def _ask_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a book-tag classification assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TagClassificationResponse",
                "schema": WRAPPER_SCHEMA,
                "strict": True,
            },
        },
        temperature=0,
    )
    return response.choices[0].message.content


def classify_tags(tags: list[str], batch_size: int = 250) -> list[TagClassification]:
    """Classify tags and return Pydantic models.

    TL;DR: Max batch_size for gpt4o-mini is 128k. 500 batch size is reasonable maximum.
    """
    results: list[TagClassification] = []
    for start in tqdm(range(0, len(tags), batch_size), desc="Tag batches"):
        batch = tags[start : start + batch_size]

        # Prompt prep
        prompt = PROMPT_TEMPLATE.format(
            genres=", ".join(GENRE_KEYS),
            forms=", ".join(FORM_KEYS),
            schema=json.dumps(WRAPPER_SCHEMA, indent=2),
            examples=FEW_SHOT_BLOCK,
            tags_block="\n".join(f"- {t}" for t in batch),
        )

        raw = _ask_model(prompt)
        payload = json.loads(raw)
        records = payload["results"]

        for rec in tqdm(records, desc="Records", leave=False):  # Pydantic instacne creation
            try:
                results.append(TagClassification(**rec))
            except ValidationError as err:
                raise ValueError(f"Invalid record returned: {rec}\n{err}") from err

    return results
