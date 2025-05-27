from pydantic import BaseModel
from enum import Enum


class ContextType(str, Enum):
    OBJECTIVE = "objective"
    UNCLASSIFIED = "unclassified"


class GenreType(str, Enum):
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    ROMANCE = "romance"
    MYSTERY = "mystery"
    THRILLER = "thriller"
    HORROR = "horror"
    YOUNG_ADULT = "young_adult"
    CHILDRENS = "childrens"
    NONFICTION = "nonfiction"
    BIOGRAPHY = "biography"
    CLASSICS = "classics"
    HISTORY = "history"
    PHILOSOPHY = "philosophy"
    RELIGION = "religion"
    SCIENCE = "science"
    LGBTQ = "lgbtq"
    DIVERSITY = "diversity"
    SCHOOL = "school"
    OTHER = "other"


class FormType(str, Enum):
    POETRY = "poetry"
    SHORT_STORIES = "short_stories"
    GRAPHIC_NOVEL = "graphic_novel"
    ESSAY = "essay"
    DRAMA = "drama"
    COMIC = "comic"
    NOVELLA = "novella"
    WEBNOVEL = "webnovel"
    ANTHOLOGY = "anthology"
    OTHER = "other"


class TagClassification(BaseModel):
    old_tag: str
    context: ContextType
    genre: Optional[GenreType] = None
    form: Optional[FormType] = None