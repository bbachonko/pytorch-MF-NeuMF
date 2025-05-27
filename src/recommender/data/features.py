import pandas as pd
import numpy as np
from datetime import datetime


def build_embedding_indices(
    book_tags_df: pd.DataFrame,
    books_df: pd.DataFrame,
    tag_vocab: list[int],
    generalized_tag_column: str = 'generalized_tags',
) -> dict[int, dict[str, list[int]]]:
    """
    Build index lists of tag IDs and generalized genre/form IDs per book.

    Actual embedding will be used directly in model implemetnation since its redundant to use eg. word2vec here.

    Returns:
        A dictionary mapping book_id → {
            'tags': [tag indices],
            'genres': [genre/form indices]
        }
    """
    tag_to_idx = {tag: i for i, tag in enumerate(tag_vocab)}

    # --- Tags (from user-tag interaction) ---
    book_tag_map = (
        book_tags_df[book_tags_df['tag_id'].isin(tag_vocab)]
        .groupby('goodreads_book_id')['tag_id']
        .apply(lambda tags: [tag_to_idx[tag] for tag in tags if tag in tag_to_idx])
        .to_dict()
    )

    # --- Generalized genres/forms (from LLM) ---
    all_labels = set()
    book_genre_map = {}

    for book_id, tags in books_df.set_index('book_id')[generalized_tag_column].dropna().items():
        tag_list = str(tags).replace(',', ' ').replace(';', ' ').split()
        tag_list = [t.lower().strip() for t in tag_list if t]
        book_genre_map[book_id] = tag_list
        all_labels.update(tag_list)

    genre_vocab = sorted(all_labels)
    genre_to_idx = {g: i for i, g in enumerate(genre_vocab)}

    book_genre_idx_map = {
        book_id: [genre_to_idx[g] for g in genres if g in genre_to_idx]
        for book_id, genres in book_genre_map.items()
    }

    # --- Combine ---
    all_books = set(book_tags_df['goodreads_book_id']) | set(book_genre_map.keys())
    output = {}
    for book_id in all_books:
        output[book_id] = {
            'tags': book_tag_map.get(book_id, []),
            'genres': book_genre_idx_map.get(book_id, [])
        }

    return output, tag_to_idx, genre_to_idx


def process_language(
    books_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One-hot encode `language_code` per book.

    Returns DataFrame indexed by book_id.
    """
    df = books_df.copy()
    df['language_code'].fillna('unknown', inplace=True)
    lang_ohe = pd.get_dummies(df['language_code'], prefix='lang')
    lang_ohe.index = df['book_id']
    return lang_ohe


def process_publication_year(
    books_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize `original_publication_year` (imputing median) and bucket into eras.

    Returns DataFrame indexed by book_id with columns ['year_norm', 'era_*'].
    """
    df = books_df.copy()
    df['original_publication_year'].fillna(df['original_publication_year'].median(), inplace=True)
    years = df['original_publication_year']
    mean, std = years.mean(), years.std(ddof=0)

    year_norm = (years - mean) / std
    eras = pd.cut(
        years,
        bins=[-np.inf, 1900, 1950, 2000, np.inf],
        labels=['pre1900', '1900_1950', '1950_2000', '2000plus']
    )
    era_ohe = pd.get_dummies(eras, prefix='era')

    year_df = pd.DataFrame({'year_norm': year_norm}, index=df['book_id'])
    era_df = era_ohe.set_index(df['book_id'])
    return year_df.join(era_df)


def process_publication_age(
    books_df: pd.DataFrame,
    reference_year: int = None,
) -> pd.DataFrame:
    """
    Compute book age in years from `original_publication_year`.

    Args:
        reference_year: Year to subtract from (default: current year).

    Returns:
        DataFrame indexed by book_id with column ['age_years'].
    """
    if reference_year is None:
        reference_year = datetime.today().year
    df = books_df.copy()
    df['original_publication_year'].fillna(reference_year, inplace=True)
    age_years = reference_year - df['original_publication_year']
    return pd.DataFrame({'age_years': age_years}, index=df['book_id'])


def build_item_content_matrix(
    generalized_tags: pd.DataFrame,
    books_df: pd.DataFrame,
    # book_tags_df: pd.DataFrame,
    # tags_df: pd.DataFrame,
    reference_date: str | datetime | None = None,
) -> pd.DataFrame:
    """
    Assemble core content features into one DataFrame indexed by *book_id*.
    Ensures all outputs are numeric — categorical columns are one-hot encoded.
    """
    # Year features
    year_feat: pd.DataFrame = process_publication_year(books_df)

    # Age-in-years feature
    ref_year: int | None = None
    if reference_date is not None:
        ref_year = (
            datetime.fromisoformat(reference_date).year
            if isinstance(reference_date, str)
            else reference_date.year
        )
    age_feat: pd.DataFrame = process_publication_age(books_df, reference_year=ref_year)

    # Language one-hot
    lang_feat: pd.DataFrame = process_language(books_df)

    # Combine features — index must be book_id
    combined_df = pd.concat([generalized_tags, year_feat, age_feat, lang_feat], axis=1)
    combined_df = combined_df.sort_index().fillna(0.0)

    # Ensure no non-numeric columns (one-hot encode if needed)
    non_numeric_cols = combined_df.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        combined_df = pd.get_dummies(combined_df, columns=non_numeric_cols)

    # Final check: only numeric
    assert combined_df.select_dtypes(include=['object', 'category']).empty, \
        f"Non-numeric columns remain: {list(non_numeric_cols)}"

    return combined_df
