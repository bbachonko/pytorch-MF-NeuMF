from collections import Counter

from recommender.data.feature_eng.book_type_enums import TagClassification, ContextType
from recommender.utils.logger import setup_logger

logger = setup_logger(__name__)

def count_unclassified_tags(classified_tags: list[TagClassification]) -> None:
    counter = Counter(tag.context for tag in classified_tags)
    total = len(classified_tags)
    unclassified = counter.get(ContextType.UNCLASSIFIED, 0)
    logger.info(f"All tags: {total}")
    logger.info(f"Number of tags marked as 'unclassified': {unclassified}")
    logger.info(f"%: {unclassified / total * 100:.2f}%")
