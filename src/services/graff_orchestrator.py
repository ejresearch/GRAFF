"""
GRAFF Orchestrator - Three-Pass Chapter Analysis Pipeline

Manages the complete GRAFF workflow:
1. Pass 1: Structure (sections, summary, entities, keywords)
2. Pass 2: Propositions (atomic facts within structure)
3. Pass 3: Key Takeaways (synthesis across propositions)
4. Validate and persist to database
"""

from typing import Optional, Callable
from ..utils.logging_config import get_logger
from .llm_client import run_three_pass_analysis
from ..models import ChapterAnalysis
from ..db import save_chapter_analysis, init_database
import uuid

logger = get_logger(__name__)


class DigestError(Exception):
    """Base exception for digest pipeline errors."""
    pass


class AnalysisError(DigestError):
    """Exception raised when analysis fails."""
    def __init__(self, message: str, original_error: Exception = None):
        self.original_error = original_error
        super().__init__(f"Analysis failed: {message}")


class StorageError(DigestError):
    """Exception raised when storage fails."""
    pass


def digest_chapter_graff(
    text: str,
    book_id: str = "unknown_book",
    chapter_title: str = "Untitled Chapter",
    chapter_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> ChapterAnalysis:
    """
    Process a chapter through the GRAFF three-pass analysis pipeline.

    Workflow:
    1. Pass 1: Extract structure (sections, summary, entities, keywords)
    2. Pass 2: Extract propositions (using structure for tagging)
    3. Pass 3: Synthesize takeaways (using structure + all propositions)
    4. Validate with Pydantic
    5. Save to database

    Args:
        text: The chapter text to analyze
        book_id: Book identifier (default: "unknown_book")
        chapter_title: Chapter title (default: "Untitled Chapter")
        chapter_id: Chapter ID (auto-generated if not provided)
        progress_callback: Optional callback(phase, message) for progress updates

    Returns:
        ChapterAnalysis: Complete validated analysis

    Raises:
        AnalysisError: If any pass fails
        StorageError: If database persistence fails
    """
    logger.info(f"Starting GRAFF pipeline for: {chapter_title}")

    # Auto-generate chapter_id if not provided
    if not chapter_id:
        chapter_id = f"ch_{uuid.uuid4().hex[:8]}"
        logger.info(f"Auto-generated chapter_id: {chapter_id}")

    def notify(phase: str, message: str, **kwargs):
        """Send progress update via callback with optional stats."""
        if progress_callback:
            progress_callback(phase, message, **kwargs)
        logger.info(f"{phase}: {message}")

    try:
        # Initialize database
        try:
            init_database()
        except Exception as e:
            logger.warning(f"Database init: {e}")

        # Run three-pass analysis
        notify("pipeline", "Starting three-pass analysis...")

        try:
            chapter = run_three_pass_analysis(
                text=text,
                book_id=book_id,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                progress_callback=notify
            )

            # Log statistics
            bloom_dist = chapter.get_bloom_distribution()
            logger.info(f"Analysis complete: {len(chapter.phase1.sections)} sections, "
                       f"{len(chapter.phase2.propositions)} propositions, "
                       f"{len(chapter.phase2.key_takeaways)} takeaways")
            logger.info(f"Bloom distribution: {bloom_dist}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise AnalysisError(f"Three-pass analysis failed: {str(e)}", e)

        # Save to database
        notify("storage", "Saving to database...")
        try:
            success = save_chapter_analysis(chapter)
            if not success:
                raise StorageError("Database save returned False")
            logger.info(f"Chapter {chapter_id} saved successfully")
            notify("storage", "Saved")
        except Exception as e:
            logger.error(f"Storage failed: {e}", exc_info=True)
            raise StorageError(f"Failed to save: {str(e)}")

        # Done
        notify("completed", f"Done! {chapter.get_proposition_count()} propositions, "
                           f"{len(chapter.phase2.key_takeaways)} takeaways")

        return chapter

    except (AnalysisError, StorageError):
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise DigestError(f"Unexpected error: {str(e)}")


def quick_digest(
    text: str,
    chapter_title: str = "Untitled Chapter",
    book_id: str = "test_book"
) -> ChapterAnalysis:
    """Quick wrapper without progress callbacks."""
    return digest_chapter_graff(
        text=text,
        book_id=book_id,
        chapter_title=chapter_title,
        progress_callback=None
    )
