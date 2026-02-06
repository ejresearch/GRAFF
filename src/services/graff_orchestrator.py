"""
GRAFF Orchestrator - Three-Pass Chapter Analysis Pipeline

Manages the complete GRAFF workflow:
1. Pass 1: Structure (sections, summary, entities, keywords) → SAVE
2. Pass 2: Propositions (atomic facts within structure) → SAVE per section
3. Pass 3: Key Takeaways (synthesis across propositions) → SAVE
4. Validate and persist to database

Data is saved incrementally after each pass to prevent data loss.
"""

from typing import Optional, Callable, List
from ..utils.logging_config import get_logger
from .llm_client import (
    run_pass1_structure,
    run_pass2_propositions,
    run_pass3_takeaways,
    ANALYSIS_TEMPERATURE,
    _load_prompt,
    _extract_section_propositions,
    PASS2_SECTION_PROMPT_PATH
)
from ..models import ChapterAnalysis, Phase2Output, Proposition
from ..db import (
    init_database,
    save_chapter_phase1,
    save_propositions,
    save_takeaways,
    delete_chapter
)
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

    Workflow (with incremental saves):
    1. Pass 1: Extract structure → SAVE immediately
    2. Pass 2: Extract propositions per section → SAVE after each section
    3. Pass 3: Synthesize takeaways → SAVE immediately
    4. Return complete ChapterAnalysis

    Data is saved incrementally so partial results are preserved if later steps fail.

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
        chapter_id = chapter_title  # Use title as ID for better readability
        logger.info(f"Using chapter_id: {chapter_id}")

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

        notify("pipeline", "Starting three-pass analysis with incremental saves...")

        # =====================================================================
        # PASS 1: Structure
        # =====================================================================
        try:
            notify("pass-1", "Extracting chapter structure...")
            structure = run_pass1_structure(
                text=text,
                book_id=book_id,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                progress_callback=notify
            )
        except Exception as e:
            logger.error(f"Pass 1 failed: {e}", exc_info=True)
            raise AnalysisError(f"Pass 1 (structure) failed: {str(e)}", e)

        # SAVE Pass 1 immediately
        notify("storage", "Saving structure to database...")
        try:
            success = save_chapter_phase1(
                chapter_id=chapter_id,
                book_id=book_id,
                chapter_title=chapter_title,
                phase1=structure
            )
            if not success:
                raise StorageError("Failed to save Phase 1")
            notify("storage", f"Structure saved ({len(structure.sections)} sections)")
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Failed to save Pass 1: {e}")
            raise StorageError(f"Failed to save structure: {str(e)}")

        # =====================================================================
        # PASS 2: Propositions (with per-section saves)
        # =====================================================================
        notify("pass-2", f"Extracting propositions from {len(structure.sections)} sections...")

        all_propositions: List[Proposition] = []
        system_prompt = _load_prompt(PASS2_SECTION_PROMPT_PATH)
        total_sections = len(structure.sections)

        for i, section in enumerate(structure.sections):
            section_num = i + 1
            notify(
                "pass-2",
                f"Section {section_num}/{total_sections}: {section.title}",
                sections=total_sections,
                current_section=section_num,
                propositions=len(all_propositions),
                latest=section.title
            )

            try:
                section_props = _extract_section_propositions(
                    text=text,
                    chapter_id=chapter_id,
                    section=section,
                    all_sections=structure.sections,
                    system_prompt=system_prompt
                )

                # Renumber with global index
                start_idx = len(all_propositions)
                for j, prop in enumerate(section_props):
                    prop.proposition_id = f"{chapter_id}_{prop.unit_id}_p{start_idx + j + 1:03d}"
                    prop.chapter_id = chapter_id

                all_propositions.extend(section_props)

                # SAVE this section's propositions immediately
                if section_props:
                    save_propositions(chapter_id, section_props)

                logger.info(f"Section {section.unit_id}: {len(section_props)} propositions (total: {len(all_propositions)})")

            except Exception as e:
                logger.error(f"Failed to extract section {section.unit_id}: {e}")
                # Continue with other sections - don't fail the whole pipeline

        notify(
            "pass-2",
            f"Extracted {len(all_propositions)} propositions from {total_sections} sections",
            sections=total_sections,
            propositions=len(all_propositions)
        )

        # =====================================================================
        # PASS 3: Takeaways
        # =====================================================================
        try:
            notify("pass-3", "Synthesizing key takeaways...")
            takeaways = run_pass3_takeaways(
                chapter_id=chapter_id,
                structure=structure,
                propositions=all_propositions,
                progress_callback=notify
            )

            # Validate takeaway references
            valid_prop_ids = {p.proposition_id for p in all_propositions}
            valid_unit_ids = {s.unit_id for s in structure.sections}

            for takeaway in takeaways:
                original_count = len(takeaway.proposition_ids)
                takeaway.proposition_ids = [pid for pid in takeaway.proposition_ids if pid in valid_prop_ids]
                if len(takeaway.proposition_ids) < original_count:
                    removed = original_count - len(takeaway.proposition_ids)
                    logger.warning(f"Takeaway {takeaway.takeaway_id}: removed {removed} invalid proposition refs")

                if takeaway.unit_id and takeaway.unit_id not in valid_unit_ids:
                    logger.warning(f"Takeaway {takeaway.takeaway_id}: invalid unit_id, setting to None")
                    takeaway.unit_id = None

        except Exception as e:
            logger.error(f"Pass 3 failed: {e}", exc_info=True)
            # Don't fail entirely - we still have structure and propositions saved
            logger.warning("Continuing without takeaways due to Pass 3 failure")
            takeaways = []

        # SAVE takeaways
        if takeaways:
            notify("storage", "Saving takeaways to database...")
            try:
                save_takeaways(chapter_id, takeaways)
                notify("storage", f"Takeaways saved ({len(takeaways)} takeaways)")
            except Exception as e:
                logger.error(f"Failed to save takeaways: {e}")
                # Don't fail - takeaways are optional

        # =====================================================================
        # Assemble final ChapterAnalysis object
        # =====================================================================
        phase2 = Phase2Output(
            propositions=all_propositions,
            key_takeaways=takeaways
        )

        chapter = ChapterAnalysis(
            schema_version="1.0",
            book_id=book_id,
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            phase1=structure,
            phase2=phase2
        )

        # Log statistics
        bloom_dist = chapter.get_bloom_distribution()
        logger.info(f"Analysis complete: {len(structure.sections)} sections, "
                   f"{len(all_propositions)} propositions, "
                   f"{len(takeaways)} takeaways")
        logger.info(f"Bloom distribution: {bloom_dist}")

        notify("completed", f"Done! {len(all_propositions)} propositions, {len(takeaways)} takeaways")

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
