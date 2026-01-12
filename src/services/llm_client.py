"""
GRAFF LLM Client - Three-Pass Chapter Analysis Pipeline

This module orchestrates the GRAFF pipeline in three passes:
- Pass 1: Structure (sections, summary, entities, keywords)
- Pass 2: Propositions (atomic facts within structure)
- Pass 3: Key Takeaways (synthesis across propositions)
"""

from typing import Dict, Optional, Any, Callable, List
import json
from pathlib import Path
from ..utils.logging_config import get_logger
from .anthropic_client import call_anthropic_structured as call_llm_structured, LLMConfigurationError, LLMAPIError
from ..models import (
    ChapterAnalysis,
    Phase1Comprehension,
    Phase2Output,
    Section,
    Entity,
    Proposition,
    KeyTakeaway
)

logger = get_logger(__name__)

# LLM configuration
ANALYSIS_TEMPERATURE = 0.15

# Prompt file paths
PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"
PASS1_PROMPT_PATH = PROMPT_DIR / "pass1_structure.txt"
PASS2_PROMPT_PATH = PROMPT_DIR / "pass2_propositions.txt"
PASS3_PROMPT_PATH = PROMPT_DIR / "pass3_takeaways.txt"


def _load_prompt(prompt_path: Path) -> str:
    """Load system prompt from file."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def run_pass1_structure(
    text: str,
    book_id: str,
    chapter_id: str,
    chapter_title: str,
    progress_callback: Optional[Callable] = None
) -> Phase1Comprehension:
    """
    Pass 1: Extract chapter structure.

    Extracts:
    - Sections (hierarchical outline)
    - Summary (one-paragraph overview)
    - Key entities (people, concepts, organizations)
    - Keywords (domain terminology)

    Args:
        text: Chapter text
        book_id: Book identifier
        chapter_id: Chapter identifier
        chapter_title: Chapter title
        progress_callback: Optional callback(phase, message, **kwargs)

    Returns:
        Phase1Comprehension with structure data
    """
    logger.info(f"Pass 1 starting: {chapter_id} - {chapter_title}")

    def notify(message: str, **kwargs):
        if progress_callback:
            progress_callback("pass-1", message, **kwargs)
        logger.info(f"pass-1: {message}")

    notify("Extracting chapter structure...")

    system_prompt = _load_prompt(PASS1_PROMPT_PATH)

    user_prompt = f"""Chapter Title: {chapter_title}
Book ID: {book_id}
Chapter ID: {chapter_id}

Chapter Text:
{text}

Extract the structure and respond with JSON."""

    response_dict = call_llm_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=ANALYSIS_TEMPERATURE,
        json_schema=None,
        max_tokens=4000
    )

    # Parse response
    sections = [Section.model_validate(s) for s in response_dict.get("sections", [])]
    key_entities = [Entity.model_validate(e) for e in response_dict.get("key_entities", [])]
    keywords = response_dict.get("keywords", [])
    summary = response_dict.get("summary", "")

    phase1 = Phase1Comprehension(
        summary=summary,
        sections=sections,
        key_entities=key_entities,
        keywords=keywords
    )

    # Send stats with progress
    notify(
        f"Found {len(sections)} sections, {len(key_entities)} entities, {len(keywords)} keywords",
        sections=len(sections),
        entities=len(key_entities),
        keywords=len(keywords),
        latest=sections[0].title if sections else None
    )
    logger.info(f"Pass 1 complete: {len(sections)} sections")

    return phase1


def run_pass2_propositions(
    text: str,
    chapter_id: str,
    structure: Phase1Comprehension,
    progress_callback: Optional[Callable] = None
) -> List[Proposition]:
    """
    Pass 2: Extract propositions within structure.

    Uses structure from Pass 1 to tag propositions with unit_ids.

    Args:
        text: Chapter text
        chapter_id: Chapter identifier
        structure: Phase1Comprehension from Pass 1
        progress_callback: Optional callback(phase, message, **kwargs)

    Returns:
        List of Proposition objects
    """
    logger.info(f"Pass 2 starting: {chapter_id}")

    def notify(message: str, **kwargs):
        if progress_callback:
            progress_callback("pass-2", message, **kwargs)
        logger.info(f"pass-2: {message}")

    notify("Extracting propositions...", sections=len(structure.sections))

    system_prompt = _load_prompt(PASS2_PROMPT_PATH)

    # Format structure for context
    sections_json = json.dumps([s.model_dump() for s in structure.sections], indent=2)

    user_prompt = f"""Chapter ID: {chapter_id}

STRUCTURE (from Pass 1):
{sections_json}

CHAPTER TEXT:
{text}

Extract all propositions and respond with JSON."""

    response_dict = call_llm_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=ANALYSIS_TEMPERATURE,
        json_schema=None,
        max_tokens=16000
    )

    # Parse propositions
    propositions = [Proposition.model_validate(p) for p in response_dict.get("propositions", [])]

    # Fix chapter_id if needed
    for prop in propositions:
        if prop.chapter_id != chapter_id:
            prop.chapter_id = chapter_id

    # Send stats with sample proposition
    sample_prop = propositions[0].proposition_text if propositions else None
    notify(
        f"Extracted {len(propositions)} propositions",
        sections=len(structure.sections),
        propositions=len(propositions),
        latest=sample_prop[:100] + "..." if sample_prop and len(sample_prop) > 100 else sample_prop
    )
    logger.info(f"Pass 2 complete: {len(propositions)} propositions")

    return propositions


def run_pass3_takeaways(
    chapter_id: str,
    structure: Phase1Comprehension,
    propositions: List[Proposition],
    progress_callback: Optional[Callable] = None
) -> List[KeyTakeaway]:
    """
    Pass 3: Synthesize key takeaways from propositions.

    Uses both structure and ALL propositions to create higher-order insights
    that can bridge across sections.

    Args:
        chapter_id: Chapter identifier
        structure: Phase1Comprehension from Pass 1
        propositions: List of Propositions from Pass 2
        progress_callback: Optional callback(phase, message, **kwargs)

    Returns:
        List of KeyTakeaway objects
    """
    logger.info(f"Pass 3 starting: {chapter_id}")

    def notify(message: str, **kwargs):
        if progress_callback:
            progress_callback("pass-3", message, **kwargs)
        logger.info(f"pass-3: {message}")

    notify(
        "Synthesizing key takeaways...",
        sections=len(structure.sections),
        propositions=len(propositions)
    )

    system_prompt = _load_prompt(PASS3_PROMPT_PATH)

    # Format structure
    sections_json = json.dumps([s.model_dump() for s in structure.sections], indent=2)

    # Format propositions (include id and text for synthesis)
    props_summary = "\n".join([
        f"- [{p.proposition_id}] ({p.unit_id}, {p.bloom_level}): {p.proposition_text}"
        for p in propositions
    ])

    user_prompt = f"""Chapter ID: {chapter_id}

STRUCTURE:
{sections_json}

PROPOSITIONS ({len(propositions)} total):
{props_summary}

Synthesize key takeaways and respond with JSON."""

    response_dict = call_llm_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=ANALYSIS_TEMPERATURE,
        json_schema=None,
        max_tokens=8000
    )

    # Parse takeaways
    takeaways = [KeyTakeaway.model_validate(t) for t in response_dict.get("key_takeaways", [])]

    # Fix chapter_id if needed
    for takeaway in takeaways:
        if takeaway.chapter_id != chapter_id:
            takeaway.chapter_id = chapter_id

    # Send stats with sample takeaway
    sample_takeaway = takeaways[0].text if takeaways else None
    notify(
        f"Synthesized {len(takeaways)} takeaways",
        sections=len(structure.sections),
        propositions=len(propositions),
        takeaways=len(takeaways),
        latest=sample_takeaway[:100] + "..." if sample_takeaway and len(sample_takeaway) > 100 else sample_takeaway
    )
    logger.info(f"Pass 3 complete: {len(takeaways)} takeaways")

    return takeaways


def run_three_pass_analysis(
    text: str,
    book_id: str,
    chapter_id: str,
    chapter_title: str,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> ChapterAnalysis:
    """
    Run the complete three-pass GRAFF pipeline.

    Pass 1: Structure → Pass 2: Propositions → Pass 3: Takeaways

    Each pass builds on the previous, with Pass 3 having full context
    of both structure and all propositions for cross-section synthesis.

    Args:
        text: Chapter text to analyze
        book_id: Book identifier
        chapter_id: Chapter identifier
        chapter_title: Chapter title
        progress_callback: Optional callback(phase, message)

    Returns:
        ChapterAnalysis: Complete validated analysis
    """
    logger.info(f"Three-pass analysis starting: {chapter_id} - {chapter_title}")

    # Pass 1: Structure
    structure = run_pass1_structure(
        text=text,
        book_id=book_id,
        chapter_id=chapter_id,
        chapter_title=chapter_title,
        progress_callback=progress_callback
    )

    # Pass 2: Propositions (with structure context)
    propositions = run_pass2_propositions(
        text=text,
        chapter_id=chapter_id,
        structure=structure,
        progress_callback=progress_callback
    )

    # Pass 3: Takeaways (with structure + propositions context)
    takeaways = run_pass3_takeaways(
        chapter_id=chapter_id,
        structure=structure,
        propositions=propositions,
        progress_callback=progress_callback
    )

    # Assemble Phase2Output
    phase2 = Phase2Output(
        propositions=propositions,
        key_takeaways=takeaways
    )

    # Assemble complete analysis
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
    logger.info(f"Three-pass analysis complete: {len(structure.sections)} sections, "
                f"{len(propositions)} propositions, {len(takeaways)} takeaways")
    logger.info(f"Bloom distribution: {bloom_dist}")

    return chapter


# Backward compatibility aliases
def run_unified_analysis(*args, **kwargs) -> ChapterAnalysis:
    """Alias for run_three_pass_analysis."""
    return run_three_pass_analysis(*args, **kwargs)


def process_chapter(
    text: str,
    book_id: str,
    chapter_id: str,
    chapter_title: str
) -> ChapterAnalysis:
    """Legacy wrapper for run_three_pass_analysis."""
    return run_three_pass_analysis(
        text=text,
        book_id=book_id,
        chapter_id=chapter_id,
        chapter_title=chapter_title,
        progress_callback=None
    )
