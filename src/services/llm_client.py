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
PASS2_SECTION_PROMPT_PATH = PROMPT_DIR / "pass2_section.txt"
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


def _find_section_text(full_text: str, section: Section, all_sections: List[Section]) -> str:
    """
    Extract just the text for a specific section from the full chapter.

    Finds the section heading and extracts text until the next section heading.
    Falls back to full text if section can't be located.
    """
    import re

    # Find this section's position
    # Try exact title match first, then fuzzy
    section_pattern = re.escape(section.title)
    match = re.search(rf'(?:^|\n).*?{section_pattern}', full_text, re.IGNORECASE)

    if not match:
        # Can't find section, return truncated full text
        logger.warning(f"Could not locate section '{section.title}' in text, using truncated text")
        return full_text[:15000]  # ~15k chars max fallback

    start_pos = match.start()

    # Find the next section's position
    end_pos = len(full_text)

    # Get sections that come after this one
    current_idx = next((i for i, s in enumerate(all_sections) if s.unit_id == section.unit_id), -1)

    if current_idx >= 0 and current_idx < len(all_sections) - 1:
        # Look for the next section's title
        next_section = all_sections[current_idx + 1]
        next_pattern = re.escape(next_section.title)
        next_match = re.search(rf'(?:^|\n).*?{next_pattern}', full_text[start_pos + 1:], re.IGNORECASE)
        if next_match:
            end_pos = start_pos + 1 + next_match.start()

    section_text = full_text[start_pos:end_pos].strip()

    # Return full section text - chunking handled by caller if needed
    return section_text


def _chunk_text(text: str, max_chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks at paragraph boundaries.

    Args:
        text: Text to split
        max_chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks for context

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds limit, save current chunk and start new
        if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from end of previous
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def _extract_chunk_propositions(
    chunk_text: str,
    chapter_id: str,
    section: Section,
    system_prompt: str,
    chunk_num: int = 1,
    total_chunks: int = 1
) -> List[Proposition]:
    """
    Extract propositions from a single chunk of section text.
    """
    user_prompt = f"""Chapter ID: {chapter_id}

TARGET SECTION:
- unit_id: {section.unit_id}
- title: {section.title}
- level: {section.level}

SECTION TEXT (part {chunk_num}/{total_chunks}):
{chunk_text}

Extract propositions from this section text.
Respond with JSON only."""

    try:
        response_dict = call_llm_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=ANALYSIS_TEMPERATURE,
            json_schema=None,
            max_tokens=8000  # Smaller chunks need fewer output tokens
        )

        propositions = [Proposition.model_validate(p) for p in response_dict.get("propositions", [])]

        # Fix chapter_id and unit_id if needed
        for prop in propositions:
            if prop.chapter_id != chapter_id:
                prop.chapter_id = chapter_id
            if prop.unit_id != section.unit_id:
                prop.unit_id = section.unit_id

        return propositions

    except Exception as e:
        logger.error(f"Failed to extract propositions from chunk {chunk_num} of section {section.unit_id}: {e}")
        return []


def _extract_section_propositions(
    text: str,
    chapter_id: str,
    section: Section,
    all_sections: List[Section],
    system_prompt: str,
    max_retries: int = 2
) -> List[Proposition]:
    """
    Extract propositions from a single section, chunking if necessary.

    Args:
        text: Full chapter text
        chapter_id: Chapter identifier
        section: Section to extract from
        all_sections: All sections (for finding boundaries)
        system_prompt: Loaded prompt template
        max_retries: Number of retries with increased tokens on truncation

    Returns:
        List of Proposition objects for this section
    """
    # Extract just this section's text
    section_text = _find_section_text(text, section, all_sections)
    logger.debug(f"Section '{section.title}' text: {len(section_text)} chars")

    # Chunk large sections to avoid output truncation
    # 4000 chars input ≈ 30-50 propositions ≈ 6000-8000 tokens output
    chunks = _chunk_text(section_text, max_chunk_size=4000, overlap=100)

    if len(chunks) > 1:
        logger.info(f"Section '{section.title}' split into {len(chunks)} chunks")

    all_propositions = []
    seen_texts = set()  # Deduplicate propositions from overlapping chunks

    for i, chunk in enumerate(chunks):
        chunk_props = _extract_chunk_propositions(
            chunk_text=chunk,
            chapter_id=chapter_id,
            section=section,
            system_prompt=system_prompt,
            chunk_num=i + 1,
            total_chunks=len(chunks)
        )

        # Deduplicate based on proposition text
        for prop in chunk_props:
            text_key = prop.proposition_text.strip().lower()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_propositions.append(prop)

    logger.info(f"Section {section.unit_id}: {len(all_propositions)} propositions from {len(chunks)} chunk(s)")
    return all_propositions


def run_pass2_propositions(
    text: str,
    chapter_id: str,
    structure: Phase1Comprehension,
    progress_callback: Optional[Callable] = None
) -> List[Proposition]:
    """
    Pass 2: Extract propositions section by section.

    Processes each section from Pass 1 independently to avoid token limits.
    This scales to any chapter size.

    Args:
        text: Chapter text
        chapter_id: Chapter identifier
        structure: Phase1Comprehension from Pass 1
        progress_callback: Optional callback(phase, message, **kwargs)

    Returns:
        List of Proposition objects
    """
    logger.info(f"Pass 2 starting: {chapter_id} ({len(structure.sections)} sections)")

    def notify(message: str, **kwargs):
        if progress_callback:
            progress_callback("pass-2", message, **kwargs)
        logger.info(f"pass-2: {message}")

    # Load prompt once
    system_prompt = _load_prompt(PASS2_SECTION_PROMPT_PATH)

    all_propositions: List[Proposition] = []
    total_sections = len(structure.sections)

    notify(f"Extracting propositions from {total_sections} sections...", sections=total_sections)

    # Process each section
    for i, section in enumerate(structure.sections):
        section_num = i + 1
        notify(
            f"Section {section_num}/{total_sections}: {section.title}",
            sections=total_sections,
            current_section=section_num,
            propositions=len(all_propositions),
            latest=section.title
        )

        section_props = _extract_section_propositions(
            text=text,
            chapter_id=chapter_id,
            section=section,
            all_sections=structure.sections,
            system_prompt=system_prompt
        )

        all_propositions.extend(section_props)
        logger.info(f"Section {section.unit_id}: {len(section_props)} propositions (total: {len(all_propositions)})")

    # Renumber proposition IDs to ensure uniqueness
    for i, prop in enumerate(all_propositions):
        prop.proposition_id = f"{chapter_id}_{prop.unit_id}_p{i+1:03d}"

    # Final stats
    sample_prop = all_propositions[0].proposition_text if all_propositions else None
    notify(
        f"Extracted {len(all_propositions)} propositions from {total_sections} sections",
        sections=total_sections,
        propositions=len(all_propositions),
        latest=sample_prop[:100] + "..." if sample_prop and len(sample_prop) > 100 else sample_prop
    )
    logger.info(f"Pass 2 complete: {len(all_propositions)} propositions from {total_sections} sections")

    return all_propositions


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

    # Validate and fix takeaway proposition references
    valid_prop_ids = {p.proposition_id for p in propositions}
    valid_unit_ids = {s.unit_id for s in structure.sections}

    for takeaway in takeaways:
        # Filter out invalid proposition references
        original_count = len(takeaway.proposition_ids)
        takeaway.proposition_ids = [pid for pid in takeaway.proposition_ids if pid in valid_prop_ids]
        if len(takeaway.proposition_ids) < original_count:
            removed = original_count - len(takeaway.proposition_ids)
            logger.warning(f"Takeaway {takeaway.takeaway_id}: removed {removed} invalid proposition references")

        # Fix invalid unit_id references
        if takeaway.unit_id and takeaway.unit_id not in valid_unit_ids:
            logger.warning(f"Takeaway {takeaway.takeaway_id}: invalid unit_id '{takeaway.unit_id}', setting to None")
            takeaway.unit_id = None

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
