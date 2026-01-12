from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
import io
import json
import asyncio
from typing import Dict, List
from .services.graff_orchestrator import digest_chapter_graff, DigestError, AnalysisError, StorageError
from .db import list_chapters, load_chapter_analysis, init_database
from .utils.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging(level="INFO")
logger = get_logger(__name__)

app = FastAPI(title="Doc Digester", version="0.2.0")

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Maximum file size: 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024

# Global progress tracker
progress_tracker: Dict[str, List[Dict]] = {}

def add_progress(job_id: str, phase: str, message: str, status: str = "in_progress", **kwargs):
    """Add a progress update for a job with optional stats"""
    if job_id not in progress_tracker:
        progress_tracker[job_id] = []
    update = {
        "phase": phase,
        "message": message,
        "status": status,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    # Add any extra stats (sections, propositions, takeaways, latest)
    update.update(kwargs)
    progress_tracker[job_id].append(update)
    logger.info(f"Job {job_id}: {phase} - {message}")

@app.get("/chapters/progress/{job_id}")
async def stream_progress(job_id: str):
    """Stream progress updates via Server-Sent Events"""
    async def event_generator():
        last_index = 0
        timeout_count = 0
        max_timeout = 3600  # 30 minutes timeout (3600 * 0.5s = 1800s = 30 min)

        while timeout_count < max_timeout:
            if job_id in progress_tracker:
                updates = progress_tracker[job_id][last_index:]
                if updates:
                    for update in updates:
                        yield f"data: {json.dumps(update)}\n\n"
                    last_index = len(progress_tracker[job_id])

                    # Check if job is complete
                    if updates[-1]["status"] in ["completed", "error"]:
                        break

            await asyncio.sleep(0.5)
            timeout_count += 0.5

        # Send final message
        yield f"data: {json.dumps({'status': 'timeout', 'message': 'Stream closed'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.on_event("startup")
async def startup_event():
    import os
    logger.info("Doc Digester API starting up")

    # Initialize database
    init_database()
    logger.info("Database initialized")

    # Environment diagnostics
    logger.info(f"ENV CHECK - OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    logger.info(f"ENV CHECK - OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'NOT SET')}")
    logger.info(f"ENV CHECK - USE_ACTUAL_LLM: {os.getenv('USE_ACTUAL_LLM', 'NOT SET')}")
    if os.getenv('OPENAI_API_KEY'):
        logger.info(f"ENV CHECK - API Key length: {len(os.getenv('OPENAI_API_KEY'))} chars")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Doc Digester API shutting down")

async def process_chapter_background(
    job_id: str,
    text: str,
    book_id: str,
    chapter_title: str,
    chapter_id: str = None
):
    """Background task to process GRAFF chapter analysis."""
    try:
        # Run the synchronous GRAFF pipeline in a thread pool
        result = await asyncio.to_thread(
            digest_chapter_graff,
            text,
            book_id,
            chapter_title,
            chapter_id,
            lambda phase, msg, **kwargs: add_progress(job_id, phase, msg, **kwargs)
        )
        add_progress(job_id, "completed", "Analysis complete!", "completed")
        logger.info(f"Successfully processed chapter: {result.chapter_id}")
    except AnalysisError as e:
        logger.error(f"Analysis error in background task: {e}")
        add_progress(job_id, "error", f"Analysis failed: {str(e)}", "error")
    except StorageError as e:
        logger.error(f"Storage error in background task: {e}")
        add_progress(job_id, "error", f"Storage failed: {str(e)}", "error")
    except DigestError as e:
        logger.error(f"Digest error in background task: {e}")
        add_progress(job_id, "error", f"Analysis failed: {str(e)}", "error")
    except Exception as e:
        logger.exception(f"Unexpected error in background task: {e}")
        add_progress(job_id, "error", f"Unexpected error: {str(e)}", "error")


@app.post("/chapters/digest")
async def chapters_digest(
    file: UploadFile = File(...),
    chapter_id: str | None = Form(default=None),
    file_name: str | None = Form(default=None),
    author_or_editor: str | None = Form(default=None),
    version: str | None = Form(default="v1"),
    created_at: str | None = Form(default=None),
    source_text: str | None = Form(default=None),
):
    """
    Process a chapter document through the 5-phase analysis pipeline.

    Returns the job_id immediately for progress tracking via SSE.
    """
    logger.info(f"Received digest request for file: {file.filename}")

    try:
        # Read file content with size validation
        content = b""
        chunk_size = 1024 * 1024  # 1MB chunks
        total_size = 0

        while chunk := await file.read(chunk_size):
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                logger.warning(f"File too large: {total_size} bytes (max: {MAX_FILE_SIZE})")
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024)}MB"
                )
            content += chunk

        if not content:
            logger.warning("Empty file uploaded")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )

        # Extract text based on file type
        filename_lower = file.filename.lower() if file.filename else ""

        if filename_lower.endswith('.docx'):
            # Handle .docx files using python-docx
            logger.info("Processing .docx file")
            try:
                doc = Document(io.BytesIO(content))
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                text = '\n'.join(paragraphs)
                logger.info(f"Extracted {len(paragraphs)} paragraphs from .docx")
            except Exception as e:
                logger.error(f"Failed to extract text from .docx: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unable to process .docx file. Please ensure it's a valid Word document."
                )
        elif filename_lower.endswith('.pdf'):
            # Handle .pdf files using PyPDF2
            logger.info("Processing .pdf file")
            try:
                pdf_reader = PdfReader(io.BytesIO(content))
                pages_text = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        pages_text.append(page_text)
                text = '\n'.join(pages_text)
                logger.info(f"Extracted text from {len(pdf_reader.pages)} pages in PDF")
            except Exception as e:
                logger.error(f"Failed to extract text from .pdf: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unable to process .pdf file. Please ensure it's a valid PDF document."
                )
        else:
            # Handle plain text files
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 decode failed, attempting with latin-1")
                try:
                    text = content.decode('latin-1')
                except UnicodeDecodeError as e:
                    logger.error(f"Failed to decode file: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Unable to decode file. Please ensure it's a valid text file."
                    )

        if len(text.strip()) < 100:
            logger.warning(f"Text too short: {len(text)} characters")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text content is too short for analysis (minimum 100 characters)"
            )

        # Extract book_id and chapter_title from metadata
        book_id = author_or_editor or "unknown_book"  # Use author as book_id for now
        chapter_title = file_name or file.filename or "Untitled Chapter"

        # Clean chapter title (remove file extension)
        if chapter_title.endswith(('.txt', '.docx', '.pdf')):
            chapter_title = '.'.join(chapter_title.split('.')[:-1])

        logger.info(f"Processing chapter: '{chapter_title}' ({len(text)} characters)")

        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())

        # Initialize progress tracking
        add_progress(job_id, "initialization", "Starting GRAFF analysis...", "in_progress")

        # Start background processing
        asyncio.create_task(process_chapter_background(
            job_id=job_id,
            text=text,
            book_id=book_id,
            chapter_title=chapter_title,
            chapter_id=chapter_id
        ))

        # Return job_id immediately for progress tracking
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Analysis started. Use the job_id to track progress."
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.exception(f"Unexpected error processing chapter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during processing"
        )

@app.get("/")
async def root():
    """Serve the web interface."""
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Doc Digester API", "version": "0.2.0"}

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "doc-digester"}

@app.get("/samples")
async def list_samples():
    """List available sample data files."""
    sample_path = Path(__file__).parent.parent / "sample_data"
    if not sample_path.exists():
        return {"samples": []}

    samples = []
    for f in sorted(sample_path.glob("*.txt")):
        samples.append({
            "filename": f.name,
            "name": f.stem.split("_", 1)[1].replace("_", " ").title() if "_" in f.stem else f.stem
        })
    return {"samples": samples}

@app.get("/samples/{filename}")
async def get_sample(filename: str):
    """Get a sample data file."""
    sample_path = Path(__file__).parent.parent / "sample_data" / filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(str(sample_path), media_type="text/plain")

@app.get("/chapters/list")
async def get_chapters_list():
    """List all available chapter analyses from the database."""
    try:
        from .db import list_chapters as db_list_chapters
        chapters_list = db_list_chapters()
        return {"chapters": chapters_list}
    except Exception as e:
        logger.error(f"Failed to list chapters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list chapters: {str(e)}"
        )


@app.get("/chapters/{chapter_id}")
async def get_chapter(chapter_id: str):
    """Get a specific chapter analysis by chapter_id."""
    try:
        from .db import load_chapter_analysis
        chapter = load_chapter_analysis(chapter_id)

        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chapter {chapter_id} not found"
            )

        # Return as dict for JSON serialization
        return chapter.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load chapter {chapter_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load chapter: {str(e)}"
        )


@app.delete("/chapters/{chapter_id}")
async def delete_chapter_endpoint(chapter_id: str):
    """Delete a specific chapter analysis by chapter_id."""
    try:
        from .db import delete_chapter
        success = delete_chapter(chapter_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chapter {chapter_id} not found"
            )

        logger.info(f"Deleted chapter: {chapter_id}")
        return {"message": "Chapter deleted successfully", "chapter_id": chapter_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chapter {chapter_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chapter: {str(e)}"
        )


@app.get("/chapters/{chapter_id}/propositions")
async def get_chapter_propositions(
    chapter_id: str,
    bloom_level: str = None
):
    """
    Get propositions for a chapter, optionally filtered by Bloom level.

    Query params:
    - bloom_level: Filter by Bloom level (remember, understand, apply, analyze)
    """
    try:
        from .db import load_chapter_analysis, get_propositions_by_bloom

        # If bloom_level filter specified, use specialized query
        if bloom_level:
            if bloom_level not in ['remember', 'understand', 'apply', 'analyze']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid Bloom level: {bloom_level}"
                )

            propositions = get_propositions_by_bloom(chapter_id, bloom_level)
            return {
                "chapter_id": chapter_id,
                "bloom_level": bloom_level,
                "count": len(propositions),
                "propositions": [p.model_dump() for p in propositions]
            }

        # Otherwise, load full chapter and return all propositions
        chapter = load_chapter_analysis(chapter_id)
        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chapter {chapter_id} not found"
            )

        return {
            "chapter_id": chapter_id,
            "count": len(chapter.phase2.propositions),
            "bloom_distribution": chapter.get_bloom_distribution(),
            "propositions": [p.model_dump() for p in chapter.phase2.propositions]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get propositions for {chapter_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get propositions: {str(e)}"
        )
