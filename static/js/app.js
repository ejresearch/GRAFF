// State
let currentResults = null;
let currentPage = 'home';

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const fileName = document.getElementById('fileName');

// Navigation
function navigateTo(page) {
  currentPage = page;
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById(`page-${page}`).classList.add('active');

  if (page === 'library') {
    loadLibrary();
  }
}
window.navigateTo = navigateTo;

// Load library
async function loadLibrary() {
  try {
    const response = await fetch('/chapters/list');
    const data = await response.json();
    const container = document.getElementById('libraryGrid');

    if (data.chapters && data.chapters.length > 0) {
      container.innerHTML = data.chapters.map(chapter => `
        <div class="library-item" onclick="loadChapter('${chapter.chapter_id}')">
          <div class="library-item-info">
            <h4>${chapter.chapter_title || chapter.chapter_id || 'Unknown'}</h4>
            <p>${chapter.proposition_count || 0} propositions / ${chapter.takeaway_count || 0} takeaways</p>
          </div>
          <div class="library-item-actions">
            <button class="delete-btn" onclick="event.stopPropagation(); deleteChapter('${chapter.chapter_id}')" title="Delete">x</button>
          </div>
        </div>
      `).join('');
    } else {
      container.innerHTML = `
        <div class="empty-state">
          <p>No analyses yet. Upload a chapter to get started.</p>
          <button class="btn" onclick="navigateTo('home')">+ New Analysis</button>
        </div>
      `;
    }
  } catch (error) {
    console.error('Error loading library:', error);
    document.getElementById('libraryGrid').innerHTML = `
      <div class="empty-state">
        <p>Error loading library. Please refresh.</p>
      </div>
    `;
  }
}

// Delete chapter
window.deleteChapter = async function(chapter_id) {
  if (!confirm('Delete this analysis?')) return;

  try {
    const response = await fetch(`/chapters/${chapter_id}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to delete');
    await loadLibrary();
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
};

// Load specific chapter
window.loadChapter = async function(chapter_id) {
  try {
    const response = await fetch(`/chapters/${chapter_id}`);
    if (!response.ok) throw new Error('Failed to load chapter');

    const data = await response.json();
    currentResults = data;
    showAnalysis(data);
    navigateTo('analysis');
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
};

// Load sample file
window.loadSampleFile = async function(filename) {
  try {
    const response = await fetch(`/samples/${filename}`);
    if (!response.ok) throw new Error('Failed to load sample');

    const text = await response.text();
    const blob = new Blob([text], { type: 'text/plain' });
    const file = new File([blob], filename, { type: 'text/plain' });

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    updateFileName(filename);

    // Auto-fill chapter ID from filename (remove number prefix)
    const chapterId = filename.replace('.txt', '').replace(/^\d+_/, '').replace(/_/g, '-');
    document.getElementById('chapterId').value = chapterId;

    navigateTo('home');
  } catch (error) {
    alert(`Error loading sample: ${error.message}`);
  }
};

// Load samples list dynamically
async function loadSamples() {
  try {
    const response = await fetch('/samples');
    const data = await response.json();

    if (data.samples && data.samples.length > 0) {
      const sampleHTML = data.samples.map(s => `
        <div class="sample-item" onclick="loadSampleFile('${s.filename}')">
          <span>${s.name}</span>
          <small>Sample</small>
        </div>
      `).join('');

      // Update both sample grids
      document.querySelectorAll('.sample-grid').forEach(grid => {
        grid.innerHTML = sampleHTML;
      });
    }
  } catch (error) {
    console.error('Failed to load samples:', error);
  }
}

// File upload handling
dropZone.addEventListener('click', (e) => {
  e.preventDefault();
  fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
    updateFileName(e.dataTransfer.files[0].name);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) {
    updateFileName(e.target.files[0].name);
  }
});

function updateFileName(name) {
  if (fileName) {
    fileName.textContent = `Selected: ${name}`;
    fileName.classList.add('show');
  }
}

// Form submission
uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('chapter_id', document.getElementById('chapterId').value);
  formData.append('version', document.getElementById('version').value);
  formData.append('source_text', document.getElementById('sourceText').value || '');
  formData.append('author_or_editor', document.getElementById('author').value || '');

  // Navigate to processing page
  navigateTo('processing');
  resetProcessingUI();

  let eventSource = null;

  try {
    const response = await fetch('/chapters/digest', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }

    const data = await response.json();
    const jobId = data.job_id;

    if (!jobId) throw new Error('No job ID returned');

    // Connect to SSE for progress
    eventSource = new EventSource(`/chapters/progress/${jobId}`);

    eventSource.onmessage = (event) => {
      try {
        const update = JSON.parse(event.data);

        if (update.status === 'timeout') {
          eventSource.close();
          pollForCompletion();
          return;
        }

        // Extract stats from update
        const stats = {
          sections: update.sections,
          propositions: update.propositions,
          takeaways: update.takeaways,
          latest: update.latest
        };
        updateProgress(update.phase, update.message, undefined, stats);

        if (update.status === 'completed') {
          eventSource.close();
          updateProgress('completed', 'Analysis complete!', 100, stats);
          setTimeout(async () => {
            const listResponse = await fetch('/chapters/list');
            const listData = await listResponse.json();
            if (listData.chapters && listData.chapters.length > 0) {
              const latestChapter = listData.chapters[listData.chapters.length - 1];
              await loadChapter(latestChapter.chapter_id);
            }
          }, 500);
        }

        if (update.status === 'error') {
          eventSource.close();
          showProcessingError(update.message || 'Analysis failed');
        }
      } catch (parseError) {
        console.error('Parse error:', parseError);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      pollForCompletion();
    };

  } catch (error) {
    if (eventSource) eventSource.close();
    showProcessingError(error.message);
  }
});

function resetProcessingUI() {
  document.getElementById('processingTitle').textContent = 'Analyzing Chapter';
  document.getElementById('processingMessage').textContent = 'Starting analysis...';
  document.getElementById('progressPercent').textContent = '0%';
  document.getElementById('progressETA').textContent = 'Estimating...';
  document.getElementById('progressBar').style.width = '0%';
  document.getElementById('liveSections').textContent = '-';
  document.getElementById('livePropositions').textContent = '-';
  document.getElementById('liveTakeaways').textContent = '-';
  document.getElementById('latestPreview').style.display = 'none';

  ['pass1', 'pass2', 'pass3'].forEach(pass => {
    const indicator = document.getElementById(`${pass}-indicator`);
    indicator.classList.remove('active', 'complete');
  });
}

function pollForCompletion() {
  updateProgress('validation', 'Checking for results...', 95);

  const poll = async () => {
    try {
      const listResponse = await fetch('/chapters/list');
      const listData = await listResponse.json();

      if (listData.chapters && listData.chapters.length > 0) {
        const latestChapter = listData.chapters[listData.chapters.length - 1];
        const createdTime = new Date(latestChapter.created_at);
        const now = new Date();
        const diffMinutes = (now - createdTime) / 1000 / 60;

        if (diffMinutes < 60) {
          updateProgress('completed', 'Analysis complete!', 100);
          await loadChapter(latestChapter.chapter_id);
          return;
        }
      }
      setTimeout(poll, 10000);
    } catch (error) {
      setTimeout(poll, 10000);
    }
  };

  setTimeout(poll, 2000);
}

function showProcessingError(message) {
  document.getElementById('processingTitle').textContent = 'Error';
  document.getElementById('processingMessage').textContent = message;
}

function updateProgress(phase, message, percent, stats = {}) {
  const phaseMap = {
    'initialization': { label: 'Initializing', percent: 5 },
    'pass-1': { label: 'Pass 1: Structure', percent: 20 },
    'pass-2': { label: 'Pass 2: Propositions', percent: 50 },
    'pass-3': { label: 'Pass 3: Takeaways', percent: 80 },
    'analysis': { label: 'Analyzing', percent: 50 },
    'storage': { label: 'Saving', percent: 95 },
    'completed': { label: 'Complete!', percent: 100 }
  };

  const phaseInfo = phaseMap[phase] || { label: 'Processing', percent: percent || 50 };
  let targetPercent = percent !== undefined ? percent : phaseInfo.percent;

  document.getElementById('processingTitle').textContent = phaseInfo.label;
  document.getElementById('processingMessage').textContent = message;
  document.getElementById('progressBar').style.width = `${targetPercent}%`;
  document.getElementById('progressPercent').textContent = `${Math.round(targetPercent)}%`;

  // Update pass indicators
  ['pass1', 'pass2', 'pass3'].forEach((pass, i) => {
    const indicator = document.getElementById(`${pass}-indicator`);
    if (!indicator) return;

    const phaseNum = phase.match(/pass-(\d)/)?.[1];
    if (phaseNum && parseInt(phaseNum) === i + 1) {
      indicator.classList.add('active');
      indicator.classList.remove('complete');
    } else if (phaseNum && parseInt(phaseNum) > i + 1) {
      indicator.classList.remove('active');
      indicator.classList.add('complete');
    } else if (phase === 'completed' || phase === 'storage') {
      indicator.classList.remove('active');
      indicator.classList.add('complete');
    }
  });

  // Update live stats
  if (stats.sections !== undefined) {
    document.getElementById('liveSections').textContent = stats.sections;
  }
  if (stats.propositions !== undefined) {
    document.getElementById('livePropositions').textContent = stats.propositions;
  }
  if (stats.takeaways !== undefined) {
    document.getElementById('liveTakeaways').textContent = stats.takeaways;
  }

  // Update latest preview
  if (stats.latest) {
    const preview = document.getElementById('latestPreview');
    const previewText = document.getElementById('latestPreviewText');
    preview.style.display = 'block';
    previewText.textContent = stats.latest;
  }

  // Update ETA
  const progressETA = document.getElementById('progressETA');
  if (phase !== 'completed') {
    const remainingPercent = 100 - targetPercent;
    const estimatedMinutes = Math.ceil((remainingPercent / 100) * 5);
    progressETA.textContent = `~${estimatedMinutes} min remaining`;
  } else {
    progressETA.textContent = 'Done!';
  }
}

// Show analysis results
function showAnalysis(data) {
  currentResults = data;

  // Title and meta
  document.getElementById('analysisTitle').textContent = data.chapter_title || data.chapter_id || 'Analysis Results';
  document.getElementById('analysisMeta').textContent = `${data.chapter_id} / ${data.book_id || 'Unknown Source'}`;

  // Summary
  document.getElementById('analysisSummary').textContent = data.phase1?.summary || 'No summary available';

  // Stats
  const sections = data.phase1?.sections || [];
  const propositions = data.phase2?.propositions || [];
  const takeaways = data.phase2?.key_takeaways || [];
  const keywords = data.phase1?.keywords || [];

  document.getElementById('analysisStats').innerHTML = `
    <span class="stat-item"><strong>${sections.length}</strong> sections</span>
    <span class="stat-item"><strong>${propositions.length}</strong> propositions</span>
    <span class="stat-item"><strong>${takeaways.length}</strong> takeaways</span>
    <span class="stat-item"><strong>${keywords.length}</strong> keywords</span>
  `;

  // Collapsible outline
  renderOutline(data);

  // Chapter takeaways
  renderChapterTakeaways(data);

  // Raw JSON
  document.getElementById('rawJson').textContent = JSON.stringify(data, null, 2);
}

function renderOutline(data) {
  const container = document.getElementById('outlineContainer');
  const sections = data.phase1?.sections || [];
  const propositions = data.phase2?.propositions || [];
  const takeaways = data.phase2?.key_takeaways || [];

  if (sections.length === 0) {
    container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-muted);">No sections found</div>';
    return;
  }

  container.innerHTML = sections.map((section, idx) => {
    // Get propositions for this section (match by prefix for subsections like 1.3.1 -> 1.3)
    const sectionProps = propositions.filter(p =>
      p.unit_id === section.unit_id || p.unit_id.startsWith(section.unit_id + '.')
    );

    // Get takeaways for this section (match by prefix)
    const sectionTakeaways = takeaways.filter(t =>
      t.unit_id === section.unit_id || (t.unit_id && t.unit_id.startsWith(section.unit_id + '.'))
    );

    const propCount = sectionProps.length;
    const takeawayCount = sectionTakeaways.length;

    return `
      <div class="outline-section" data-section="${idx}">
        <div class="outline-header" onclick="toggleSection(${idx})">
          <div class="outline-toggle">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </div>
          <span class="outline-title">${section.unit_id}: ${section.title}</span>
          <span class="outline-meta">${propCount} props / ${takeawayCount} takeaways</span>
        </div>
        <div class="outline-content">
          ${sectionProps.length > 0 ? `
            <div class="nested-group">
              <div class="nested-label">Propositions</div>
              ${sectionProps.map(p => `
                <div class="nested-item">
                  ${p.proposition_text}
                  <div class="nested-item-meta">
                    <span class="bloom-badge bloom-${p.bloom_level}">${p.bloom_level}</span>
                    <span style="font-size: 0.6875rem; color: var(--text-muted);">${p.bloom_verb} / ${p.source_type}</span>
                  </div>
                </div>
              `).join('')}
            </div>
          ` : ''}
          ${sectionTakeaways.length > 0 ? `
            <div class="nested-group">
              <div class="nested-label">Section Takeaways</div>
              ${sectionTakeaways.map(t => `
                <div class="nested-item" style="border-left: 2px solid var(--bg-inverse); padding-left: 1rem;">
                  ${t.text}
                  <div class="nested-item-meta">
                    <span class="bloom-badge bloom-${t.dominant_bloom_level}">${t.dominant_bloom_level}</span>
                    <span style="font-size: 0.6875rem; color: var(--text-muted);">${t.proposition_ids.length} propositions</span>
                  </div>
                </div>
              `).join('')}
            </div>
          ` : ''}
          ${sectionProps.length === 0 && sectionTakeaways.length === 0 ? `
            <div style="color: var(--text-muted); font-size: 0.8125rem;">No propositions or takeaways for this section</div>
          ` : ''}
        </div>
      </div>
    `;
  }).join('');
}

window.toggleSection = function(idx) {
  const section = document.querySelector(`.outline-section[data-section="${idx}"]`);
  if (section) {
    section.classList.toggle('open');
  }
};

function renderChapterTakeaways(data) {
  const container = document.getElementById('chapterTakeaways');
  const takeaways = data.phase2?.key_takeaways || [];

  // Get takeaways that span multiple sections or have no unit_id (chapter-level)
  const chapterTakeaways = takeaways.filter(t => !t.unit_id || t.unit_id === null);

  // If no chapter-level takeaways, show all takeaways
  const displayTakeaways = chapterTakeaways.length > 0 ? chapterTakeaways : takeaways.slice(0, 5);

  if (displayTakeaways.length === 0) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.875rem;">No takeaways found</div>';
    return;
  }

  container.innerHTML = displayTakeaways.map(t => `
    <div class="takeaway-card">
      <div class="takeaway-text">${t.text}</div>
      <div class="takeaway-meta">
        <span class="bloom-badge bloom-${t.dominant_bloom_level}">${t.dominant_bloom_level}</span>
        / Based on ${t.proposition_ids.length} propositions
        ${t.unit_id ? `/ ${t.unit_id}` : ''}
      </div>
    </div>
  `).join('');
}

// Export functions
window.exportMarkdown = function() {
  if (!currentResults) return;

  const data = currentResults;
  const sections = data.phase1?.sections || [];
  const propositions = data.phase2?.propositions || [];
  const takeaways = data.phase2?.key_takeaways || [];

  let md = `# ${data.chapter_title || data.chapter_id}\n\n`;
  md += `**Source:** ${data.book_id || 'Unknown'}\n\n`;
  md += `## Summary\n\n${data.phase1?.summary || 'No summary'}\n\n`;

  md += `## Chapter Outline\n\n`;

  sections.forEach(section => {
    md += `### ${section.unit_id}: ${section.title}\n\n`;

    const sectionProps = propositions.filter(p =>
      p.unit_id === section.unit_id || p.unit_id.startsWith(section.unit_id + '.')
    );
    const sectionTakeaways = takeaways.filter(t =>
      t.unit_id === section.unit_id || (t.unit_id && t.unit_id.startsWith(section.unit_id + '.'))
    );

    if (sectionProps.length > 0) {
      md += `**Propositions:**\n\n`;
      sectionProps.forEach(p => {
        md += `- [${p.bloom_level}] ${p.proposition_text}\n`;
      });
      md += `\n`;
    }

    if (sectionTakeaways.length > 0) {
      md += `**Takeaways:**\n\n`;
      sectionTakeaways.forEach(t => {
        md += `> ${t.text}\n\n`;
      });
    }
  });

  // Chapter-level takeaways
  const chapterTakeaways = takeaways.filter(t => !t.unit_id);
  if (chapterTakeaways.length > 0) {
    md += `## Key Takeaways\n\n`;
    chapterTakeaways.forEach(t => {
      md += `> ${t.text}\n\n`;
    });
  }

  // Download
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${data.chapter_id || 'analysis'}.md`;
  a.click();
  URL.revokeObjectURL(url);
};

window.copyJson = function() {
  if (!currentResults) return;

  const text = JSON.stringify(currentResults, null, 2);
  navigator.clipboard.writeText(text).then(() => {
    alert('JSON copied to clipboard');
  });
};

window.toggleRawJson = function() {
  const rawJson = document.getElementById('rawJson');
  rawJson.classList.toggle('hidden');
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
});
