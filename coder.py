import os
import re
import glob as glob_module
import subprocess
import fnmatch
import argparse
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle

# Global working directory - all operations will be restricted to this directory
WORKING_DIRECTORY = None

# Rate Limiting Configuration (Free Tier Limits)
RATE_LIMITS = {
    'rpm': 10,
    'tpm': 250_000,
    'rpd': 250
}

class RateLimiter:
    """Intelligent rate limiter for Gemini API that manages RPM, TPM, and RPD limits."""
    
    def __init__(self, rpm: int = 10, tpm: int = 250_000, rpd: int = 250):
        self.rpm_limit = rpm
        self.tpm_limit = tpm
        self.rpd_limit = rpd
        
        # Thread-safe tracking
        self._lock = threading.Lock()
        
        # Sliding window tracking using deques
        self._requests_minute = deque()  # (timestamp, tokens_used)
        self._requests_day = deque()     # (timestamp, tokens_used)
        
        # Current period tracking
        self._current_minute_tokens = 0
        self._current_minute_requests = 0
        self._current_day_requests = 0
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_delay_time = 0
        
        console.print(f"[bold bright_blue]⚡ Rate Limiter Initialized:[/bold bright_blue]")
        console.print(f"[dim]  • {rpm} requests/minute, {tpm:,} tokens/minute, {rpd} requests/day[/dim]")
    
    def _cleanup_old_entries(self, now: datetime):
        """Remove entries older than the tracking window."""
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        # Clean minute window
        while self._requests_minute and self._requests_minute[0][0] < minute_ago:
            old_entry = self._requests_minute.popleft()
            self._current_minute_requests -= 1
            self._current_minute_tokens -= old_entry[1]
        
        # Clean day window
        while self._requests_day and self._requests_day[0][0] < day_ago:
            self._requests_day.popleft()
            self._current_day_requests -= 1
    
    def _calculate_required_delay(self, now: datetime, estimated_tokens: int = 1000) -> float:
        """Calculate the minimum delay needed to stay within all rate limits."""
        delays = []
        
        # Check RPM limit
        if self._current_minute_requests >= self.rpm_limit:
            # Find when the oldest request will expire
            oldest_request_time = self._requests_minute[0][0]
            rpm_delay = (oldest_request_time + timedelta(minutes=1) - now).total_seconds()
            if rpm_delay > 0:
                delays.append(rpm_delay)
        
        # Check TPM limit
        if self._current_minute_tokens + estimated_tokens > self.tpm_limit:
            if self._requests_minute:
                # Find when enough tokens will be freed up
                needed_tokens = (self._current_minute_tokens + estimated_tokens) - self.tpm_limit
                tokens_freed = 0
                
                for timestamp, tokens in self._requests_minute:
                    tokens_freed += tokens
                    if tokens_freed >= needed_tokens:
                        tpm_delay = (timestamp + timedelta(minutes=1) - now).total_seconds()
                        if tpm_delay > 0:
                            delays.append(tpm_delay)
                        break
        
        # Check RPD limit
        if self._current_day_requests >= self.rpd_limit:
            if self._requests_day:
                oldest_day_request = self._requests_day[0][0]
                rpd_delay = (oldest_day_request + timedelta(days=1) - now).total_seconds()
                if rpd_delay > 0:
                    delays.append(rpd_delay)
        
        return max(delays) if delays else 0
    
    def _estimate_tokens(self, conversation_history: List[Dict], user_message: str = "") -> int:
        """Estimate token usage for a request."""
        # Basic estimation based on character count
        # This is a rough estimate - the actual API will give us precise counts
        total_chars = len(user_message)
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            for part in msg.get('parts', []):
                if isinstance(part, dict) and 'text' in part:
                    total_chars += len(part['text'])
                elif hasattr(part, 'text'):
                    total_chars += len(part.text or '')
        
        # Rough estimate: ~4 characters per token (varies by language/content)
        estimated_tokens = total_chars // 4
        
        # Add buffer for function calls and system prompt
        estimated_tokens += 1000
        
        # Cap the estimate to prevent over-conservative delays
        return min(estimated_tokens, 10000)
    
    def wait_if_needed(self, conversation_history: List[Dict] = None, user_message: str = "") -> Dict[str, Any]:
        """Wait if necessary to respect rate limits before making a request."""
        with self._lock:
            now = datetime.now()
            self._cleanup_old_entries(now)
            
            # Estimate token usage for this request
            estimated_tokens = self._estimate_tokens(conversation_history or [], user_message)
            
            # Calculate required delay
            delay_needed = self._calculate_required_delay(now, estimated_tokens)
            
            delay_info = {
                'delay_applied': delay_needed,
                'estimated_tokens': estimated_tokens,
                'current_rpm': self._current_minute_requests,
                'current_tpm': self._current_minute_tokens,
                'current_rpd': self._current_day_requests
            }
            
            if delay_needed > 0:
                self.total_delay_time += delay_needed
                
                # Show rate limit info
                console.print(f"[bold yellow]⏳ Rate Limit: Waiting {delay_needed:.1f}s[/bold yellow]")
                
                # Show which limit is causing the delay
                if self._current_minute_requests >= self.rpm_limit:
                    console.print(f"[dim]   • RPM limit reached: {self._current_minute_requests}/{self.rpm_limit}[/dim]")
                if self._current_minute_tokens + estimated_tokens > self.tpm_limit:
                    console.print(f"[dim]   • TPM limit approaching: {self._current_minute_tokens:,}+{estimated_tokens:,}/{self.tpm_limit:,}[/dim]")
                if self._current_day_requests >= self.rpd_limit:
                    console.print(f"[dim]   • RPD limit reached: {self._current_day_requests}/{self.rpd_limit}[/dim]")
                
                time.sleep(delay_needed)
                
                # Update time after delay
                now = datetime.now()
                self._cleanup_old_entries(now)
            
            # Record the pending request (will be updated with actual tokens later)
            self._requests_minute.append((now, estimated_tokens))
            self._requests_day.append((now, estimated_tokens))
            self._current_minute_requests += 1
            self._current_minute_tokens += estimated_tokens
            self._current_day_requests += 1
            self.total_requests += 1
            
            return delay_info
    
    def update_actual_usage(self, actual_tokens: int):
        """Update the rate limiter with actual token usage from the API response."""
        with self._lock:
            if self._requests_minute:
                # Update the most recent request with actual token count
                timestamp, estimated_tokens = self._requests_minute[-1]
                
                # Adjust current counters
                token_difference = actual_tokens - estimated_tokens
                self._current_minute_tokens += token_difference
                
                # Update the record
                self._requests_minute[-1] = (timestamp, actual_tokens)
                self._requests_day[-1] = (timestamp, actual_tokens)
                
                self.total_tokens += actual_tokens
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        with self._lock:
            now = datetime.now()
            self._cleanup_old_entries(now)
            
            return {
                'requests_this_minute': self._current_minute_requests,
                'tokens_this_minute': self._current_minute_tokens,
                'requests_today': self._current_day_requests,
                'rpm_limit': self.rpm_limit,
                'tpm_limit': self.tpm_limit,
                'rpd_limit': self.rpd_limit,
                'total_requests': self.total_requests,
                'total_tokens': self.total_tokens,
                'total_delay_time': self.total_delay_time,
                'rpm_usage_percent': (self._current_minute_requests / self.rpm_limit) * 100,
                'tpm_usage_percent': (self._current_minute_tokens / self.tpm_limit) * 100,
                'rpd_usage_percent': (self._current_day_requests / self.rpd_limit) * 100
            }

# Initialize Rich console first (needed by rate limiter)
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',  # Bright blue prompt
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# Global rate limiter instance
rate_limiter = RateLimiter(**RATE_LIMITS)

class ContextCacheManager:
    """Manages intelligent explicit context caching for Gemini API to reduce costs and improve performance.
    
    Smart Caching Strategy:
    - Analyzes TOTAL context size (system + conversation + tools + message) before caching
    - Only caches when the complete context exceeds API minimums (1,024+ tokens)
    - Caches conversation history when total context is substantial, not tiny individual pieces
    - Automatically determines when explicit caching provides value vs. relying on implicit caching
    - Focuses on large conversations where caching previous context reduces repeated costs
    
    How It Works:
    - Estimates total tokens that will be sent to API (system prompt + all conversation + tools)
    - If total ≥ 1,024 tokens AND conversation has enough history, creates context cache
    - Cache contains system prompt + older conversation (excludes last 2 messages)
    - Recent messages sent normally while older context served from cache
    - Results in significant cost savings for large conversation contexts
    
    API Requirements (from Gemini documentation):
    - Minimum tokens: 1,024 for 2.5 Flash, 2,048 for 2.5 Pro
    - TTL format: String like "300s" for 5 minutes, "7200s" for 2 hours
    - Model version: Must use explicit version suffix like "gemini-2.5-flash-001"
    - Cost savings: Cached tokens billed at reduced rate vs regular input tokens
    """
    
    def __init__(self, client: genai.Client):
        self.client = client
        self._lock = threading.Lock()
        
        # Cache storage
        self._caches = {}  # cache_key -> cache_object
        self._cache_metadata = {}  # cache_key -> metadata
        
        # Cache statistics
        self.total_caches_created = 0
        self.total_cache_hits = 0
        self.total_tokens_cached = 0
        
        # Configuration
        self.min_tokens_for_caching = 1024  # Gemini 2.5 Flash minimum (2,048 for 2.5 Pro)
        self.default_ttl_hours = 6  # Reasonable default for development sessions
        
        console.print(f"[bold bright_blue]💾 Context Cache Manager Initialized[/bold bright_blue]")
        console.print(f"[dim]  • Minimum tokens for caching: {self.min_tokens_for_caching:,}[/dim]")
        console.print(f"[dim]  • Default TTL: {self.default_ttl_hours} hours[/dim]")
    
    def _estimate_token_count(self, content: str) -> int:
        """Estimate token count for content (rough approximation)."""
        return standardize_token_estimation(content)
    
    def _create_cache_key(self, content_type: str, content_hash: str) -> str:
        """Create a unique cache key."""
        return f"{content_type}_{content_hash}_{int(time.time() // 3600)}"
    
    def _get_content_hash(self, content: str) -> str:
        """Get a hash of content for cache key generation."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def create_cache(self, content: str, content_type: str, ttl_hours: Optional[int] = None) -> Optional[str]:
        """Create an explicit cache for content if it meets the criteria."""
        try:
            estimated_tokens = self._estimate_token_count(content)
            
            if estimated_tokens < self.min_tokens_for_caching:
                return None
            
            content_hash = self._get_content_hash(content)
            cache_key = self._create_cache_key(content_type, content_hash)
            
            # Check if we already have this cache
            if cache_key in self._caches:
                console.print(f"[dim]💾 Using existing cache for {content_type}[/dim]")
                return self._caches[cache_key].name
            
            # Convert TTL to string format as required by API
            ttl_hours_value = ttl_hours or self.default_ttl_hours
            ttl_seconds = ttl_hours_value * 3600
            ttl_string = f"{ttl_seconds}s"
            
            with console.status(f"[bold yellow]💾 Creating cache for {content_type}...[/bold yellow]"):
                # Use the correct API format according to documentation
                    cache = self.client.caches.create(
                        model="gemini-2.5-flash",
                    config=types.CreateCachedContentConfig(
                        display_name=f"{content_type}_{content_hash}",
                        system_instruction=content,
                        ttl=ttl_string
                    )
                )
            
            # Store cache information
            with self._lock:
                self._caches[cache_key] = cache
                self._cache_metadata[cache_key] = {
                    'content_type': content_type,
                    'estimated_tokens': estimated_tokens,
                    'created_at': datetime.now(),
                    'ttl_hours': ttl_hours_value,
                    'content_hash': content_hash
                }
                self.total_caches_created += 1
                self.total_tokens_cached += estimated_tokens
            
            console.print(f"[bold green]✅ Cache created for {content_type}: {estimated_tokens:,} tokens[/bold green]")
            console.print(f"[dim]   Cache ID: {cache.name}[/dim]")
            
            return cache.name
            
        except Exception as e:
            console.print(f"[dim yellow]💾 Cache creation failed: {e}[/dim yellow]")
            console.print(f"[dim]Continuing without cache for {content_type}[/dim]")
            return None
    

    
    def create_file_cache(self, file_path: str, content: str) -> Optional[str]:
        """Create a cache for large file content.
        
        Only caches files that are substantial enough to benefit from caching.
        This is useful for large files that might be referenced multiple times.
        """
        # Check if the file content is worth caching
        estimated_tokens = self._estimate_token_count(content)
        if estimated_tokens < self.min_tokens_for_caching:
            console.print(f"[dim]💾 File too small for caching: {file_path} ({estimated_tokens} < {self.min_tokens_for_caching} tokens)[/dim]")
            return None
        
        cache_content = f"File: {file_path}\n\n{content}"
        
        console.print(f"[dim green]💾 Caching large file: {file_path} ({estimated_tokens:,} tokens)[/dim green]")
        
        return self.create_cache(
            content=cache_content,
            content_type=f"file_{os.path.basename(file_path)}",
            ttl_hours=4  # Medium TTL for file content
        )
    

    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics."""
        with self._lock:
            active_caches = len(self._caches)
            cache_types = {}
            
            for metadata in self._cache_metadata.values():
                content_type = metadata['content_type']
                cache_types[content_type] = cache_types.get(content_type, 0) + 1
            
            return {
                'active_caches': active_caches,
                'total_created': self.total_caches_created,
                'total_tokens_cached': self.total_tokens_cached,
                'cache_types': cache_types,
                'estimated_cost_savings': self.total_tokens_cached * 0.25 / 1000  # Rough estimate
            }

    def should_cache_total_context(self, conversation_history: List[Dict], user_message: str = "") -> bool:
        """Determine if the TOTAL context (system + conversation + tools + message) should be cached.
        
        This is the correct approach - cache when the entire context that will be sent 
        to the API exceeds the minimum, not individual pieces.
        """
        # Estimate total context that will be sent to API
        total_chars = len(SYSTEM_PROMPT)  # System instruction
        total_chars += len(user_message)   # Current user message
        
        # Add conversation history
        for msg in conversation_history:
            for part in msg.get("parts", []):
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
                elif hasattr(part, 'text') and part.text:
                    total_chars += len(part.text)
        
        # Add tool definitions (rough estimate)
        total_chars += 5000  # Tools schema and descriptions
        
        estimated_total_tokens = total_chars // 4
        return estimated_total_tokens >= self.min_tokens_for_caching
    
    def create_context_cache(self, conversation_history: List[Dict], user_message: str = "") -> Optional[str]:
        """Create a cache for the substantial conversation context.
        
        This caches the conversation history (not current message) when the total context
        is large enough to benefit from caching.
        """
        if not self.should_cache_total_context(conversation_history, user_message):
            return None
        
        # Cache the conversation history (excluding the current message)
        # This way the current message can still be sent normally while previous context is cached
        if len(conversation_history) < 3:
            return None  # Need some conversation to cache
        
        # Convert older conversation to cacheable format (exclude last 2 messages to keep fresh)
        conversation_text = f"System: {SYSTEM_PROMPT}\n\n"
        
        for msg in conversation_history[:-2]:  # Cache all but last 2 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            for part in msg.get("parts", []):
                if isinstance(part, dict) and "text" in part:
                    conversation_text += f"{role}: {part['text']}\n\n"
                elif hasattr(part, 'text') and part.text:
                    conversation_text += f"{role}: {part.text}\n\n"
        
        estimated_tokens = self._estimate_token_count(conversation_text)
        console.print(f"[dim green]💾 Creating context cache: {estimated_tokens:,} tokens[/dim green]")
        
        return self.create_cache(
            content=conversation_text,
            content_type="conversation_context",
            ttl_hours=2  # Shorter TTL for dynamic conversation content
        )




# Comprehensive default exclusions for better performance and relevance
DEFAULT_EXCLUDE_PATTERNS = {
    # Python ecosystem
    'python': [
        '.venv', 'venv', '.env', 'env', '.virtualenv', 'virtualenv',
        '__pycache__', '.pytest_cache', '.mypy_cache', '.coverage',
        '.tox', '.nox', 'build', 'dist', 'eggs', '*.egg-info',
        '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.dylib'
    ],
    
    # Node.js/JavaScript ecosystem  
    'nodejs': [
        'node_modules', '.next', '.nuxt', '.output', '.vercel',
        '.turbo', '.parcel-cache', 'coverage', '.nyc_output',
        'storybook-static', '.cache', '.webpack', '.rollup.cache'
    ],
    
    # Build systems and outputs
    'build': [
        'build', 'dist', 'out', 'target', 'bin', 'obj', 'release',
        'debug', '.build', '_build', 'cmake-build-*', '.gradle',
        'vendor', 'deps', '_deps'
    ],
    
    # Version control and IDEs
    'vcs_ide': [
        '.git', '.svn', '.hg', 'CVS', '.bzr',
        '.idea', '.vscode', '.vs', '.atom', '.sublime-*',
        '*.swp', '*.swo', '*~', '.project', '.classpath'
    ],
    
    # System and temporary files
    'system': [
        '.DS_Store', 'Thumbs.db', 'Desktop.ini', '*.tmp', '*.temp',
        '.tmp', '.temp', 'tmp', 'temp', '.Trash-*', '$RECYCLE.BIN'
    ],
    
    # Logs and databases
    'data': [
        '*.log', '*.db', '*.sqlite', '*.sqlite3', 'logs',
        '*.pid', '*.lock', '.lock', 'lockfiles'
    ],
    
    # Archives and binaries
    'binary': [
        '*.zip', '*.tar', '*.gz', '*.7z', '*.rar', '*.exe', '*.dmg',
        '*.iso', '*.img', '*.bin', '*.deb', '*.rpm', '*.msi'
    ],
    
    # Media files (typically not relevant for code analysis)
    'media': [
        '*.png', '*.jpg', '*.jpeg', '*.gif', '*.ico', '*.svg', '*.webp',
        '*.avif', '*.bmp', '*.tiff', '*.psd', '*.ai', '*.sketch',
        '*.mp4', '*.webm', '*.mov', '*.avi', '*.mkv', '*.flv',
        '*.mp3', '*.wav', '*.ogg', '*.m4a', '*.aac', '*.flac'
    ],
    
    # Documentation builds
    'docs': [
        '_site', '.jekyll-cache', '.docusaurus', 'sphinx-build',
        '.vuepress', '.gitbook', 'book'
    ],
    
    # Language-specific
    'other_languages': [
        # Rust
        'Cargo.lock', 'target',
        # Go
        'go.sum', 'vendor',
        # Java
        '*.class', '.gradle', 'gradle-wrapper.*',
        # C/C++
        '*.o', '*.obj', '*.so', '*.dll', '*.lib', '*.a',
        # .NET
        'packages', '*.nupkg', 'bin', 'obj',
        # Ruby
        '.bundle', 'vendor/bundle',
        # PHP
        'vendor', 'composer.lock'
    ]
}

# Flatten all patterns into a single list
ALL_DEFAULT_EXCLUDES = []
for category in DEFAULT_EXCLUDE_PATTERNS.values():
    ALL_DEFAULT_EXCLUDES.extend(category)

def should_exclude_path(path: str, additional_excludes: List[str] = None) -> bool:
    """Check if a path should be excluded based on default patterns and additional excludes."""
    excludes = ALL_DEFAULT_EXCLUDES + (additional_excludes or [])
    
    path_name = os.path.basename(path)
    path_lower = path_name.lower()
    
    for pattern in excludes:
        # Direct name match
        if pattern == path_name or pattern == path_lower:
            return True
            
        # Glob pattern match
        if fnmatch.fnmatch(path_name, pattern) or fnmatch.fnmatch(path_lower, pattern):
            return True
            
        # Check if any part of the path matches
        if pattern in path or pattern.lower() in path.lower():
            return True
    
    return False

def get_smart_excludes(include_media: bool = False, include_build: bool = False) -> List[str]:
    """Get smart default excludes with options to include certain categories."""
    excludes = []
    
    # Always exclude these
    for category in ['python', 'nodejs', 'vcs_ide', 'system', 'data', 'binary', 'docs', 'other_languages']:
        excludes.extend(DEFAULT_EXCLUDE_PATTERNS[category])
    
    # Conditionally include/exclude
    if not include_build:
        excludes.extend(DEFAULT_EXCLUDE_PATTERNS['build'])
    
    if not include_media:
        excludes.extend(DEFAULT_EXCLUDE_PATTERNS['media'])
    
    return excludes

# --------------------------------------------------------------------------------
# 1. Configure Google GenAI client and load environment variables
# --------------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize cache manager
cache_manager = ContextCacheManager(client)

# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

# --------------------------------------------------------------------------------
# 2.1. Define Function Calling Tools - Using automatic function calling
# --------------------------------------------------------------------------------

def read_file_tool(file_path: str, limit: Optional[int] = None, offset: Optional[int] = None) -> str:
    """Read the content of a single file from the filesystem with optional line limits and offset.
    
    Args:
        file_path: The path to the file to read (relative or absolute)
        limit: Maximum number of lines to read (optional)
        offset: Line number to start reading from (1-indexed, optional)
    
    Returns:
        The content of the file as a string
    """
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        lines = content.splitlines()
        
        # Apply offset and limit if specified
        if offset is not None:
            start_idx = max(0, offset - 1)  # Convert to 0-indexed
            lines = lines[start_idx:]
            
        if limit is not None:
            lines = lines[:limit]
            
        filtered_content = '\n'.join(lines)
        
        # Enhanced visual feedback with display path
        display_path = get_display_path(normalized_path)
        total_lines = len(content.splitlines())
        filtered_lines = len(lines)
        size_kb = len(filtered_content.encode('utf-8')) / 1024
        
        console.print(f"       📖 Reading: [bright_cyan]{display_path}[/bright_cyan]")
        if offset or limit:
            range_info = f" (lines {offset or 1}-{(offset or 1) + filtered_lines - 1} of {total_lines})"
            console.print(f"       📊 File size: {size_kb:.1f} KB, {filtered_lines} lines{range_info}")
        else:
            console.print(f"       📊 File size: {size_kb:.1f} KB, {filtered_lines} lines")
        
        # Show code preview with display path
        show_code_preview(filtered_content, display_path)
        
        return filtered_content
    except Exception as e:
        error_msg = f"Error reading file '{file_path}': {str(e)}"
        console.print(f"       [bold red]❌ Failed to read file: {error_msg}[/bold red]")
        return error_msg

def read_multiple_files_tool(file_paths: List[str]) -> str:
    """Read multiple files from the filesystem.
    
    Args:
        file_paths: List of file paths to read
    
    Returns:
        Combined content of all files with clear separators
    """
    try:
        results = []
        for file_path in file_paths:
            normalized_path = normalize_path(file_path)
            content = read_local_file(normalized_path)
            results.append(f"=== {normalized_path} ===\n{content}")
            
            # Enhanced visual feedback
            lines_count = len(content.splitlines())
            size_kb = len(content.encode('utf-8')) / 1024
            console.print(f"       📖 Reading: [bright_cyan]{normalized_path}[/bright_cyan] ({size_kb:.1f} KB, {lines_count} lines)")
        
        combined_content = "\n\n".join(results)
        console.print(f"       📚 Read {len(file_paths)} files successfully")
        return combined_content
    except Exception as e:
        error_msg = f"Error reading files: {str(e)}"
        console.print(f"       [bold red]❌ Failed to read files: {error_msg}[/bold red]")
        return error_msg

def show_code_preview(content: str, file_path: str, max_lines: int = 25):
    """Show a collapsible code preview with diff-style line numbers."""
    lines = content.splitlines()
    total_lines = len(lines)
    
    if total_lines <= max_lines:
        # Show full content with line numbers
        numbered_content = "\n".join([f"[dim]{i+1:>3}[/dim] [green]+[/green] {line}" for i, line in enumerate(lines)])
        console.print(Panel(
            numbered_content,
            border_style="dim blue",
            padding=(1, 2),
            title=f"[bold bright_cyan]✨ {file_path} (New file - {total_lines} lines)[/bold bright_cyan]",
            title_align="left"
        ))
    else:
        # Show preview with line numbers
        preview_lines = lines[:max_lines]
        numbered_preview = "\n".join([f"[dim]{i+1:>3}[/dim] [green]+[/green] {line}" for i, line in enumerate(preview_lines)])
        console.print(Panel(
            f"{numbered_preview}\n\n[dim]... {total_lines - max_lines} more lines ...[/dim]\n[bright_yellow]💡 Full content written to file[/bright_yellow]",
            border_style="dim blue",
            padding=(1, 2),
            title=f"[bold bright_cyan]✨ {file_path} (New file preview - {max_lines}/{total_lines} lines)[/bold bright_cyan]",
            title_align="left"
        ))

def create_file_tool(file_path: str, content: str) -> str:
    """Create a new file or overwrite an existing file with the provided content.
    
    Args:
        file_path: The path where the file should be created
        content: The content to write to the file
    
    Returns:
        Success message or error message
    """
    try:
        # Use standardized file stats display
        show_file_stats(file_path, content, "📝 Creating")
        
        # Show code preview with display path
        display_path = get_display_path(file_path)
        show_code_preview(content, display_path)
        
        create_file(file_path, content)
        
        stats = calculate_file_stats(content)
        return f"Successfully created file '{display_path}' ({stats['lines']} lines, {stats['size_kb']:.1f} KB)"
    except Exception as e:
        return handle_tool_error("create file", file_path, e)

def create_multiple_files_tool(files: List[dict]) -> str:
    """Create multiple files at once.
    
    Args:
        files: List of dictionaries with 'path' and 'content' keys
    
    Returns:
        Success message with file count or error message
    """
    try:
        created_files = []
        total_size = 0
        total_lines = 0
        
        for file_data in files:
            file_path = file_data['path']
            content = file_data['content']
            
            lines_count = len(content.splitlines())
            size_kb = len(content.encode('utf-8')) / 1024
            
            console.print(f"       📝 Creating: [bright_cyan]{file_path}[/bright_cyan]")
            console.print(f"       📊 Content: {size_kb:.1f} KB, {lines_count} lines")
            
            # Show code preview
            show_code_preview(content, file_path)
            
            create_file(file_path, content)
            created_files.append(file_path)
            total_size += size_kb
            total_lines += lines_count
        
        console.print(f"       📚 Created {len(created_files)} files successfully")
        return f"Successfully created {len(created_files)} files ({total_lines} total lines, {total_size:.1f} KB total)"
    except Exception as e:
        error_msg = f"Error creating files: {str(e)}"
        console.print(f"       [bold red]❌ Failed to create files: {error_msg}[/bold red]")
        return error_msg

def show_edit_preview(original_snippet: str, new_snippet: str, file_path: str):
    """Show a preview of the edit being made with diff-style formatting."""
    original_lines = original_snippet.splitlines()
    new_lines = new_snippet.splitlines()
    
    # Create a diff-style preview
    preview_table = Table(
        title=f"📝 Edit Preview: {file_path}",
        show_header=True,
        header_style="bold bright_yellow",
        border_style="bright_yellow"
    )
    preview_table.add_column("Before (Original)", style="red dim", width=50)
    preview_table.add_column("After (Modified)", style="green", width=50)
    
    # Format before content with line numbers and removal indicators
    before_content = "\n".join([f"[dim]{i+1:>3}[/dim] [red]-[/red] {line}" for i, line in enumerate(original_lines[:10])])
    if len(original_lines) > 10:
        before_content += f"\n[dim]... {len(original_lines) - 10} more lines ...[/dim]"
    
    # Format after content with line numbers and addition indicators  
    after_content = "\n".join([f"[dim]{i+1:>3}[/dim] [green]+[/green] {line}" for i, line in enumerate(new_lines[:10])])
    if len(new_lines) > 10:
        after_content += f"\n[dim]... {len(new_lines) - 10} more lines ...[/dim]"
    
    preview_table.add_row(before_content, after_content)
    console.print(preview_table)

def edit_file_tool(file_path: str, original_snippet: str, new_snippet: str) -> str:
    """Edit an existing file by replacing a specific snippet with new content.
    
    Args:
        file_path: The path to the file to edit
        original_snippet: The exact text snippet to find and replace
        new_snippet: The new text to replace the original snippet with
    
    Returns:
        Success message or error message
    """
    try:
        console.print(f"       ✏️  Editing: [bright_cyan]{file_path}[/bright_cyan]")
        
        # Show edit preview
        show_edit_preview(original_snippet, new_snippet, file_path)
        
        apply_diff_edit(file_path, original_snippet, new_snippet)
        
        return f"Successfully edited file '{file_path}' - replaced {len(original_snippet)} chars with {len(new_snippet)} chars"
    except Exception as e:
        error_msg = f"Error editing file '{file_path}': {str(e)}"
        console.print(f"       [bold red]❌ Failed to edit file: {error_msg}[/bold red]")
        return error_msg

def list_directory_tool(path: str, ignore: Optional[List[str]] = None, respect_git_ignore: bool = True, use_smart_excludes: bool = True) -> str:
    """List files and subdirectories within a specific directory.
    
    Args:
        path: The absolute path of the directory to inspect
        ignore: List of glob patterns to exclude from results (in addition to smart defaults)
        respect_git_ignore: If true, ignores files specified in .gitignore
        use_smart_excludes: If true, automatically excludes common irrelevant directories
    
    Returns:
        Formatted listing of directory contents
    """
    try:
        normalized_path = normalize_path(path)
        if not os.path.isdir(normalized_path):
            error_msg = f"Path '{path}' is not a directory or does not exist"
            console.print(f"       [bold red]❌ {error_msg}[/bold red]")
            return error_msg
            
        ignore_patterns = ignore or []
        gitignore_patterns = []
        
        # Add smart default exclusions
        if use_smart_excludes:
            smart_excludes = get_smart_excludes()
            ignore_patterns.extend(smart_excludes)
        
        # Load .gitignore patterns if requested
        if respect_git_ignore:
            gitignore_path = os.path.join(normalized_path, '.gitignore')
            if os.path.exists(gitignore_path):
                try:
                    with open(gitignore_path, 'r') as f:
                        gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                except Exception:
                    pass  # Ignore gitignore parsing errors
        
        all_ignore_patterns = ignore_patterns + gitignore_patterns
        
        items = []
        dirs = []
        files = []
        excluded_count = 0
        
        for item in os.listdir(normalized_path):
            item_path = os.path.join(normalized_path, item)
            
            # Check if item should be ignored using the smart exclusion function
            if should_exclude_path(item_path, all_ignore_patterns):
                excluded_count += 1
                continue
                
            if os.path.isdir(item_path):
                dirs.append(f"📁 {item}/")
            else:
                # Get file size
                try:
                    size = os.path.getsize(item_path)
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024*1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    files.append(f"📄 {item} ({size_str})")
                except Exception:
                    files.append(f"📄 {item}")
        
        # Sort and combine
        dirs.sort()
        files.sort()
        items = dirs + files
        
        display_path = get_display_path(normalized_path)
        console.print(f"       📂 Listing: [bright_cyan]{display_path}[/bright_cyan]")
        console.print(f"       📊 Found: {len(dirs)} directories, {len(files)} files")
        if excluded_count > 0:
            console.print(f"       🚫 Excluded: {excluded_count} items (build artifacts, dependencies, etc.)")
        
        if items:
            # Show preview in a table
            listing_table = Table(
                title=f"📂 Directory Contents: {os.path.basename(normalized_path)}",
                show_header=False,
                border_style="bright_blue"
            )
            listing_table.add_column("Items", style="bright_cyan")
            
            # Show first 20 items in preview
            preview_items = items[:20]
            for item in preview_items:
                listing_table.add_row(item)
            
            if len(items) > 20:
                listing_table.add_row(f"[dim]... and {len(items) - 20} more items[/dim]")
            
            console.print(listing_table)
        
        result = f"Directory listing for: {normalized_path}\n"
        result += f"Total: {len(dirs)} directories, {len(files)} files\n"
        if excluded_count > 0:
            result += f"Excluded: {excluded_count} items (automatically filtered)\n"
        result += "\n" + "\n".join(items)
        
        return result
        
    except Exception as e:
        error_msg = f"Error listing directory '{path}': {str(e)}"
        console.print(f"       [bold red]❌ Failed to list directory: {error_msg}[/bold red]")
        return error_msg

def search_file_content_tool(pattern: str, include: Optional[str] = None, path: Optional[str] = None, use_smart_excludes: bool = True) -> str:
    """Search for a text pattern inside files using regex.
    
    Args:
        pattern: The regular expression pattern to search for
        include: Glob pattern to filter which files are searched
        path: Directory to search within (defaults to current directory)
        use_smart_excludes: If true, automatically excludes common irrelevant files/directories
    
    Returns:
        Search results with file paths and matching lines
    """
    try:
        search_path = normalize_path(path) if path else os.getcwd()
        include_pattern = include or "**/*"
        
        console.print(f"       🔍 Searching: [bright_cyan]{search_path}[/bright_cyan]")
        console.print(f"       📋 Pattern: [yellow]{pattern}[/yellow]")
        console.print(f"       📁 Files: [dim]{include_pattern}[/dim]")
        
        # Compile regex pattern
        try:
            regex = re.compile(pattern, re.MULTILINE)
        except re.error as e:
            error_msg = f"Invalid regex pattern: {str(e)}"
            console.print(f"       [bold red]❌ {error_msg}[/bold red]")
            return error_msg
        
        matches = []
        files_searched = 0
        files_excluded = 0
        
        # Get smart exclusions
        smart_excludes = get_smart_excludes() if use_smart_excludes else []
        
        # Get list of files to search
        if include_pattern.startswith('**'):
            # Recursive search
            for root, dirs, files in os.walk(search_path):
                # Filter directories using smart exclusions
                if use_smart_excludes:
                    dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(root, d), smart_excludes)]
                else:
                    # Skip hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, search_path)
                    
                    # Skip using smart exclusions
                    if use_smart_excludes and should_exclude_path(file_path, smart_excludes):
                        files_excluded += 1
                        continue
                    
                    # Skip hidden files if not using smart excludes
                    if not use_smart_excludes and file.startswith('.'):
                        continue
                        
                    # Check if file matches include pattern
                    if include and not fnmatch.fnmatch(rel_path, include_pattern.replace('**/', '')):
                        continue
                    
                    # Skip binary files
                    if is_binary_file(file_path):
                        continue
                        
                    files_searched += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for i, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                matches.append({
                                    'file': rel_path,
                                    'line_num': i,
                                    'line_content': line.strip()
                                })
                                
                    except Exception:
                        continue  # Skip files that can't be read
        else:
            # Single directory search
            pattern_path = os.path.join(search_path, include_pattern)
            for file_path in glob_module.glob(pattern_path):
                if os.path.isfile(file_path):
                    # Skip using smart exclusions
                    if use_smart_excludes and should_exclude_path(file_path, smart_excludes):
                        files_excluded += 1
                        continue
                    
                    # Skip binary files
                    if is_binary_file(file_path):
                        continue
                        
                    files_searched += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        rel_path = os.path.relpath(file_path, search_path)
                        for i, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                matches.append({
                                    'file': rel_path,
                                    'line_num': i,
                                    'line_content': line.strip()
                                })
                                
                    except Exception:
                        continue
        
        console.print(f"       📊 Searched {files_searched} files, found {len(matches)} matches")
        if files_excluded > 0:
            console.print(f"       🚫 Excluded {files_excluded} files (build artifacts, dependencies, etc.)")
        
        # Show preview of matches
        if matches:
            preview_table = Table(
                title=f"🔍 Search Results: '{pattern}'",
                show_header=True,
                header_style="bold bright_yellow",
                border_style="bright_yellow"
            )
            preview_table.add_column("File", style="bright_cyan", width=30)
            preview_table.add_column("Line", style="bold white", width=5)
            preview_table.add_column("Content", style="dim white", width=50)
            
            # Show first 10 matches
            for match in matches[:10]:
                preview_table.add_row(
                    match['file'],
                    str(match['line_num']),
                    match['line_content'][:80] + ('...' if len(match['line_content']) > 80 else '')
                )
            
            if len(matches) > 10:
                preview_table.add_row("[dim]...[/dim]", "[dim]...[/dim]", f"[dim]... and {len(matches) - 10} more matches[/dim]")
            
            console.print(preview_table)
        
        # Format results
        result = f"Search results for pattern: {pattern}\n"
        result += f"Searched {files_searched} files in: {search_path}\n"
        result += f"Found {len(matches)} matches\n\n"
        
        for match in matches:
            result += f"{match['file']}:{match['line_num']}: {match['line_content']}\n"
        
        return result
        
    except Exception as e:
        error_msg = f"Error searching files: {str(e)}"
        console.print(f"       [bold red]❌ Failed to search: {error_msg}[/bold red]")
        return error_msg

def glob_tool(pattern: str, case_sensitive: bool = False, path: Optional[str] = None, recursive: bool = True, respect_git_ignore: bool = True, use_smart_excludes: bool = True) -> str:
    """Find files and directories that match a specified glob pattern.
    
    Args:
        pattern: The glob pattern to match against file paths
        case_sensitive: If true, pattern matching will be case-sensitive
        path: Directory to start the search from (defaults to current directory)
        recursive: If true, searches directories recursively (defaults to True)
        respect_git_ignore: If true, ignores files matching .gitignore patterns
        use_smart_excludes: If true, automatically excludes common irrelevant files/directories
    
    Returns:
        List of matching file paths
    """
    try:
        search_path = normalize_path(path) if path else os.getcwd()
        
        console.print(f"       🔍 Globbing: [bright_cyan]{search_path}[/bright_cyan]")
        console.print(f"       📋 Pattern: [yellow]{pattern}[/yellow]")
        
        # Handle case sensitivity
        if not case_sensitive and os.name == 'nt':  # Windows
            pattern = pattern.lower()
        
        gitignore_patterns = []
        if respect_git_ignore:
            gitignore_path = os.path.join(search_path, '.gitignore')
            if os.path.exists(gitignore_path):
                try:
                    with open(gitignore_path, 'r') as f:
                        gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                except Exception:
                    pass
        
        # Get smart exclusions
        smart_excludes = get_smart_excludes() if use_smart_excludes else []
        all_excludes = gitignore_patterns + smart_excludes
        
        # Use glob to find matches
        full_pattern = os.path.join(search_path, pattern)
        raw_matches = glob_module.glob(full_pattern, recursive=recursive)
        
        # Filter matches
        matches = []
        excluded_count = 0
        for match in raw_matches:
            rel_path = os.path.relpath(match, search_path)

            
            # Check exclusions (gitignore + smart excludes)
            if should_exclude_path(match, all_excludes):
                excluded_count += 1
                continue
            
            # Additional gitignore pattern check for compatibility
            should_ignore = False
            for ignore_pattern in gitignore_patterns:
                if fnmatch.fnmatch(rel_path, ignore_pattern):
                    should_ignore = True
                    break
            
            if not should_ignore:
                # Determine file type and size
                if os.path.isdir(match):
                    matches.append(f"📁 {rel_path}/")
                else:
                    try:
                        size = os.path.getsize(match)
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024*1024:
                            size_str = f"{size/1024:.1f}KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f}MB"
                        matches.append(f"📄 {rel_path} ({size_str})")
                    except Exception:
                        matches.append(f"📄 {rel_path}")
        
        matches.sort()
        
        console.print(f"       📊 Found {len(matches)} matches")
        if excluded_count > 0:
            console.print(f"       🚫 Excluded {excluded_count} items (build artifacts, dependencies, etc.)")
        
        # Show preview
        if matches:
            preview_table = Table(
                title=f"🔍 Glob Results: '{pattern}'",
                show_header=False,
                border_style="bright_green"
            )
            preview_table.add_column("Matches", style="bright_cyan")
            
            # Show first 15 matches
            for match in matches[:15]:
                preview_table.add_row(match)
                
            if len(matches) > 15:
                preview_table.add_row(f"[dim]... and {len(matches) - 15} more matches[/dim]")
            
            console.print(preview_table)
        
        result = f"Glob pattern: {pattern}\n"
        result += f"Search path: {search_path}\n"
        result += f"Found {len(matches)} matches\n\n"
        result += "\n".join(matches)
        
        return result
        
    except Exception as e:
        error_msg = f"Error with glob pattern '{pattern}': {str(e)}"
        console.print(f"       [bold red]❌ Failed to glob: {error_msg}[/bold red]")
        return error_msg

def replace_tool(file_path: str, new_string: str, old_string: str, expected_replacements: int = 1) -> str:
    """Replace a specific string of text within a file.
    
    Args:
        file_path: The absolute path to the file to modify
        new_string: The new text that will replace the old_string
        old_string: The exact, literal text to be replaced
        expected_replacements: Number of expected replacements (defaults to 1)
    
    Returns:
        Success message or error message
    """
    try:
        normalized_path = normalize_path(file_path)
        console.print(f"       🔄 Replacing in: [bright_cyan]{normalized_path}[/bright_cyan]")
        
        # Read file content
        if not os.path.exists(normalized_path):
            error_msg = f"File '{file_path}' does not exist"
            console.print(f"       [bold red]❌ {error_msg}[/bold red]")
            return error_msg
            
        with open(normalized_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check how many replacements would be made
        actual_count = content.count(old_string)
        
        if actual_count == 0:
            error_msg = f"Text to replace not found in file"
            console.print(f"       [bold red]❌ {error_msg}[/bold red]")
            return error_msg
        
        if actual_count != expected_replacements:
            warning_msg = f"Expected {expected_replacements} replacements, but found {actual_count} occurrences"
            console.print(f"       [bold yellow]⚠ {warning_msg}[/bold yellow]")
            
            if actual_count > expected_replacements:
                console.print(f"       [dim]Will replace all {actual_count} occurrences[/dim]")
        
        # Show replacement preview
        show_edit_preview(old_string, new_string, normalized_path)
        
        # Perform replacement
        new_content = content.replace(old_string, new_string)
        
        # Write back to file
        with open(normalized_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        console.print(f"       ✅ Successfully replaced {actual_count} occurrence(s)")
        
        return f"Successfully replaced {actual_count} occurrence(s) in '{file_path}'"
        
    except Exception as e:
        error_msg = f"Error replacing text in file '{file_path}': {str(e)}"
        console.print(f"       [bold red]❌ Failed to replace: {error_msg}[/bold red]")
        return error_msg

def write_file_tool(content: str, file_path: str) -> str:
    """Write content to a file, creating it if it doesn't exist or overwriting if it does.
    
    Args:
        content: The content to write into the file
        file_path: The absolute path of the file to write to
    
    Returns:
        Success message or error message
    """
    # This is essentially identical to create_file_tool - reuse its logic
    return create_file_tool(file_path, content).replace("created", "wrote")

def read_many_files_tool(paths: List[str], exclude: Optional[List[str]] = None, include: Optional[List[str]] = None, recursive: bool = True, respect_git_ignore: bool = True, useDefaultExcludes: bool = True) -> str:
    """Read the contents of multiple files at once based on paths or glob patterns.
    
    Args:
        paths: List of file paths or glob patterns
        exclude: Glob patterns for files/directories to exclude from reading
        include: Glob patterns to specifically include (overrides exclusions)
        recursive: If true, searches directories recursively
        respect_git_ignore: If true, respects .gitignore rules
        useDefaultExcludes: If true, excludes common directories like node_modules and .git
    
    Returns:
        Combined content of all matching files with clear separators
    """
    try:
        console.print(f"       📚 Reading multiple files: {len(paths)} pattern(s)")
        
        exclude_patterns = exclude or []
        include_patterns = include or []
        
        # Default exclusions using comprehensive smart excludes
        if useDefaultExcludes:
            smart_excludes = get_smart_excludes()
            exclude_patterns.extend(smart_excludes)
        
        # Load gitignore patterns
        gitignore_patterns = []
        if respect_git_ignore:
            try:
                with open('.gitignore', 'r') as f:
                    gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                exclude_patterns.extend(gitignore_patterns)
            except Exception:
                pass
        
        all_files = []
        
        # Process each path/pattern
        for path_pattern in paths:
            if os.path.isfile(path_pattern):
                # Direct file path
                all_files.append(path_pattern)
            elif os.path.isdir(path_pattern):
                # Directory - walk recursively if requested
                if recursive:
                    for root, dirs, files in os.walk(path_pattern):
                        # Filter directories
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                        
                        for file in files:
                            if not file.startswith('.'):
                                all_files.append(os.path.join(root, file))
                else:
                    # Just files in the directory
                    try:
                        for item in os.listdir(path_pattern):
                            item_path = os.path.join(path_pattern, item)
                            if os.path.isfile(item_path) and not item.startswith('.'):
                                all_files.append(item_path)
                    except Exception:
                        continue
            else:
                # Glob pattern
                matches = glob_module.glob(path_pattern, recursive=recursive)
                all_files.extend([f for f in matches if os.path.isfile(f)])
        
        # Remove duplicates and apply filters
        unique_files = list(set(all_files))
        filtered_files = []
        
        for file_path in unique_files:
            rel_path = os.path.relpath(file_path)
            
            # Check exclusions using smart exclusion function
            if should_exclude_path(file_path, exclude_patterns):
                continue
            
            # Check inclusions (overrides exclusions)
            should_include = not include_patterns  # If no include patterns, include by default
            for pattern in include_patterns:
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_path, pattern):
                    should_include = True
                    break
            
            # Skip binary files
            if should_include and not is_binary_file(file_path):
                filtered_files.append(file_path)
        
        filtered_files.sort()
        
        console.print(f"       📊 Processing {len(filtered_files)} files")
        
        # Read files and combine content
        results = []
        total_size = 0
        total_lines = 0
        skipped_files = []
        
        for file_path in filtered_files:
            try:
                rel_path = os.path.relpath(file_path)
                content = read_local_file(file_path)
                lines_count = len(content.splitlines())
                size_kb = len(content.encode('utf-8')) / 1024
                
                results.append(f"=== {rel_path} ===\n{content}")
                total_size += size_kb
                total_lines += lines_count
                
                console.print(f"       📄 Read: [bright_cyan]{rel_path}[/bright_cyan] ({size_kb:.1f} KB, {lines_count} lines)")
                
            except Exception as e:
                skipped_files.append(f"{file_path}: {str(e)}")
                continue
        
        console.print(f"       ✅ Successfully read {len(results)} files")
        if skipped_files:
            console.print(f"       ⚠ Skipped {len(skipped_files)} files due to errors")
        
        combined_content = "\n\n".join(results)
        
        # Show summary
        summary_table = Table(
            title="📚 Multi-file Read Summary",
            show_header=True,
            header_style="bold bright_green",
            border_style="bright_green"
        )
        summary_table.add_column("Metric", style="bold white")
        summary_table.add_column("Value", style="bright_cyan")
        
        summary_table.add_row("Files processed", str(len(results)))
        summary_table.add_row("Total size", f"{total_size:.1f} KB")
        summary_table.add_row("Total lines", str(total_lines))
        if skipped_files:
            summary_table.add_row("Files skipped", str(len(skipped_files)))
        
        console.print(summary_table)
        
        return combined_content
        
    except Exception as e:
        error_msg = f"Error reading multiple files: {str(e)}"
        console.print(f"       [bold red]❌ Failed to read multiple files: {error_msg}[/bold red]")
        return error_msg

def run_shell_command_tool(command: str, description: Optional[str] = None, directory: Optional[str] = None) -> str:
    """Execute a command in the system's shell.
    
    Args:
        command: The exact command to be executed
        description: Brief description of the command's purpose
        directory: Directory in which to run the command (defaults to current directory)
    
    Returns:
        Command output and execution details
    """
    try:
        work_dir = normalize_path(directory) if directory else os.getcwd()
        
        console.print(f"       ⚡ Executing: [yellow]{command}[/yellow]")
        if description:
            console.print(f"       📝 Purpose: [dim]{description}[/dim]")
        console.print(f"       📁 Directory: [bright_cyan]{work_dir}[/bright_cyan]")
        
        # Execute command with timeout
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
                encoding='utf-8',
                errors='ignore'
            )
            
            execution_time = time.time() - start_time
            
            # Format output
            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            
            output = "\n\n".join(output_parts) if output_parts else "(No output)"
            
            # Show execution results
            if result.returncode == 0:
                console.print(f"       ✅ Command completed successfully (exit code: 0)")
                console.print(f"       ⏱️  Execution time: {execution_time:.2f} seconds")
                
                # Show output preview
                if output.strip():
                    output_preview = output[:500] + ('...' if len(output) > 500 else '')
                    console.print(Panel(
                        output_preview,
                        title=f"[bold bright_green]📤 Command Output[/bold bright_green]",
                        border_style="bright_green",
                        padding=(1, 2)
                    ))
            else:
                console.print(f"       ❌ Command failed (exit code: {result.returncode})")
                console.print(f"       ⏱️  Execution time: {execution_time:.2f} seconds")
                
                # Show error preview
                error_preview = output[:500] + ('...' if len(output) > 500 else '')
                console.print(Panel(
                    error_preview,
                    title=f"[bold bright_red]❌ Command Error[/bold bright_red]",
                    border_style="bright_red",
                    padding=(1, 2)
                ))
            
            # Format result
            result_text = f"Command: {command}\n"
            result_text += f"Directory: {work_dir}\n"
            result_text += f"Exit code: {result.returncode}\n"
            result_text += f"Execution time: {execution_time:.2f} seconds\n\n"
            result_text += output
            
            return result_text
            
        except subprocess.TimeoutExpired:
            console.print(f"       ⏰ Command timed out after 60 seconds")
            return f"Command timed out: {command}\nDirectory: {work_dir}\nTimeout: 60 seconds"
            
    except Exception as e:
        error_msg = f"Error executing command '{command}': {str(e)}"
        console.print(f"       [bold red]❌ Failed to execute command: {error_msg}[/bold red]")
        return error_msg

# --------------------------------------------------------------------------------
# 3. system prompt
# --------------------------------------------------------------------------------
SYSTEM_PROMPT = dedent("""\
    You are an elite software engineer called Gemini Engineer with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.

    IMPORTANT SECURITY CONSTRAINT:
    - All file operations are restricted to a designated working directory for security
    - You cannot access or modify files outside this workspace
    - This prevents accidental system file access and ensures safe operation
    - Relative paths are resolved within the working directory
    - Absolute paths outside the working directory will be rejected
    - Users can change the working directory during the session using the /path command

    Core capabilities:
    1. Code Analysis & Discussion
       - Analyze code with expert-level insight
       - Explain complex concepts clearly
       - Suggest optimizations and best practices
       - Debug issues with precision

    2. File Operations (via function calls):
       - read_file: Read a single file's content with optional line limits and offset
       - read_multiple_files: Read multiple files at once
       - read_many_files: Read multiple files based on paths or glob patterns with advanced filtering
       - create_file: Create or overwrite a single file
       - create_multiple_files: Create multiple files at once
       - write_file: Write content to a file (alias for create_file with different parameter order)
       - edit_file: Make precise edits to existing files using snippet replacement
       - replace: Replace specific text within a file with expected replacement count validation

    3. Directory & File System Operations:
       - list_directory: List files and subdirectories with smart exclusions for build artifacts
       - glob: Find files and directories matching glob patterns with smart filtering
       - search_file_content: Search for regex patterns with automatic exclusion of irrelevant files

    4. System Operations:
       - run_shell_command: Execute shell commands with timeout and directory control

    Guidelines:
    1. Provide natural, conversational responses explaining your reasoning
    2. Use function calls when you need to read or modify files or explore the file system
    3. For file operations:
       - Always read files first before editing them to understand the context
       - Use precise snippet matching for edits
       - Explain what changes you're making and why
       - Consider the impact of changes on the overall codebase
       - Use glob and search_file_content for discovering relevant files
       - Remember all operations are restricted to the working directory
    4. Follow language-specific best practices
    5. Suggest tests or validation steps when appropriate
    6. Be thorough in your analysis and recommendations
    7. Use appropriate tools for the task:
       - Use glob for finding files by pattern
       - Use search_file_content for finding specific content across files
       - Use list_directory for exploring directory structures
       - Use read_many_files for bulk file reading with pattern matching
       - Use run_shell_command for system operations when needed
       - All file system tools automatically exclude common build artifacts and dependencies
    8. If you encounter path restriction errors, explain the security constraint to the user

    IMPORTANT: In your thinking process, if you realize that something requires a tool call, cut your thinking short and proceed directly to the tool call. Don't overthink - act efficiently when file operations are needed.

    Remember: You're a senior engineer working within a secure, sandboxed environment - be thoughtful, precise, and explain your reasoning clearly.
""")

# --------------------------------------------------------------------------------
# 4. Helper functions 
# --------------------------------------------------------------------------------

def calculate_file_stats(content: str) -> Dict[str, Any]:
    """Calculate common file statistics."""
    lines_count = len(content.splitlines())
    size_kb = len(content.encode('utf-8')) / 1024
    return {
        'lines': lines_count,
        'size_kb': size_kb,
        'size_bytes': len(content.encode('utf-8'))
    }

def show_file_stats(file_path: str, content: str, operation: str = "📊 Content") -> None:
    """Display standardized file statistics."""
    stats = calculate_file_stats(content)
    display_path = get_display_path(file_path)
    console.print(f"       {operation}: [bright_cyan]{display_path}[/bright_cyan]")
    console.print(f"       📊 Size: {stats['size_kb']:.1f} KB, {stats['lines']} lines")

def handle_tool_error(operation: str, identifier: str, error: Exception) -> str:
    """Standardized error handling for tools."""
    error_msg = f"Error {operation} '{identifier}': {str(error)}"
    console.print(f"       [bold red]❌ Failed to {operation}: {error_msg}[/bold red]")
    return error_msg

def create_function_dispatcher() -> Dict[str, callable]:
    """Create a function dispatcher to replace manual if/elif chains."""
    return {
        "read_file_tool": read_file_tool,
        "read_multiple_files_tool": read_multiple_files_tool, 
        "read_many_files_tool": read_many_files_tool,
        "create_file_tool": create_file_tool,
        "create_multiple_files_tool": create_multiple_files_tool,
        "write_file_tool": write_file_tool,
        "edit_file_tool": edit_file_tool,
        "replace_tool": replace_tool,
        "list_directory_tool": list_directory_tool,
        "glob_tool": glob_tool,
        "search_file_content_tool": search_file_content_tool,
        "run_shell_command_tool": run_shell_command_tool
    }

def apply_smart_exclusions(files: List[str], exclude_patterns: List[str] = None, 
                          use_smart_excludes: bool = True, respect_git_ignore: bool = True,
                          base_path: str = ".") -> List[str]:
    """Apply consistent exclusion logic across tools."""
    exclude_patterns = exclude_patterns or []
    
    if use_smart_excludes:
        exclude_patterns.extend(get_smart_excludes())
    
    if respect_git_ignore:
        gitignore_path = os.path.join(base_path, '.gitignore')
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, 'r') as f:
                    gitignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    exclude_patterns.extend(gitignore_patterns)
            except Exception:
                pass
    
    filtered_files = []
    for file_path in files:
        if not should_exclude_path(file_path, exclude_patterns) and not is_binary_file(file_path):
            filtered_files.append(file_path)
    
    return filtered_files

def standardize_token_estimation(content: str) -> int:
    """Unified token estimation logic."""
    # Use the more sophisticated estimation from RateLimiter but make it standalone
    total_chars = len(content)
    # Rough estimate: ~4 characters per token (varies by language/content)
    estimated_tokens = total_chars // 4
    return estimated_tokens

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file with robust encoding handling."""
    # Try different encodings to handle various file types
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise OSError(f"Cannot read file '{file_path}': {str(e)}")
    
    # If all encodings fail, try reading as binary and decode with error handling
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            return content.decode('utf-8', errors='replace')
    except Exception as e:
        raise OSError(f"Cannot read file '{file_path}' with any encoding: {str(e)}")

def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    file_path = Path(path)
    
    # Security checks
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    normalized_path = Path(normalize_path(str(file_path)))
    
    # Validate reasonable file size for operations
    if len(content) > 5_000_000:  # 5MB limit
        raise ValueError("File content exceeds 5MB size limit")
    
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with open(normalized_path, "w", encoding="utf-8") as f:
        f.write(content)

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    if not files_to_edit:
        return
    
    table = Table(title="📝 Proposed Edits", show_header=True, header_style="bold bright_blue", show_lines=True, border_style="blue")
    table.add_column("File Path", style="bright_cyan", no_wrap=True)
    table.add_column("Original", style="red dim")
    table.add_column("New", style="bright_green")

    for edit in files_to_edit:
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)
    
    console.print(table)

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    """Reads the file at 'path', replaces the first occurrence of 'original_snippet' with 'new_snippet', then overwrites."""
    try:
        content = read_local_file(path)
        
        # Verify we're replacing the exact intended occurrence
        occurrences = content.count(original_snippet)
        if occurrences == 0:
            raise ValueError("Original snippet not found")
        if occurrences > 1:
            console.print(f"[bold yellow]⚠ Multiple matches ({occurrences}) found - requiring line numbers for safety[/bold yellow]")
            console.print("[dim]Use format:\n--- original.py (lines X-Y)\n+++ modified.py[/dim]")
            raise ValueError(f"Ambiguous edit: {occurrences} matches")
        
        updated_content = content.replace(original_snippet, new_snippet, 1)
        create_file(path, updated_content)
        console.print(f"[bold blue]✓[/bold blue] Applied diff edit to '[bright_cyan]{path}[/bright_cyan]'")

    except FileNotFoundError:
        console.print(f"[bold red]✗[/bold red] File not found for diff editing: '[bright_cyan]{path}[/bright_cyan]'")
    except ValueError as e:
        console.print(f"[bold yellow]⚠[/bold yellow] {str(e)} in '[bright_cyan]{path}[/bright_cyan]'. No changes made.")
        console.print("\n[bold blue]Expected snippet:[/bold blue]")
        console.print(Panel(original_snippet, title="Expected", border_style="blue", title_align="left"))
        console.print("\n[bold blue]Actual file content:[/bold blue]")
        console.print(Panel(content, title="Actual", border_style="yellow", title_align="left"))

def try_handle_add_command(user_input: str) -> bool:
    global WORKING_DIRECTORY
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        path_to_add = user_input[len(prefix):].strip()
        try:
            # Show working directory context
            if WORKING_DIRECTORY:
                console.print(f"[dim]Working directory: {WORKING_DIRECTORY}[/dim]")
            
            normalized_path = normalize_path(path_to_add)
            if os.path.isdir(normalized_path):
                # Handle entire directory
                add_directory_to_conversation(normalized_path)
            else:
                # Handle a single file as before
                content = read_local_file(normalized_path)
                
                # Try to cache large files for cost optimization (automatic size checking)
                try:
                        cache_name = cache_manager.create_file_cache(normalized_path, content)
                    # The cache manager will handle size checking and provide appropriate feedback
                except Exception as e:
                    console.print(f"[dim yellow]⚠ File caching failed: {e}[/dim yellow]")
                
                conversation_history.append({
                    "role": "user",
                    "parts": [{"text": f"Content of file '{normalized_path}':\n\n{content}"}]
                })
                
                # Show relative path for better UX
                if WORKING_DIRECTORY:
                    try:
                        rel_path = os.path.relpath(normalized_path, WORKING_DIRECTORY)
                        console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{rel_path}[/bright_cyan]' to conversation.\n")
                    except ValueError:
                        console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
                else:
                    console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
                    
        except ValueError as e:
            # This will catch working directory restriction errors
            console.print(f"[bold red]✗[/bold red] Access denied: {e}\n")
        except OSError as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False

def try_handle_history_command(user_input: str) -> bool:
    if user_input.strip().lower() == "/history":
        console.print("\n[bold bright_blue]📋 Conversation History[/bold bright_blue]")
        
        if len(conversation_history) == 0:  # No conversation history yet
            console.print("[dim]No conversation history yet.[/dim]\n")
            return True
            
        history_table = Table(
            title="💬 Chat History",
            show_header=True,
            header_style="bold bright_blue",
            border_style="bright_blue"
        )
        history_table.add_column("Turn", style="bold white", width=6)
        history_table.add_column("Role", style="bright_cyan", width=10)
        history_table.add_column("Content Preview", style="dim white", width=60)
        
        turn = 0
        for i, msg in enumerate(conversation_history):
            turn += 1
            role = "🤖 Assistant" if msg["role"] == "model" else "👤 User"
            
            # Extract text content from parts
            content_preview = ""
            for part in msg["parts"]:
                if isinstance(part, dict) and "text" in part:
                    content_preview = part["text"][:100] + ("..." if len(part["text"]) > 100 else "")
                    break
                elif hasattr(part, 'text') and part.text:
                    content_preview = part.text[:100] + ("..." if len(part.text) > 100 else "")
                    break
                elif hasattr(part, 'function_call'):
                    content_preview = f"🔧 Function: {part.function_call.name}"
                    break
            
            history_table.add_row(f"{turn}", role, content_preview)
        
        console.print(history_table)
        console.print(f"\n[dim]Total messages: {len(conversation_history)}[/dim]\n")
        return True
    return False

def add_directory_to_conversation(directory_path: str):
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        # Use comprehensive smart exclusions instead of the old limited list
        smart_excludes = get_smart_excludes()
        skipped_files = []
        added_files = []
        total_files_processed = 0
        max_files = 1000  # Reasonable limit for files to process
        max_file_size = 5_000_000  # 5MB limit

        for root, dirs, files in os.walk(directory_path):
            if total_files_processed >= max_files:
                console.print(f"[bold yellow]⚠[/bold yellow] Reached maximum file limit ({max_files})")
                break

            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            # Filter directories using smart exclusions
            dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(root, d), smart_excludes)]

            for file in files:
                if total_files_processed >= max_files:
                    break

                full_path = os.path.join(root, file)
                
                # Skip using smart exclusions
                if should_exclude_path(full_path, smart_excludes):
                    skipped_files.append(full_path)
                    continue

                try:
                    # Check file size before processing
                    if os.path.getsize(full_path) > max_file_size:
                        skipped_files.append(f"{full_path} (exceeds size limit)")
                        continue

                    # Check if it's binary
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue

                    normalized_path = normalize_path(full_path)
                    content = read_local_file(normalized_path)
                    conversation_history.append({
                        "role": "user",
                        "parts": [{"text": f"Content of file '{normalized_path}':\n\n{content}"}]
                    })
                    added_files.append(normalized_path)
                    total_files_processed += 1

                except OSError:
                    skipped_files.append(full_path)

        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]' to conversation.")
        if added_files:
            console.print(f"\n[bold bright_blue]📁 Added files:[/bold bright_blue] [dim]({len(added_files)} of {total_files_processed})[/dim]")
            for f in added_files:
                console.print(f"  [bright_cyan]📄 {f}[/bright_cyan]")
        if skipped_files:
            console.print(f"\n[bold yellow]⏭ Skipped files:[/bold yellow] [dim]({len(skipped_files)})[/dim]")
            for f in skipped_files[:10]:  # Show only first 10 to avoid clutter
                console.print(f"  [yellow dim]⚠ {f}[/yellow dim]")
            if len(skipped_files) > 10:
                console.print(f"  [dim]... and {len(skipped_files) - 10} more[/dim]")
        console.print()

def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(peek_size)
        # If there is a null byte in the sample, treat it as binary
        if b'\0' in chunk:
            return True
        return False
    except Exception:
        # If we fail to read, just treat it as binary to be safe
        return True

def ensure_file_in_context(file_path: str) -> bool:
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in part.get("text", "") for msg in conversation_history for part in msg.get("parts", [])):
            conversation_history.append({
                "role": "user",
                "parts": [{"text": f"{file_marker}:\n\n{content}"}]
            })
        return True
    except OSError:
        console.print(f"[bold red]✗[/bold red] Could not read file '[bright_cyan]{file_path}[/bright_cyan]' for editing context")
        return False

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path with security checks and working directory enforcement."""
    global WORKING_DIRECTORY
    
    # Convert to Path object for easier manipulation
    path = Path(path_str)
    
    # If working directory is set, enforce restrictions
    if WORKING_DIRECTORY:
        working_dir = Path(WORKING_DIRECTORY).resolve()
        
        # If path is relative, make it relative to working directory
        if not path.is_absolute():
            path = working_dir / path
        
        # Resolve the path to handle any .. or . components
        try:
            resolved_path = path.resolve()
        except Exception:
            raise ValueError(f"Invalid path: {path_str}")
        
        # Ensure the resolved path is within the working directory
        try:
            resolved_path.relative_to(working_dir)
        except ValueError:
            raise ValueError(f"Access denied: Path '{path_str}' is outside the allowed working directory '{WORKING_DIRECTORY}'")
        
        return str(resolved_path)
    else:
        # Original behavior when no working directory is set
        path = path.resolve()
    
    # Prevent directory traversal attacks
    if ".." in path.parts:
        raise ValueError(f"Invalid path: {path_str} contains parent directory references")
    
    return str(path)

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = []  # Only user and model messages, system instruction goes in config

# --------------------------------------------------------------------------------
# 6. Google GenAI API interaction with streaming
# --------------------------------------------------------------------------------

def trim_conversation_history():
    """Trim conversation history to prevent token limit issues while preserving tool call sequences"""
    if len(conversation_history) <= 20:  # Don't trim if conversation is still small
        return
        
    # Keep only the last 15 messages to prevent token overflow
    if len(conversation_history) > 15:
        conversation_history[:] = conversation_history[-15:]

def validate_and_fix_conversation_history():
    """Validate and fix conversation history to prevent function call flow errors."""
    global conversation_history
    
    if not conversation_history:
        return
    
    fixed_history = []
    pending_function_calls = []
    
    for i, msg in enumerate(conversation_history):
        role = msg.get("role")
        parts = msg.get("parts", [])
        
        # Check for function calls in model messages
        has_function_calls = False
        has_text = False
        
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                has_function_calls = True
                pending_function_calls.append(part.function_call.name)
            elif (isinstance(part, dict) and "text" in part) or (hasattr(part, 'text') and part.text):
                has_text = True
        
        # Check for function responses in user messages
        has_function_responses = False
        for part in parts:
            if hasattr(part, 'function_response') or (hasattr(part, 'name') and hasattr(part, 'response')):
                has_function_responses = True
                if pending_function_calls:
                    pending_function_calls.pop(0)  # Remove one pending call
        
        # Add message to fixed history
        fixed_history.append(msg)
    
    # If there are unmatched function calls, remove the problematic messages
    if pending_function_calls:
        console.print(f"[dim yellow]🔧 Fixing {len(pending_function_calls)} unmatched function calls in conversation history[/dim yellow]")
        
        # Remove messages from the end until all function calls are matched
        while pending_function_calls and fixed_history:
            last_msg = fixed_history[-1]
            if last_msg.get("role") == "model":
                # Check if this message has unmatched function calls
                for part in last_msg.get("parts", []):
                    if hasattr(part, 'function_call') and part.function_call:
                        if part.function_call.name in pending_function_calls:
                            pending_function_calls.remove(part.function_call.name)
                            fixed_history.pop()
                            break
                else:
                    break
            else:
                break
    
    # Update the global conversation history
    conversation_history[:] = fixed_history

def check_conversation_structure() -> bool:
    """Check if conversation history has proper structure for API calls."""
    if not conversation_history:
        return True
    
    function_calls_waiting_response = []
    
    for i, msg in enumerate(conversation_history):
        role = msg.get("role")
        parts = msg.get("parts", [])
        
        if role == "model":
            # Check for function calls
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls_waiting_response.append(i)
        elif role == "user":
            # Check for function responses
            for part in parts:
                if hasattr(part, 'function_response') or (hasattr(part, 'name') and hasattr(part, 'response')):
                    if function_calls_waiting_response:
                        function_calls_waiting_response.pop(0)
    
    # If there are unmatched function calls, the structure is invalid
    if function_calls_waiting_response:
        console.print(f"[dim yellow]⚠ Conversation structure issue: {len(function_calls_waiting_response)} unmatched function calls[/dim yellow]")
        return False
    
    return True

def stream_gemini_response(user_message: str):
    try:
        console.print("\n[bold bright_blue]🤖 Gemini Engineer>[/bold bright_blue] ", end="")
        
        # Apply rate limiting before making the request
        rate_limiter.wait_if_needed(conversation_history, user_message)
        
        console.print("\n[bold bright_blue]🧠 Thinking and analyzing request...[/bold bright_blue]")
        
        # Add user message to conversation history
        conversation_history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })
        
        # Validate and fix conversation history to prevent function call flow errors
        validate_and_fix_conversation_history()
        
        # Check conversation structure and fix if needed
        if not check_conversation_structure():
            console.print(f"[bold yellow]🔧 Fixing conversation structure issues[/bold yellow]")
            validate_and_fix_conversation_history()
        
        # Trim conversation history if it gets too long
        trim_conversation_history()
        
        # Try to create cache for the total context (only if it's worth caching)
        context_cache_name = None
        
        # Check if the TOTAL context (system + conversation + tools + message) justifies caching
        conversation_text = "".join(
            part.get("text", "") if isinstance(part, dict) else (getattr(part, 'text', None) or "")
            for msg in conversation_history 
            for part in msg.get("parts", [])
        )
        total_context_tokens = cache_manager._estimate_token_count(
            SYSTEM_PROMPT + user_message + conversation_text
        ) + 1000  # Add buffer for tools
        
        console.print(f"[dim]💾 Analyzing context: {total_context_tokens:,} tokens total (system + conversation + tools + message)[/dim]")
        
        # Check for recent function calls that might conflict with caching
        has_recent_function_calls = False
        if len(conversation_history) >= 2:
            for msg in conversation_history[-2:]:  # Check last 2 messages
                for part in msg.get("parts", []):
                    if (hasattr(part, 'function_call') and part.function_call) or \
                       (hasattr(part, 'function_response') or (hasattr(part, 'name') and hasattr(part, 'response'))):
                        has_recent_function_calls = True
                        break
                if has_recent_function_calls:
                    break
        
        if has_recent_function_calls:
            console.print(f"[dim yellow]💾 Context caching disabled - recent function calls detected[/dim yellow]")
        elif total_context_tokens >= cache_manager.min_tokens_for_caching and len(conversation_history) > 2:
            try:
                context_cache_name = cache_manager.create_context_cache(conversation_history, user_message)
                if context_cache_name:
                    console.print(f"[dim green]✅ Context cache created - will reduce costs for this large conversation[/dim green]")
                else:
                    console.print(f"[dim yellow]⚠ Context caching skipped - insufficient conversation history for effective caching[/dim yellow]")
            except Exception as e:
                console.print(f"[dim yellow]⚠ Context caching failed: {e}[/dim yellow]")
        else:
            if total_context_tokens < cache_manager.min_tokens_for_caching:
                console.print(f"[dim]💾 Context too small for caching ({total_context_tokens:,} < {cache_manager.min_tokens_for_caching:,} tokens)[/dim]")
            else:
                console.print(f"[dim]💾 Insufficient conversation history for caching[/dim]")
        
        # Prepare request configuration
        config_params = {
            "tools": [
                read_file_tool, read_multiple_files_tool, read_many_files_tool,
                create_file_tool, create_multiple_files_tool, write_file_tool,
                edit_file_tool, replace_tool,
                list_directory_tool, glob_tool, search_file_content_tool,
                run_shell_command_tool
            ],
            "temperature": 0.0,
            "automatic_function_calling": {'disable': True},
            "thinking_config": types.ThinkingConfig(
                thinking_budget=-1,  # Dynamic thinking - model decides based on complexity
                include_thoughts=True  # Enable thought summaries
            )
        }
        
        # Use cached context if available, otherwise regular system instruction
        if context_cache_name:
            config_params["cached_content"] = context_cache_name
            # When using cache, send only recent messages (cache contains older context)
            # The current user message is already in conversation_history
            recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
        else:
            config_params["system_instruction"] = SYSTEM_PROMPT
            # When not using cache, send full conversation history (current message already added)
            recent_messages = conversation_history
        
        # Enable thinking for complex reasoning with function calls with streaming
        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=recent_messages,  # Use appropriate message set (cached or full history)
                config=types.GenerateContentConfig(**config_params)
            )
        except Exception as e:
            # Handle function call flow errors specifically
            if "function response turn" in str(e).lower() or "invalid_argument" in str(e).lower():
                console.print(f"[bold yellow]🔧 Function call flow error detected - attempting to fix conversation state[/bold yellow]")
                
                # Clear conversation history of any incomplete function call sequences
                cleaned_history = []
                for msg in conversation_history:
                    role = msg.get("role")
                    parts = msg.get("parts", [])
                    
                    # Only keep messages that don't have function calls or have complete function call pairs
                    has_function_call = any(hasattr(part, 'function_call') and part.function_call for part in parts)
                    has_function_response = any(hasattr(part, 'function_response') or 
                                              (hasattr(part, 'name') and hasattr(part, 'response')) for part in parts)
                    
                    # Keep text-only messages and complete function sequences
                    if not has_function_call or (role == "user" and has_function_response):
                        # For model messages with function calls, only keep if there's a corresponding response
                        if role == "model" and has_function_call:
                            # Check if next message in history has function response
                            current_index = conversation_history.index(msg)
                            if current_index + 1 < len(conversation_history):
                                next_msg = conversation_history[current_index + 1]
                                next_has_response = any(hasattr(part, 'function_response') or 
                                                      (hasattr(part, 'name') and hasattr(part, 'response')) 
                                                      for part in next_msg.get("parts", []))
                                if next_has_response:
                                    cleaned_history.append(msg)
                            # If it's the last message and has function calls, skip it
                        else:
                            cleaned_history.append(msg)
                
                # Update conversation history
                conversation_history[:] = cleaned_history
                console.print(f"[dim yellow]🔧 Cleaned conversation history - {len(conversation_history)} messages remaining[/dim yellow]")
                
                # Retry with cleaned history (without caching to avoid further complications)
                config_params_retry = config_params.copy()
                config_params_retry.pop("cached_content", None)  # Remove caching
                config_params_retry["system_instruction"] = SYSTEM_PROMPT
                
                try:
                    response_stream = client.models.generate_content_stream(
                        model="gemini-2.5-flash",
                        contents=conversation_history,  # Use cleaned full history
                        config=types.GenerateContentConfig(**config_params_retry)
                    )
                except Exception as retry_error:
                    # If retry also fails, fall back to a fresh conversation with just the current message
                    console.print(f"[bold yellow]🔧 Retry failed - starting fresh conversation[/bold yellow]")
                    console.print(f"[dim yellow]Original error: {str(e)}[/dim yellow]")
                    console.print(f"[dim yellow]Retry error: {str(retry_error)}[/dim yellow]")
                    
                    # Save current conversation for debugging if needed
                    if len(conversation_history) > 1:
                        console.print(f"[dim]💾 Previous conversation had {len(conversation_history)} messages[/dim]")
                    
                    # Reset to just the current user message
                    conversation_history[:] = [{"role": "user", "parts": [{"text": user_message}]}]
                    
                    # Simple config without caching or complex features
                    simple_config = {
                        "tools": config_params["tools"],
                        "temperature": 0.0,
                        "automatic_function_calling": {'disable': True},
                        "system_instruction": SYSTEM_PROMPT
                    }
                    
                    response_stream = client.models.generate_content_stream(
                        model="gemini-2.5-flash",
                        contents=conversation_history,
                        config=types.GenerateContentConfig(**simple_config)
                    )
            else:
                # Re-raise other exceptions
                raise

        # Collect the complete response from streaming
        response = None
        thinking_parts = []
        response_parts = []
        all_text_parts = []  # Collect all non-thinking text parts
        
        console.print("\n[bold bright_magenta]🧠 Streaming Thoughts:[/bold bright_magenta]")
        current_thought = ""
        
        for chunk in response_stream:
            if (hasattr(chunk, 'candidates') and chunk.candidates and 
                len(chunk.candidates) > 0 and chunk.candidates[0] is not None):
                candidate = chunk.candidates[0]
                if (hasattr(candidate, 'content') and candidate.content is not None and 
                    hasattr(candidate.content, 'parts') and candidate.content.parts is not None):
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
                            # Stream thinking in real-time
                            current_thought += part.text
                            console.print(f"[dim bright_magenta]{part.text}[/dim bright_magenta]", end="")
                            thinking_parts.append(part.text)
                        elif hasattr(part, 'text') and part.text and not getattr(part, 'thought', False):
                            # Collect all non-thinking text parts
                            all_text_parts.append(part.text)
                            response_parts.append(part.text)
                        elif hasattr(part, 'function_call') and part.function_call:
                            response = chunk  # Save the response with function calls
            response = chunk  # Keep updating with latest chunk
        
        if current_thought:
            console.print("\n")

        # Check for function calls in the response
        function_calls = []
        if (hasattr(response, 'candidates') and response.candidates and 
            len(response.candidates) > 0 and response.candidates[0] is not None):
            candidate = response.candidates[0]
            if (hasattr(candidate, 'content') and candidate.content is not None and 
                hasattr(candidate.content, 'parts') and candidate.content.parts is not None):
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)

                # Handle multiple rounds of function calling
        total_function_calls = 0
        round_number = 1
        max_rounds = 25  # Prevent infinite loops
        
        while function_calls and round_number <= max_rounds:
            console.print(f"\n[bold bright_cyan]⚡ Round {round_number}: {len(function_calls)} function call(s) detected[/bold bright_cyan]")
            console.print("[dim]Executing functions step by step...[/dim]\n")
            
            # Add the assistant response to conversation history
            conversation_history.append({
                "role": "model",
                "parts": response.candidates[0].content.parts
            })
            
            # Execute each function call and collect all responses
            function_response_parts = []
            
            for i, function_call in enumerate(function_calls, 1):
                total_function_calls += 1
                console.print(f"[bold blue]📋 Step {total_function_calls}: Calling {function_call.name}[/bold blue]")
                
                # Show function call details (truncated to avoid clutter)
                truncated_args = {}
                for k, v in function_call.args.items():
                    if isinstance(v, str) and len(v) > 50:
                        truncated_args[k] = f"{v[:50]}..."
                    elif isinstance(v, list) and len(str(v)) > 100:
                        truncated_args[k] = f"[{len(v)} items]"
                    else:
                        truncated_args[k] = v
                
                args_display = ", ".join([f"{k}={truncated_args[k]}" for k in truncated_args.keys()])
                console.print(f"[bright_cyan]   → {function_call.name}({args_display})[/bright_cyan]")
                
                # Execute the function using dispatcher
                with console.status(f"[bold yellow]⏳ Executing {function_call.name}...[/bold yellow]"):
                    try:
                        dispatcher = create_function_dispatcher()
                        if function_call.name in dispatcher:
                            result = dispatcher[function_call.name](**function_call.args)
                        else:
                            result = f"Unknown function: {function_call.name}"
                        
                        console.print(f"[bold green]   ✓ {function_call.name} completed successfully[/bold green]")
                        console.print(f"[dim]   Result: {result[:100]}{'...' if len(result) > 100 else ''}[/dim]\n")
                        
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        console.print(f"[bold red]   ✗ {function_call.name} failed: {e}[/bold red]\n")

                # Collect function response (don't add to conversation yet)
                function_response_part = types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": result}
                )
                function_response_parts.append(function_response_part)
            
            # Add ALL function responses as a single conversation turn (critical for API compliance)
            # This fixes the "function response turn must come immediately after function call" error
            # by batching all function responses into one user message instead of separate messages
            conversation_history.append({
                "role": "user", 
                "parts": function_response_parts
            })

            console.print(f"[bold bright_green]🎉 Round {round_number} completed ({len(function_calls)} function(s))![/bold bright_green]")
            
            # Check if model wants to make more function calls
            console.print(f"\n[bold bright_blue]🔄 Checking if more function calls needed (Round {round_number + 1})...[/bold bright_blue]\n")
            
            with console.status("[bold bright_blue]🧠 Determining next steps...[/bold bright_blue]"):
                try:
                    # Apply rate limiting for follow-up requests
                    rate_limiter.wait_if_needed(conversation_history, "")
                    
                    # Use the same caching configuration for follow-up requests
                    follow_up_config = config_params.copy()
                    
                    # For follow-up requests, we need to update the contents based on cache usage
                    if context_cache_name:
                        # If using cache, send only recent messages (not the full history)
                        follow_up_contents = conversation_history[-2:]  # Last 2 messages
                    else:
                        # If not using cache, send full conversation history
                        follow_up_contents = conversation_history
                    
                    next_response_stream = client.models.generate_content_stream(
                        model="gemini-2.5-flash",
                        contents=follow_up_contents,
                        config=types.GenerateContentConfig(**follow_up_config)
                    )
                    
                    # Collect the next response
                    next_response = None
                    next_all_text_parts = []
                    
                    console.print("[bold bright_magenta]🧠 Continued Thinking:[/bold bright_magenta]")
                    current_thought = ""
                    
                    for chunk in next_response_stream:
                        if (hasattr(chunk, 'candidates') and chunk.candidates and 
                            len(chunk.candidates) > 0 and chunk.candidates[0] is not None):
                            candidate = chunk.candidates[0]
                            if (hasattr(candidate, 'content') and candidate.content is not None and 
                                hasattr(candidate.content, 'parts') and candidate.content.parts is not None):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
                                        current_thought += part.text
                                        console.print(f"[dim bright_magenta]{part.text}[/dim bright_magenta]", end="")
                                    elif hasattr(part, 'text') and part.text and not getattr(part, 'thought', False):
                                        next_all_text_parts.append(part.text)
                                    elif hasattr(part, 'function_call') and part.function_call:
                                        next_response = chunk
                        next_response = chunk
                    
                    if current_thought:
                        console.print("\n")
                    
                    # Update rate limiter with actual usage from follow-up request
                    if (next_response is not None and hasattr(next_response, 'usage_metadata') and 
                        next_response.usage_metadata is not None):
                        follow_up_tokens = getattr(next_response.usage_metadata, 'total_token_count', 0) or 0
                        if follow_up_tokens > 0:
                            rate_limiter.update_actual_usage(follow_up_tokens)
                    
                    # Check for more function calls in the next response
                    function_calls = []
                    if (hasattr(next_response, 'candidates') and next_response.candidates and 
                        len(next_response.candidates) > 0 and next_response.candidates[0] is not None):
                        candidate = next_response.candidates[0]
                        if (hasattr(candidate, 'content') and candidate.content is not None and 
                            hasattr(candidate.content, 'parts') and candidate.content.parts is not None):
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    function_calls.append(part.function_call)
                    
                    response = next_response  # Update response for next iteration
                    round_number += 1
                    
                    if not function_calls:
                        # No more function calls, show final response
                        complete_response_text = "".join(next_all_text_parts)
                        
                        if complete_response_text:
                            console.print("[bold bright_blue]🤖 Final Response:[/bold bright_blue]")
                            console.print(Panel(
                                complete_response_text,
                                border_style="bright_blue",
                                padding=(1, 2),
                                title="[bold bright_cyan]🎯 Complete Analysis[/bold bright_cyan]",
                                title_align="left"
                            ))
                            
                            # Add final response to conversation history
                            conversation_history.append({
                                "role": "model",
                                "parts": [{"text": complete_response_text}]
                            })
                        break
                    
                except Exception as e:
                    # Handle specific rate limiting errors during function calling
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        console.print(f"\n[bold red]🚦 Rate Limit Exceeded in Function Calling Round {round_number}[/bold red]")
                        console.print(f"[yellow]Please wait before continuing the conversation.[/yellow]")
                        console.print(f"[dim]Use '/status' to check current rate limits[/dim]")
                        break
                    elif "quota" in str(e).lower() or "exceeded" in str(e).lower():
                        console.print(f"\n[bold red]💳 API Quota Exceeded in Function Calling Round {round_number}[/bold red]")
                        console.print(f"[yellow]You have exceeded your API quota limits.[/yellow]")
                        break
                    else:
                        console.print(f"[bold red]❌ Error in function calling round {round_number}: {str(e)}[/bold red]")
                        break
        
        if round_number > max_rounds:
            console.print(f"[bold yellow]⚠ Reached maximum function calling rounds ({max_rounds}). Stopping to prevent infinite loops.[/bold yellow]")
        
        # Show final execution summary
        if total_function_calls > 0:
            console.print(f"\n[bold bright_green]📊 Total function calls executed: {total_function_calls}[/bold bright_green]")
            
        else:
            # No function calls, just regular response
            console.print("\n[bold bright_green]💬 Direct response (no function calls needed)[/bold bright_green]")
            
            # Use collected text parts for the complete response
            complete_response_text = "".join(all_text_parts)
            
            if complete_response_text:
                console.print(Panel(
                    complete_response_text,
                    border_style="bright_green",
                    padding=(1, 2),
                    title="[bold bright_cyan]🤖 Gemini Response[/bold bright_cyan]",
                    title_align="left"
                ))
                
                # Add complete response to conversation history
                conversation_history.append({
                    "role": "model", 
                    "parts": [{"text": complete_response_text}]
                })
            else:
                # Fallback to response.text if available
                if hasattr(response, 'text') and response.text:
                    console.print(Panel(
                        response.text,
                        border_style="bright_green",
                        padding=(1, 2),
                        title="[bold bright_cyan]🤖 Gemini Response[/bold bright_cyan]",
                        title_align="left"
                    ))
                    
                    # Add response to conversation history
                    conversation_history.append({
                        "role": "model", 
                        "parts": [{"text": response.text}]
                    })
                else:
                    console.print("[bold yellow]⚠ No response text captured[/bold yellow]")

        # Update rate limiter with actual token usage from the main response
        if (response is not None and hasattr(response, 'usage_metadata') and 
            response.usage_metadata is not None):
            actual_tokens = getattr(response.usage_metadata, 'total_token_count', 0) or 0
            if actual_tokens > 0:
                rate_limiter.update_actual_usage(actual_tokens)

        # Show token usage with thinking metrics and rate limiting info at the end
        if (response is not None and hasattr(response, 'usage_metadata') and 
            response.usage_metadata is not None):
            usage = response.usage_metadata
            thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
            output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            total_tokens = getattr(usage, 'total_token_count', 0) or 0
            
            if thinking_tokens > 0 or output_tokens > 0 or total_tokens > 0:
                usage_table = Table(
                    title="🧮 Token Usage & Rate Limiting",
                    show_header=True,
                    header_style="bold bright_yellow",
                    border_style="bright_yellow"
                )
                usage_table.add_column("Metric", style="bold white", width=20)
                usage_table.add_column("Current", style="bright_cyan", width=12)
                usage_table.add_column("Limit", style="dim white", width=12)
                usage_table.add_column("Usage %", style="bright_green", width=10)
                
                thinking_pct = (thinking_tokens / total_tokens * 100) if total_tokens > 0 else 0
                output_pct = (output_tokens / total_tokens * 100) if total_tokens > 0 else 0
                
                # Get current rate limiter status
                status = rate_limiter.get_status()
                
                usage_table.add_row("Thinking Tokens", f"{thinking_tokens:,}", f"{total_tokens:,}", f"{thinking_pct:.1f}%")
                usage_table.add_row("Output Tokens", f"{output_tokens:,}", f"{total_tokens:,}", f"{output_pct:.1f}%")
                usage_table.add_row("Total Tokens", f"{total_tokens:,}", "—", "100.0%")
                
                # Add cache hit information if available
                cached_input_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                if cached_input_tokens > 0:
                    cache_pct = (cached_input_tokens / total_tokens * 100) if total_tokens > 0 else 0
                    usage_table.add_row("💾 Cached Tokens", f"{cached_input_tokens:,}", f"{total_tokens:,}", f"{cache_pct:.1f}%")
                
                usage_table.add_row("", "", "", "")  # Separator
                usage_table.add_row("Requests/Min", f"{status['requests_this_minute']}", f"{status['rpm_limit']}", f"{status['rpm_usage_percent']:.1f}%")
                usage_table.add_row("Tokens/Min", f"{status['tokens_this_minute']:,}", f"{status['tpm_limit']:,}", f"{status['tpm_usage_percent']:.1f}%")
                usage_table.add_row("Requests/Day", f"{status['requests_today']}", f"{status['rpd_limit']}", f"{status['rpd_usage_percent']:.1f}%")
                
                console.print(usage_table)
                console.print()

        return {"success": True}

    except Exception as e:
        error_msg = f"Gemini API error: {str(e)}"
        
        # Handle specific rate limiting errors
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            console.print(f"\n[bold red]🚦 Rate Limit Exceeded[/bold red]")
            console.print(f"[yellow]The request exceeded API rate limits. Please wait before making more requests.[/yellow]")
            console.print(f"[dim]Technical details: {error_msg}[/dim]")
            
            # Show current rate limit status
            try:
                status = rate_limiter.get_status()
                console.print(f"\n[bold yellow]Current Usage:[/bold yellow]")
                console.print(f"• Requests this minute: {status['requests_this_minute']}/{status['rpm_limit']}")
                console.print(f"• Tokens this minute: {status['tokens_this_minute']:,}/{status['tpm_limit']:,}")
                console.print(f"• Requests today: {status['requests_today']}/{status['rpd_limit']}")
                console.print(f"\n[dim]Use '/status' for detailed rate limiting information[/dim]")
            except Exception:
                pass
            
            return {"error": "Rate limit exceeded", "retry_after": 60}
        
        # Handle quota exceeded errors
        elif "quota" in str(e).lower() or "exceeded" in str(e).lower():
            console.print(f"\n[bold red]💳 API Quota Exceeded[/bold red]")
            console.print(f"[yellow]You have exceeded your API quota limits.[/yellow]")
            console.print(f"[dim]Please check your plan and billing details at: https://ai.google.dev/gemini-api/docs/rate-limits[/dim]")
            console.print(f"\n[dim]Technical details: {error_msg}[/dim]")
            return {"error": "Quota exceeded"}
        
        # Handle other API errors
        else:
            console.print(f"\n[bold red]❌ {error_msg}[/bold red]")
            return {"error": error_msg}

# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------

def main():
    global WORKING_DIRECTORY
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Gemini Engineer - AI Code Assistant with Function Calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python coder.py                           # Run in current directory
  python coder.py --dir /path/to/project    # Run in specific directory
  python coder.py -d ./my-project          # Run in relative directory
        """
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Set working directory (all file operations will be restricted to this directory)',
        metavar='PATH'
    )
    
    args = parser.parse_args()
    
    # Set working directory if provided
    if args.dir:
        try:
            working_path = Path(args.dir).resolve()
            if not working_path.exists():
                console.print(f"[bold red]❌ Error: Directory '{args.dir}' does not exist[/bold red]")
                return
            if not working_path.is_dir():
                console.print(f"[bold red]❌ Error: '{args.dir}' is not a directory[/bold red]")
                return
            
            WORKING_DIRECTORY = str(working_path)
            os.chdir(WORKING_DIRECTORY)  # Change to working directory
            
        except Exception as e:
            console.print(f"[bold red]❌ Error setting working directory: {e}[/bold red]")
            return
    else:
        WORKING_DIRECTORY = os.getcwd()
    
    # Create a beautiful gradient-style welcome panel
    welcome_text = """[bold bright_blue]🤖 Gemini Engineer[/bold bright_blue] [bright_cyan]with Function Calling[/bright_cyan]
[dim blue]Powered by Gemini 2.5 Flash with Advanced Reasoning & Context Caching[/dim blue]"""
    
    console.print(Panel.fit(
        welcome_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold bright_cyan]🤖 AI Code Assistant[/bold bright_cyan]",
        title_align="center"
    ))
    
    # Show working directory information
    working_dir_info = f"""[bold bright_green]📁 Working Directory:[/bold bright_green] [bright_cyan]{WORKING_DIRECTORY}[/bright_cyan]
[dim]All file operations will be restricted to this directory for security[/dim]"""
    
    console.print(Panel(
        working_dir_info,
        border_style="bright_green",
        padding=(1, 2),
        title="[bold bright_green]🔒 Security Scope[/bold bright_green]",
        title_align="left"
    ))
    
    # Create an elegant instruction panel
    instructions = """[bold bright_blue]📁 File Operations:[/bold bright_blue]
  • [bright_cyan]/add path/to/file[/bright_cyan] - Include a single file in conversation
  • [bright_cyan]/add path/to/folder[/bright_cyan] - Include all files in a folder
  • [dim]The AI can automatically read and create files using function calls[/dim]

[bold bright_blue]🎯 Commands:[/bold bright_blue]
  • [bright_cyan]/path [directory][/bright_cyan] - Change working directory or show current
  • [bright_cyan]/status[/bright_cyan] or [bright_cyan]/limits[/bright_cyan] - Show API rate limiting status and usage
  • [bright_cyan]/cache[/bright_cyan] or [bright_cyan]/cache clear[/bright_cyan] - Manage context caches
  • [bright_cyan]/help[/bright_cyan] or [bright_cyan]/h[/bright_cyan] - Show command line usage help
  • [bright_cyan]/history[/bright_cyan] - View conversation history
  • [bright_cyan]exit[/bright_cyan] or [bright_cyan]quit[/bright_cyan] - End the session
  • Just ask naturally - the AI will handle file operations automatically!

[bold bright_yellow]⚠ Security Note:[/bold bright_yellow]
  • All file operations are restricted to the working directory
  • The AI cannot access or modify files outside this scope
  • Use [bright_cyan]/path[/bright_cyan] to switch between project directories

[bold bright_red]🚦 Rate Limiting:[/bold bright_red]
  • Free Tier: 10 req/min, 250K tokens/min, 250 req/day
  • Automatic delays applied when approaching limits
  • Use [bright_cyan]/status[/bright_cyan] to monitor current usage

[bold bright_magenta]💾 Context Caching:[/bold bright_magenta]
  • Intelligent caching based on TOTAL context size (system + conversation + tools)
  • Only caches when entire context exceeds 1K tokens with substantial conversation history
  • Reduces token costs by ~75% for large conversations through smart context reuse"""
    
    console.print(Panel(
        instructions,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]💡 How to Use[/bold blue]",
        title_align="left"
    ))
    console.print()

    while True:
        try:
            user_input = prompt_session.prompt("🔵 You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold bright_blue]👋 Goodbye! Happy coding![/bold bright_blue]")
            break

        if try_handle_add_command(user_input):
            continue
            
        if try_handle_history_command(user_input):
            continue
            
        if try_handle_help_command(user_input):
            continue
            
        if try_handle_path_command(user_input):
            continue
            
        if try_handle_status_command(user_input):
            continue
            
        if try_handle_cache_command(user_input):
            continue

        response_data = stream_gemini_response(user_input)
        
        if response_data.get("error"):
            console.print(f"[bold red]❌ Error: {response_data['error']}[/bold red]")

    console.print("[bold blue]✨ Session finished. Thank you for using Gemini Engineer![/bold blue]")

def show_help():
    """Show help information about command line options."""
    help_text = """[bold bright_blue]🛠️  Command Line Options:[/bold bright_blue]

[bold bright_cyan]Working Directory:[/bold bright_cyan]
  [yellow]python coder.py[/yellow]                    # Run in current directory
  [yellow]python coder.py --dir /path/to/project[/yellow] # Run in specific directory  
  [yellow]python coder.py -d ./my-project[/yellow]        # Run in relative directory

[bold bright_cyan]Runtime Commands:[/bold bright_cyan]
  [yellow]/path[/yellow]                             # Show current working directory
  [yellow]/path /new/directory[/yellow]              # Change to new directory
  [yellow]/path ..[/yellow]                          # Go up one directory level
  [yellow]/path ~[/yellow]                           # Go to home directory
  [yellow]/add file.py[/yellow]                      # Add file to conversation
  [yellow]/status[/yellow] or [yellow]/limits[/yellow]                 # Show API rate limiting status
  [yellow]/cache[/yellow] or [yellow]/cache clear[/yellow]              # Manage context caches
  [yellow]/history[/yellow]                          # Show conversation history

[bold bright_cyan]Security Features:[/bold bright_cyan]
  • All file operations are restricted to the working directory
  • Prevents accidental access to system files
  • Safe for AI-assisted development
  • Use /path to switch between project directories

[bold bright_cyan]Rate Limiting & Optimization:[/bold bright_cyan]
  • Free Tier: 10 requests/min, 250K tokens/min, 250 requests/day
  • Automatic delays applied when approaching limits
  • Context caching reduces costs by ~75% for large conversations
  • Intelligent caching based on total context size analysis
  • Use [yellow]/status[/yellow] to monitor usage and cache statistics

[bold bright_cyan]Usage Examples:[/bold bright_cyan]
  [yellow]python coder.py -d ./my-react-app[/yellow]      # Work on React project
  [yellow]python coder.py --dir ~/Documents/code[/yellow]  # Work in specific folder
  [yellow]python coder.py[/yellow]                        # Work in current directory
  
  [dim]Then during session:[/dim]
  [yellow]/path ../other-project[/yellow]             # Switch to sibling project
  [yellow]/path ~/Desktop/new-project[/yellow]        # Switch to different project
  [yellow]/status[/yellow]                           # Check rate limiting status"""
    
    console.print(Panel(
        help_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="[bold bright_cyan]📚 Help & Usage[/bold bright_cyan]",
        title_align="left"
    ))

def try_handle_help_command(user_input: str) -> bool:
    """Handle /help command to show usage information."""
    if user_input.strip().lower() in ["/help", "/h", "help"]:
        show_help()
        return True
    return False

def get_display_path(file_path: str) -> str:
    """Get a user-friendly display path (relative to working directory if possible)."""
    global WORKING_DIRECTORY
    if WORKING_DIRECTORY:
        try:
            return os.path.relpath(file_path, WORKING_DIRECTORY)
        except ValueError:
            return file_path
    return file_path

def try_handle_path_command(user_input: str) -> bool:
    """Handle /path command to change or show the working directory."""
    global WORKING_DIRECTORY
    
    # Check for /path command variants
    path_prefixes = ["/path ", "/dir ", "/cd "]
    show_commands = ["/path", "/dir", "/cd", "/pwd"]
    
    user_input_lower = user_input.strip().lower()
    
    # Show current working directory
    if user_input_lower in show_commands:
        console.print(f"\n[bold bright_green]📁 Current Working Directory:[/bold bright_green]")
        console.print(f"[bright_cyan]{WORKING_DIRECTORY}[/bright_cyan]")
        
        # Show some directory info
        try:
            items = os.listdir(WORKING_DIRECTORY)
            dirs = [item for item in items if os.path.isdir(os.path.join(WORKING_DIRECTORY, item)) and not item.startswith('.')]
            files = [item for item in items if os.path.isfile(os.path.join(WORKING_DIRECTORY, item)) and not item.startswith('.')]
            
            console.print(f"[dim]Contains: {len(dirs)} directories, {len(files)} files[/dim]\n")
        except Exception:
            console.print()
            
        return True
    
    # Change working directory
    for prefix in path_prefixes:
        if user_input_lower.startswith(prefix):
            new_path = user_input[len(prefix):].strip()
            
            if not new_path:
                console.print("[bold red]❌ Please provide a directory path[/bold red]")
                console.print("[dim]Usage: /path <directory> or /path to show current directory[/dim]\n")
                return True
            
            try:
                # Handle special shortcuts
                if new_path == "..":
                    if WORKING_DIRECTORY:
                        new_path = os.path.dirname(WORKING_DIRECTORY)
                    else:
                        new_path = os.path.dirname(os.getcwd())
                elif new_path == "~":
                    new_path = os.path.expanduser("~")
                elif new_path.startswith("~/"):
                    new_path = os.path.expanduser(new_path)
                
                # Resolve the path
                if not os.path.isabs(new_path):
                    # Relative to current working directory
                    new_path = os.path.join(WORKING_DIRECTORY or os.getcwd(), new_path)
                
                new_path = os.path.abspath(new_path)
                
                # Validate the directory exists
                if not os.path.exists(new_path):
                    console.print(f"[bold red]❌ Directory does not exist: {new_path}[/bold red]\n")
                    return True
                    
                if not os.path.isdir(new_path):
                    console.print(f"[bold red]❌ Path is not a directory: {new_path}[/bold red]\n")
                    return True
                
                # Update working directory
                old_path = WORKING_DIRECTORY
                WORKING_DIRECTORY = new_path
                os.chdir(WORKING_DIRECTORY)
                
                console.print(f"[bold bright_green]✅ Working directory changed![/bold bright_green]")
                console.print(f"[dim]From:[/dim] [bright_red]{old_path or 'system default'}[/bright_red]")
                console.print(f"[dim]To:[/dim]   [bright_cyan]{WORKING_DIRECTORY}[/bright_cyan]")
                
                # Show some info about the new directory
                try:
                    items = os.listdir(WORKING_DIRECTORY)
                    dirs = [item for item in items if os.path.isdir(os.path.join(WORKING_DIRECTORY, item)) and not item.startswith('.')]
                    files = [item for item in items if os.path.isfile(os.path.join(WORKING_DIRECTORY, item)) and not item.startswith('.')]
                    
                    console.print(f"[dim]New directory contains: {len(dirs)} directories, {len(files)} files[/dim]")
                    
                    # Show first few items as preview
                    if dirs or files:
                        preview_items = (dirs[:3] + files[:3])[:5]
                        if preview_items:
                            console.print(f"[dim]Preview: {', '.join(preview_items)}{' ...' if len(dirs + files) > 5 else ''}[/dim]")
                except Exception:
                    pass
                
                console.print()
                
            except Exception as e:
                console.print(f"[bold red]❌ Error changing directory: {e}[/bold red]\n")
            
            return True
    
    return False

def try_handle_status_command(user_input: str) -> bool:
    """Handle /status command to show rate limiter status and usage statistics."""
    status_commands = ["/status", "/limits", "/usage", "/rate"]
    
    if user_input.strip().lower() in status_commands:
        status = rate_limiter.get_status()
        
        console.print("\n[bold bright_blue]📊 Rate Limiter Status[/bold bright_blue]")
        
        # Main usage table
        usage_table = Table(
            title="🚦 Current API Usage",
            show_header=True,
            header_style="bold bright_cyan",
            border_style="bright_cyan"
        )
        usage_table.add_column("Limit Type", style="bold white", width=15)
        usage_table.add_column("Current", style="bright_cyan", width=12)
        usage_table.add_column("Limit", style="dim white", width=12)
        usage_table.add_column("Usage %", style="bright_green", width=10)
        usage_table.add_column("Status", style="bold yellow", width=12)
        
        # Determine status colors and indicators
        def get_status_indicator(percentage):
            if percentage >= 90:
                return "[bold red]🚨 Critical[/bold red]"
            elif percentage >= 75:
                return "[bold yellow]⚠ High[/bold yellow]"
            elif percentage >= 50:
                return "[yellow]📈 Moderate[/yellow]"
            else:
                return "[green]✅ Good[/green]"
        
        # Add rows with status indicators
        usage_table.add_row(
            "Requests/Min",
            f"{status['requests_this_minute']}",
            f"{status['rpm_limit']}",
            f"{status['rpm_usage_percent']:.1f}%",
            get_status_indicator(status['rpm_usage_percent'])
        )
        usage_table.add_row(
            "Tokens/Min",
            f"{status['tokens_this_minute']:,}",
            f"{status['tpm_limit']:,}",
            f"{status['tpm_usage_percent']:.1f}%",
            get_status_indicator(status['tpm_usage_percent'])
        )
        usage_table.add_row(
            "Requests/Day",
            f"{status['requests_today']}",
            f"{status['rpd_limit']}",
            f"{status['rpd_usage_percent']:.1f}%",
            get_status_indicator(status['rpd_usage_percent'])
        )
        
        console.print(usage_table)
        
        # Session statistics
        stats_table = Table(
            title="📈 Session Statistics",
            show_header=True,
            header_style="bold bright_green",
            border_style="bright_green"
        )
        stats_table.add_column("Metric", style="bold white", width=20)
        stats_table.add_column("Value", style="bright_cyan", width=15)
        
        stats_table.add_row("Total Requests", f"{status['total_requests']:,}")
        stats_table.add_row("Total Tokens", f"{status['total_tokens']:,}")
        
        if status['total_delay_time'] > 0:
            delay_minutes = status['total_delay_time'] / 60
            if delay_minutes < 1:
                delay_str = f"{status['total_delay_time']:.1f} seconds"
            else:
                delay_str = f"{delay_minutes:.1f} minutes"
            stats_table.add_row("Total Delay Time", delay_str)
        else:
            stats_table.add_row("Total Delay Time", "0 seconds")
        
        # Calculate average tokens per request
        if status['total_requests'] > 0:
            avg_tokens = status['total_tokens'] / status['total_requests']
            stats_table.add_row("Avg Tokens/Request", f"{avg_tokens:,.0f}")
        
        console.print(stats_table)
        
        # Show cache statistics
        try:
            cache_stats = cache_manager.get_cache_stats()
            
            cache_table = Table(
                title="💾 Context Caching Statistics",
                show_header=True,
                header_style="bold bright_magenta",
                border_style="bright_magenta"
            )
            cache_table.add_column("Metric", style="bold white", width=20)
            cache_table.add_column("Value", style="bright_cyan", width=15)
            
            cache_table.add_row("Active Caches", f"{cache_stats['active_caches']:,}")
            cache_table.add_row("Total Created", f"{cache_stats['total_created']:,}")
            cache_table.add_row("Tokens Cached", f"{cache_stats['total_tokens_cached']:,}")
            
            if cache_stats['estimated_cost_savings'] > 0:
                cache_table.add_row("Est. Cost Savings", f"${cache_stats['estimated_cost_savings']:.3f}")
            
            # Show cache types breakdown
            if cache_stats['cache_types']:
                cache_types_str = ", ".join([f"{k}: {v}" for k, v in cache_stats['cache_types'].items()])
                cache_table.add_row("Cache Types", cache_types_str)
            
            console.print(cache_table)
            
        except Exception as e:
            console.print(f"[dim red]Cache stats unavailable: {e}[/dim red]")
        
        # Show time until limits reset
        now = datetime.now()
        minute_reset = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        day_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        time_to_minute = (minute_reset - now).total_seconds()
        time_to_day = (day_reset - now).total_seconds() / 3600  # in hours
        
        reset_info = f"""[bold bright_yellow]🕒 Reset Times:[/bold bright_yellow]
[dim]• Minute limits reset in: {time_to_minute:.0f} seconds
• Daily limits reset in: {time_to_day:.1f} hours[/dim]"""
        
        console.print(Panel(
            reset_info,
            border_style="bright_yellow",
            padding=(1, 2),
            title="[bold bright_yellow]⏰ Reset Schedule[/bold bright_yellow]",
            title_align="left"
        ))
        
        # Show warnings if approaching limits
        warnings = []
        if status['rpm_usage_percent'] >= 80:
            warnings.append("⚠ Approaching requests per minute limit")
        if status['tpm_usage_percent'] >= 80:
            warnings.append("⚠ Approaching tokens per minute limit")
        if status['rpd_usage_percent'] >= 80:
            warnings.append("⚠ Approaching daily request limit")
        
        if warnings:
            warning_text = "\n".join(warnings)
            console.print(Panel(
                warning_text,
                border_style="bold yellow",
                padding=(1, 2),
                title="[bold yellow]⚠ Warnings[/bold yellow]",
                title_align="left"
            ))
        
        console.print()
        return True
    
    return False

def try_handle_cache_command(user_input: str) -> bool:
    """Handle /cache command for cache management operations."""
    cache_commands = ["/cache", "/cache clear", "/cache stats", "/cache info"]
    
    user_input_lower = user_input.strip().lower()
    
    if user_input_lower == "/cache" or user_input_lower == "/cache info":
        # Show cache information
        try:
            cache_stats = cache_manager.get_cache_stats()
            
            console.print("\n[bold bright_magenta]💾 Context Cache Information[/bold bright_magenta]")
            
            info_table = Table(
                title="💾 Cache Overview",
                show_header=True,
                header_style="bold bright_magenta",
                border_style="bright_magenta"
            )
            info_table.add_column("Metric", style="bold white", width=25)
            info_table.add_column("Value", style="bright_cyan", width=20)
            info_table.add_column("Description", style="dim white", width=40)
            
            info_table.add_row(
                "Active Caches",
                f"{cache_stats['active_caches']:,}",
                "Currently active cache objects"
            )
            info_table.add_row(
                "Total Created",
                f"{cache_stats['total_created']:,}",
                "Total caches created this session"
            )
            info_table.add_row(
                "Tokens Cached",
                f"{cache_stats['total_tokens_cached']:,}",
                "Total tokens stored in caches"
            )
            info_table.add_row(
                "Min Cache Size",
                f"{cache_manager.min_tokens_for_caching:,} tokens",
                "Minimum content size for caching"
            )
            info_table.add_row(
                "Default TTL",
                f"{cache_manager.default_ttl_hours} hours",
                "Default cache expiration time"
            )
            
            if cache_stats['estimated_cost_savings'] > 0:
                info_table.add_row(
                    "Est. Cost Savings",
                    f"${cache_stats['estimated_cost_savings']:.3f}",
                    "Approximate cost reduction from caching"
                )
            
            console.print(info_table)
            
            # Show cache types if available
            if cache_stats['cache_types']:
                types_info = "\n".join([f"• {k}: {v} cache(s)" for k, v in cache_stats['cache_types'].items()])
                console.print(Panel(
                    types_info,
                    title="[bold bright_magenta]📋 Cache Types[/bold bright_magenta]",
                    border_style="bright_magenta",
                    padding=(1, 2),
                    title_align="left"
                ))
            
            console.print("[dim]Use '/cache clear' to clear all caches[/dim]\n")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error retrieving cache info: {e}[/bold red]\n")
        
        return True
    
    elif user_input_lower == "/cache clear":
        # Clear all caches
        try:
            with console.status("[bold yellow]🧹 Clearing all caches...[/bold yellow]"):
                cache_stats_before = cache_manager.get_cache_stats()
                
                # Clear all caches
                cleared_count = 0
                for cache_key, cache_obj in list(cache_manager._caches.items()):
                    try:
                        cache_manager.client.caches.delete(name=cache_obj.name)
                        cleared_count += 1
                    except Exception:
                        pass  # Ignore individual deletion errors
                
                # Reset cache manager state
                with cache_manager._lock:
                    cache_manager._caches.clear()
                    cache_manager._cache_metadata.clear()
            
            console.print(f"[bold green]✅ Cleared {cleared_count} cache(s)[/bold green]")
            console.print(f"[dim]Freed up {cache_stats_before['total_tokens_cached']:,} cached tokens[/dim]\n")
            
        except Exception as e:
            console.print(f"[bold red]❌ Error clearing caches: {e}[/bold red]\n")
        
        return True
    
    elif user_input_lower == "/cache stats":
        # Show detailed cache statistics (same as /status but cache-focused)
        try:
            cache_stats = cache_manager.get_cache_stats()
            
            console.print("\n[bold bright_magenta]💾 Cache Statistics[/bold bright_magenta]")
            
            # Detailed stats table
            stats_table = Table(
                title="📊 Detailed Cache Statistics",
                show_header=True,
                header_style="bold bright_magenta",
                border_style="bright_magenta"
            )
            stats_table.add_column("Metric", style="bold white", width=25)
            stats_table.add_column("Value", style="bright_cyan", width=15)
            
            stats_table.add_row("Active Caches", f"{cache_stats['active_caches']:,}")
            stats_table.add_row("Total Created", f"{cache_stats['total_created']:,}")
            stats_table.add_row("Tokens Cached", f"{cache_stats['total_tokens_cached']:,}")
            
            if cache_stats['estimated_cost_savings'] > 0:
                stats_table.add_row("Estimated Savings", f"${cache_stats['estimated_cost_savings']:.3f}")
            
            # Calculate efficiency metrics
            if cache_stats['total_created'] > 0:
                avg_tokens_per_cache = cache_stats['total_tokens_cached'] / cache_stats['total_created']
                stats_table.add_row("Avg Tokens/Cache", f"{avg_tokens_per_cache:,.0f}")
            
            console.print(stats_table)
            
            # Show cache type breakdown
            if cache_stats['cache_types']:
                type_table = Table(
                    title="📋 Cache Types Breakdown",
                    show_header=True,
                    header_style="bold bright_yellow",
                    border_style="bright_yellow"
                )
                type_table.add_column("Cache Type", style="bold white")
                type_table.add_column("Count", style="bright_cyan")
                
                for cache_type, count in cache_stats['cache_types'].items():
                    type_table.add_row(cache_type, f"{count}")
                
                console.print(type_table)
            
            console.print()
            
        except Exception as e:
            console.print(f"[bold red]❌ Error retrieving cache stats: {e}[/bold red]\n")
        
        return True
    
    return False

if __name__ == "__main__":
    main()