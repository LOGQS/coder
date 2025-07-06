# Gemini Engineer

🤖 **AI Code Assistant with Function Calling** - Powered by Google's Gemini 2.5 Flash

> **Note**: This is an unfinished project based on deepseek-engineer, adapted for the Gemini API. A modified and a lot more improved version of this will be integrated as a subsystem into ATLAS2.

## Overview

Gemini Engineer is an interactive AI coding assistant that provides intelligent code analysis, file operations, and project management through natural language conversations. It leverages Google's Gemini 2.5 Flash model with advanced reasoning capabilities and automatic function calling.

## Key Features

### 🔧 **Function Calling & File Operations**
- **Smart File Management**: Read, create, edit, and manage files with AI assistance
- **Directory Operations**: List, search, and navigate project structures
- **Intelligent Exclusions**: Automatic filtering of build artifacts, dependencies, and irrelevant files
- **Bulk Operations**: Process multiple files with pattern matching and glob support

### 🛡️ **Security & Safety**
- **Sandboxed Environment**: All operations restricted to designated working directory
- **Path Validation**: Prevents directory traversal and unauthorized access
- **Safe Execution**: Controlled shell command execution with timeouts

### ⚡ **Performance & Optimization**
- **Rate Limiting**: Intelligent API rate management (10 req/min, 250K tokens/min, 250 req/day)
- **Context Caching**: Smart caching reduces token costs by ~75% for large conversations
- **Streaming Responses**: Real-time thinking and response streaming
- **Token Optimization**: Automatic context size analysis and optimization

### 💬 **Interactive Experience**
- **Natural Language Interface**: Conversational interaction with the AI
- **Rich Console Output**: Beautiful, color-coded terminal interface
- **Command System**: Built-in commands for file management and system control
- **Session Management**: Conversation history and context preservation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Usage
```bash
# Run in current directory
python coder.py

# Run in specific project directory
python coder.py --dir /path/to/project
python coder.py -d ./my-project
```

### Interactive Commands
- `/add file.py` - Include file in conversation
- `/add ./src/` - Include entire directory
- `/path /new/directory` - Change working directory
- `/status` - Check API usage and rate limits
- `/cache` - Manage context caches
- `/history` - View conversation history
- `/help` - Show all available commands

### Example Session
```
🔵 You> Can you analyze my React app structure and suggest improvements?

🤖 Gemini Engineer> I'll analyze your React app structure. Let me start by exploring the project...

[AI automatically reads files, analyzes structure, and provides recommendations]
```

## Architecture

### Core Components
- **Rate Limiter**: Intelligent API quota management with sliding windows
- **Context Cache Manager**: Automatic caching for large conversations
- **Function Dispatcher**: Modular tool system for file operations
- **Security Layer**: Working directory enforcement and path validation

### Supported Operations
- File I/O (read, write, edit, replace)
- Directory listing and navigation
- Content search with regex support
- Shell command execution
- Glob pattern matching
- Multi-file operations

## Configuration

### Rate Limiting (Free Tier)
- **Requests per minute**: 10
- **Tokens per minute**: 250,000
- **Requests per day**: 250

### Context Caching
- **Minimum tokens**: 1,024 (Gemini 2.5 Flash)
- **Default TTL**: 6 hours
- **Auto-caching**: When total context exceeds thresholds

## Project Status

⚠️ **This project is currently unfinished** and serves as a proof-of-concept adaptation of deepseek-engineer for the Gemini API.

### Planned Integration
A refined version of this codebase will be integrated as a subsystem into **ATLAS2**, providing AI-assisted development capabilities within the larger ATLAS2 ecosystem.

### Known Limitations
- Some advanced features may not be fully implemented
- Error handling could be improved in certain scenarios
- Documentation and testing coverage is incomplete
