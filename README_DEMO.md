# Demo Guide for MCP Server

This guide shows how to run the demo script with OpenAI LLM.

## Setup

### 1. Install Dependencies

```bash
pip install python-dotenv
```

### 2. Create .env File

Copy the example file and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` file and update the values:

```env
OPENAI_BASE_URL=https://lqd-test1.hpda.vn/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=Arcee-Vylinh
```

### 3. Run Demo

```bash
python demo.py
```

## Demo Features

The demo script demonstrates three use cases:

1. **Text Summarization**: Generate abstractive summary from text
2. **Mindmap Generation (JSON)**: Create mindmap structure in JSON format
3. **Mindmap Generation (Markdown)**: Create mindmap in Markdown format (Vietnamese)

## Configuration

All configuration is done through the `.env` file:

- `OPENAI_BASE_URL`: Base URL for OpenAI-compatible API
- `OPENAI_API_KEY`: API key for authentication
- `OPENAI_MODEL`: Model name to use (e.g., "Arcee-Vylinh")
- `SUMMARY_DEFAULT_TYPE`: Default summary type (abstractive/extractive)
- `TOOLS_LANGUAGE`: Default language (vietnamese/english)
- `SUMMARY_MAX_TOKENS`: Maximum tokens for summary
- `MINDMAP_DEFAULT_NODES`: Default maximum nodes for mindmap
- `MINDMAP_DEFAULT_DEPTH`: Default maximum depth for mindmap

## Example Output

The demo will show:
- Input text length
- Generated summary/mindmap
- Metadata (processing time, language, etc.)

## Troubleshooting

### Error: .env file not found
- Make sure you've created a `.env` file in the project root
- Copy from `.env.example` if needed

### Error: OPENAI_API_KEY not found
- Check that `.env` file contains `OPENAI_API_KEY`
- Verify the file is in the correct location

### Import Error: No module named 'dotenv'
- Install python-dotenv: `pip install python-dotenv`

### API Connection Errors
- Verify the `OPENAI_BASE_URL` is correct
- Check that the API key is valid
- Ensure network connectivity to the API endpoint
