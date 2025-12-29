# Universal File Compression Suite

A FastAPI-based web application for compressing and converting files between multiple formats. Supports images, documents, archives, audio, and video files.

## Features

### Currently Implemented
- **Image Format Conversion**: JPG, PNG, WebP, BMP, TIFF, GIF, ICO with format recommendations
- **Image Resizing**: Resize images to specific dimensions with aspect ratio preservation
- **Image Compression**: Multiple preset quality levels (low, medium, high, ultra)
- **PDF Operations**: Merge multiple PDF files into one
- **Document Conversion**: Basic PDF and TXT conversion (pandoc and python-docx optional)
- **Archive Conversion**: ZIP, TAR, 7Z format conversion (requires py7zr for 7Z)
- **Audio/Video Conversion**: MP3, WAV, FLAC, AAC, OGG, MP4, AVI, MKV, WebM, MOV (requires FFmpeg)
- **Security**: File size limits, filename sanitization, rate limiting, CORS restrictions
- **Web Interface**: Modern, responsive UI with dark mode support and format recommendations
- **Download History**: View and download previously processed files

### Image Format Recommendations
- **JPEG**: Best for photos and images with many colors. Small file sizes, lossy compression.
- **PNG**: Best for graphics, logos, and images requiring transparency. Lossless format.
- **WebP**: Modern format with 25-35% better compression than JPEG. Best for web use.
- **GIF**: Good for simple graphics and animations. Limited to 256 colors.
- **TIFF**: High-quality format for professional photography and printing.
- **BMP**: Uncompressed format, very large file sizes. Rarely used.
- **ICO**: Windows icon format for applications and favicons.

### Requirements

#### Core Dependencies
- Python 3.10+
- FastAPI
- Uvicorn
- Pillow (PIL) - for image processing

#### Optional Dependencies
- **python-docx** - for DOCX document conversion
- **py7zr** - for 7Z archive support
- **FFmpeg** (system package) - for audio/video conversion
- **pandoc** (system package) - for advanced document conversion

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd DocFormatting
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install optional system dependencies (if needed)

**FFmpeg** (for audio/video conversion):
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Pandoc** (for document conversion):
```bash
# Ubuntu/Debian
sudo apt-get install pandoc

# macOS
brew install pandoc

# Windows
# Download from https://pandoc.org/installing.html
```

## Configuration

The application can be configured using environment variables:

- `UPLOAD_DIR` - Upload directory (default: `uploads`)
- `DOWNLOAD_DIR` - Download directory (default: `downloads`)
- `TEMP_DIR` - Temporary files directory (default: `temp`)
- `LOG_DIR` - Log files directory (default: `logs`)
- `MAX_FILE_SIZE_MB` - Maximum file size in MB (default: `500`)
- `CORS_ORIGINS` - Comma-separated list of allowed CORS origins (default: `http://localhost:8000,http://127.0.0.1:8000`)
- `HOST` - Server host (default: `127.0.0.1`)
- `PORT` - Server port (default: `8000`)
- `RATE_LIMIT_PER_MINUTE` - Rate limit per IP address (default: `60`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `LOG_FILE_MAX_BYTES` - Maximum log file size in bytes (default: `10485760` = 10MB)
- `LOG_FILE_BACKUP_COUNT` - Number of log file backups to keep (default: `5`)

### Example configuration

Create a `.env` file (not included in repo):

```bash
MAX_FILE_SIZE_MB=1000
CORS_ORIGINS=http://localhost:8000,http://example.com
HOST=0.0.0.0
PORT=8080
LOG_LEVEL=DEBUG
```

## Running the Application

### Option 1: Using the provided script

```bash
# Start server
./start-server.sh

# Stop server
./stop-server.sh
```

### Option 2: Direct execution

```bash
python main.py
```

### Option 3: Using uvicorn directly

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Usage

1. Open your browser and navigate to `http://127.0.0.1:8000`
2. Upload files by dragging and dropping or clicking to browse
3. Select compression quality (30-100)
4. Choose output format (optional, defaults to original format)
5. Click "Compress Files" to process
6. Download processed files from the results or history tab

## API Endpoints

### Core Endpoints
- `GET /` - Web interface
- `GET /api/health` - Health check
- `GET /api/formats` - Get supported formats and capabilities
- `GET /api/history` - Get download history
- `GET /api/stats` - Get server statistics
- `GET /api/status` - Get service status and capabilities
- `GET /api/files/{filename}` - Get file information
- `DELETE /api/files/{filename}` - Delete a file
- `POST /api/cleanup` - Clean up old files
- `GET /downloads/{filename}` - Download a processed file

### Conversion Endpoints
- `POST /api/compress` - Compress/convert a file (generic)
- `POST /api/convert-image-format` - Convert image to different format
- `POST /api/resize-image` - Resize image to specific dimensions
- `POST /api/compress-image-preset` - Compress image with preset quality
- `POST /api/merge-pdfs` - Merge multiple PDF files

### Documentation
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Security Features

- **File Size Limits**: Configurable maximum file size to prevent DoS attacks
- **Filename Sanitization**: Prevents path traversal attacks
- **File Type Validation**: Blocks executable files and validates file types
- **Rate Limiting**: Prevents abuse with configurable rate limits
- **CORS Restrictions**: Configurable allowed origins
- **Input Validation**: Validates all input parameters

## Limitations

- Document conversion is basic without pandoc/python-docx installed
- Video/audio conversion requires FFmpeg to be installed and in PATH
- Archive conversion for 7Z format requires py7zr
- Large files are processed in memory chunks, but very large files may still use significant memory
- Some format conversions may not be available depending on installed dependencies

## Project Structure

```
DocFormatting/
├── main.py              # Main FastAPI application
├── file_utils.py        # Utility functions for file operations
├── converters.py        # Converter classes (legacy, not used in main.py)
├── index.html          # Web interface
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── start-server.sh    # Server startup script
├── stop-server.sh     # Server shutdown script
├── uploads/           # Upload directory (created automatically)
├── downloads/         # Download directory (created automatically)
├── temp/              # Temporary files (created automatically)
└── logs/              # Log files (created automatically)
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]