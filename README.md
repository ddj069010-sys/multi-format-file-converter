# Universal File Converter

A simple FastAPI-based tool to convert files between multiple formats from one place.  
It supports images, documents, archives and audio/video files.

## Features

- Image conversion (JPG, PNG, WebP, BMP, TIFF, GIF, ICO).  
- Document conversion using Pandoc (PDF, DOCX, ODT, TXT, HTML, RTF).  
- Archive conversion (ZIP, TAR, 7Z, RAR).
- Audio and video conversion using FFmpeg (MP3, WAV, AAC, OGG, MP4, MKV, WEBM, etc.). 

## Tech stack

- Python  
- FastAPI  
- Uvicorn  
- Pillow, python-docx, py7zr, FFmpeg, Pandoc.
- <br>
- And also you can run it locally by cloning the code and get universal format chanhing app running locally.
## How to run

# 1. Clone this repository
git clone https://github.com/ddj069010-sys/multi-format-file-converter.git
cd multi-format-file-converter

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the API server
uvicorn main:app --reload
