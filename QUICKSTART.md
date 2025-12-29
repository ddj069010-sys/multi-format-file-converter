# Quick Start Guide

## 🚀 Run the Application

### Option 1: Simple Run Script (Recommended)
```bash
./run.sh
```

### Option 2: Manual Start
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Option 3: Background Service
```bash
./start-server.sh   # Start in background
./stop-server.sh    # Stop the server
```

## 🌐 Access the Application

Once running, open your browser:
- **Web Interface**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📋 Features

✅ Image conversion (JPG, PNG, WebP, BMP, TIFF, GIF)  
✅ Document conversion (PDF, TXT, DOCX)  
✅ Archive conversion (ZIP, TAR, 7Z)  
✅ Audio/Video conversion (requires FFmpeg)  
✅ Download history  
✅ File management  
✅ Security features (rate limiting, file validation)  

## 🔧 Configuration

Set environment variables or edit `main.py` Config class:
- `MAX_FILE_SIZE_MB` - Maximum upload size (default: 500MB)
- `CORS_ORIGINS` - Allowed origins
- `RATE_LIMIT_PER_MINUTE` - Rate limit (default: 60)
- `HOST` - Server host (default: 127.0.0.1)
- `PORT` - Server port (default: 8000)

## 📝 Notes

- Logs are stored in `logs/app.log`
- Files are stored in `uploads/` (temp) and `downloads/`
- FFmpeg required for audio/video conversion
- Pandoc optional for advanced document conversion

## ❓ Troubleshooting

**Port already in use:**
```bash
lsof -i :8000
kill -9 <PID>
```

**Permission errors:**
```bash
chmod -R 755 uploads downloads temp logs
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

