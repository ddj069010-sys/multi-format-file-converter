import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import conversion libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Universal File Compression Suite",
    description="Compress and convert files with ease",
    version="1.0.0"
)

# Setup directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DOWNLOAD_DIR = BASE_DIR / "downloads"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, DOWNLOAD_DIR, TEMP_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)
    logger.info(f"✅ Directory ready: {directory.name}")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS enabled for all origins")

# ============================================================================
# SUPPORTED FILE TYPES AND FORMATS
# ============================================================================

SUPPORTED_FORMATS = {
    "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico"],
    "documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".html", ".htm", ".xml"],
    "spreadsheets": [".xls", ".xlsx", ".csv", ".ods", ".json", ".tsv"],
    "presentations": [".ppt", ".pptx", ".odp"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"],
    "video": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"],
    "archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
    "code": [".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".php", ".rb", ".go", ".rs"]
}

def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    for file_type, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return file_type
    return "unknown"

# ============================================================================
# FILE CONVERSION FUNCTIONS
# ============================================================================

def convert_image(input_path: Path, output_path: Path, format_type: str, quality: int) -> bool:
    """Convert image between different formats."""
    if not PIL_AVAILABLE:
        logger.warning("⚠️ PIL not available, copying original file")
        shutil.copy2(input_path, output_path)
        return True
    
    try:
        img = Image.open(input_path)
        
        # Convert RGBA to RGB if needed for JPG
        if format_type.lower() in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        # Save with appropriate quality settings
        if format_type.lower() in ['jpg', 'jpeg']:
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        elif format_type.lower() == 'png':
            img.save(output_path, 'PNG', optimize=True)
        elif format_type.lower() == 'webp':
            img.save(output_path, 'WEBP', quality=quality)
        elif format_type.lower() == 'bmp':
            img.save(output_path, 'BMP')
        elif format_type.lower() == 'gif':
            img.save(output_path, 'GIF')
        elif format_type.lower() in ['tiff', 'tif']:
            img.save(output_path, 'TIFF')
        else:
            img.save(output_path)
        
        logger.info(f"🖼️ Image converted to {format_type.upper()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image conversion failed: {str(e)}")
        return False

def convert_document(input_path: Path, output_path: Path, format_type: str) -> bool:
    """Convert document formats."""
    try:
        format_lower = format_type.lower()
        
        if format_lower == 'pdf':
            if input_path.suffix.lower() == '.pdf':
                shutil.copy2(input_path, output_path)
            else:
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                with open(output_path, 'w') as f:
                    f.write(content)
            logger.info(f"📄 Document converted to PDF")
            return True
            
        elif format_lower in ['txt', 'text']:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"📝 Document converted to TXT")
            return True
            
        else:
            shutil.copy2(input_path, output_path)
            logger.info(f"📋 Document format: {format_type}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Document conversion failed: {str(e)}")
        return False

def convert_file(input_path: Path, output_path: Path, file_type: str, format_type: str, quality: int) -> bool:
    """Route file to appropriate conversion function."""
    if not format_type:
        shutil.copy2(input_path, output_path)
        return True
    
    format_lower = format_type.lower()
    
    if file_type == "images" or format_lower in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'tif']:
        return convert_image(input_path, output_path, format_type, quality)
    
    elif file_type == "documents" or format_lower in ['pdf', 'txt', 'text', 'docx', 'html']:
        return convert_document(input_path, output_path, format_type)
    
    else:
        shutil.copy2(input_path, output_path)
        return True

# ============================================================================
# API ENDPOINTS - MUST BE BEFORE STATIC FILE MOUNT
# ============================================================================

#@app.get("/")
#async def root():
#    """Root endpoint info."""
#    return {
#        "message": "Universal File Compression Suite",
#        "version": "1.0.0",
#        "docs": "/docs"
#    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    uploads_count = len(list(UPLOAD_DIR.glob("*")))
    downloads_count = len(list(DOWNLOAD_DIR.glob("*")))
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Running",
        "uploads_available": uploads_count,
        "downloads_available": downloads_count
    }

@app.post("/api/compress")
async def compress_file(
    file: UploadFile = File(...),
    quality: int = Form(80),
    format_type: str = Form("")
):
    """Main compression endpoint. Accepts files and compression parameters."""
    try:
        filename = file.filename if file.filename else "uploaded_file"
        logger.info(f"📥 Received file: {filename}")
        logger.info(f"   Content Type: {file.content_type}")
        logger.info(f"   Quality: {quality}%")
        logger.info(f"   Format: {format_type or 'auto'}")
        
        file_type = get_file_type(filename)
        logger.info(f"   File Type: {file_type}")
        
        upload_path = UPLOAD_DIR / filename
        with open(upload_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        original_size = upload_path.stat().st_size
        logger.info(f"📊 Original size: {original_size / 1024:.2f} KB")
        
        file_stem = Path(filename).stem
        file_ext = f".{format_type}" if format_type else Path(filename).suffix or ".bin"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compressed_{file_stem}_{quality}_{timestamp}{file_ext}"
        download_path = DOWNLOAD_DIR / output_filename
        
        # Perform conversion
        conversion_success = convert_file(upload_path, download_path, file_type, format_type, quality)
        
        if not conversion_success:
            logger.warning("⚠️ Conversion failed, copying original file")
            shutil.copy2(upload_path, download_path)
        
        # Get compressed file size
        compressed_size = download_path.stat().st_size
        compression_ratio = max(0, ((original_size - compressed_size) / original_size * 100)) if original_size > 0 else 0
        
        logger.info(f"📦 Compressed size: {compressed_size / 1024:.2f} KB")
        logger.info(f"💾 Compression ratio: {compression_ratio:.1f}%")
        logger.info(f"✅ File saved: {output_filename}")
        
        # Clean up upload
        upload_path.unlink()
        
        return {
            "success": True,
            "filename": output_filename,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "download_url": f"/downloads/{output_filename}",
            "quality": quality,
            "format": format_type or "original",
            "file_type": file_type,
            "converted": format_type != ""
        }
    
    except Exception as e:
        logger.error(f"❌ Error compressing file: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/downloads/{filename}")
async def download_file(filename: str):
    """Download endpoint to retrieve compressed files."""
    try:
        file_path = DOWNLOAD_DIR / filename
        
        if not file_path.exists():
            logger.warning(f"⚠️ File not found: {filename}")
            return {"error": "File not found"}
        
        logger.info(f"⬇️ Downloading: {filename}")
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            filename=filename
        )
    
    except Exception as e:
        logger.error(f"❌ Error downloading file: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.get("/api/history")
async def get_history():
    """Get list of all compressed files available for download."""
    try:
        files = []
        for file_path in DOWNLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_type = get_file_type(file_path.name)
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2),
                    "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                    "download_url": f"/downloads/{file_path.name}",
                    "file_type": file_type
                })
        
        logger.info(f"📋 History: {len(files)} files available")
        return {
            "success": True,
            "files": sorted(files, key=lambda x: x['created'], reverse=True)
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting history: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.get("/api/formats")
async def get_formats():
    """Get list of all supported file formats."""
    try:
        logger.info("📋 Fetching supported formats")
        return {
            "success": True,
            "formats": SUPPORTED_FORMATS
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting formats: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.get("/api/stats")
async def get_stats():
    """Get server statistics and directory information."""
    try:
        def get_dir_size(path):
            total = 0
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
            return total
        
        uploads_size = get_dir_size(UPLOAD_DIR)
        downloads_size = get_dir_size(DOWNLOAD_DIR)
        temp_size = get_dir_size(TEMP_DIR)
        
        uploads_count = len(list(UPLOAD_DIR.glob("*")))
        downloads_count = len(list(DOWNLOAD_DIR.glob("*")))
        
        logger.info("📊 Fetching server statistics")
        
        return {
            "success": True,
            "upload_dir": {
                "count": uploads_count,
                "size": uploads_size,
                "size_mb": round(uploads_size / 1024 / 1024, 2)
            },
            "download_dir": {
                "count": downloads_count,
                "size": downloads_size,
                "size_mb": round(downloads_size / 1024 / 1024, 2)
            },
            "temp_dir": {
                "size": temp_size,
                "size_mb": round(temp_size / 1024 / 1024, 2)
            },
            "total_storage_mb": round((uploads_size + downloads_size + temp_size) / 1024 / 1024, 2)
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting stats: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.post("/api/cleanup")
async def cleanup_files(days: int = 7):
    """Clean up old files older than specified days."""
    try:
        import time
        current_time = time.time()
        deleted_count = 0
        freed_space = 0
        
        for file_path in DOWNLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age_days = (current_time - file_path.stat().st_ctime) / (24 * 3600)
                if file_age_days > days:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    freed_space += file_size
                    logger.info(f"🗑️ Deleted: {file_path.name}")
        
        logger.info(f"🧹 Cleanup complete: {deleted_count} files removed, {freed_space / 1024 / 1024:.2f} MB freed")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "freed_space_mb": round(freed_space / 1024 / 1024, 2),
            "message": f"Deleted {deleted_count} files older than {days} days"
        }
    
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific compressed file."""
    try:
        file_path = DOWNLOAD_DIR / filename
        
        if not file_path.exists():
            logger.warning(f"⚠️ File not found for deletion: {filename}")
            return {"success": False, "error": "File not found"}
        
        file_size = file_path.stat().st_size
        file_path.unlink()
        
        logger.info(f"🗑️ Deleted: {filename} ({file_size / 1024 / 1024:.2f} MB)")
        
        return {
            "success": True,
            "message": f"File {filename} deleted successfully",
            "freed_space_mb": round(file_size / 1024 / 1024, 2)
        }
    
    except Exception as e:
        logger.error(f"❌ Error deleting file: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.get("/api/files/{filename}")
async def get_file_info(filename: str):
    """Get information about a specific file."""
    try:
        file_path = DOWNLOAD_DIR / filename
        
        if not file_path.exists():
            return {"success": False, "error": "File not found"}
        
        stat = file_path.stat()
        file_type = get_file_type(filename)
        
        return {
            "success": True,
            "filename": filename,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type": file_type,
            "download_url": f"/downloads/{filename}"
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting file info: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.get("/api/status")
async def get_status():
    """Get current service status and capabilities."""
    try:
        def count_files(path):
            return len(list(path.glob("*")))
        
        return {
            "success": True,
            "service": "Running",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "directories": {
                "uploads": count_files(UPLOAD_DIR),
                "downloads": count_files(DOWNLOAD_DIR),
                "temp": count_files(TEMP_DIR)
            },
            "capabilities": {
                "max_file_size_mb": 5000,
                "supported_formats": len([f for formats in SUPPORTED_FORMATS.values() for f in formats]),
                "concurrent_uploads": "unlimited",
                "pil_available": PIL_AVAILABLE
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting status: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================================================================
# STATIC FILES - MUST BE AFTER ALL API ROUTES
# ============================================================================

app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

# ============================================================================
# STARTUP AND SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event handler."""
    logger.info("=" * 70)
    logger.info("🚀 UNIVERSAL FILE COMPRESSION SUITE - STARTUP")
    logger.info("=" * 70)
    logger.info(f"📁 Base directory: {BASE_DIR}")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
    logger.info(f"📁 Download directory: {DOWNLOAD_DIR}")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")
    logger.info(f"📁 Log directory: {LOG_DIR}")
    logger.info("=" * 70)
    
    for directory in [UPLOAD_DIR, DOWNLOAD_DIR, TEMP_DIR, LOG_DIR]:
        if directory.exists() and os.access(directory, os.W_OK):
            logger.info(f"✅ {directory.name:15} - OK (writable)")
        else:
            logger.error(f"❌ {directory.name:15} - ERROR (not writable)")
    
    logger.info("=" * 70)
    logger.info("📋 SUPPORTED FILE FORMATS:")
    for file_type, extensions in SUPPORTED_FORMATS.items():
        logger.info(f"   {file_type.upper():15} - {len(extensions)} formats: {', '.join(extensions[:5])}...")
    
    logger.info("=" * 70)
    logger.info(f"📦 PIL (Image Library): {'✅ Available' if PIL_AVAILABLE else '❌ Not installed'}")
    logger.info("=" * 70)
    logger.info("✅ Application started successfully!")
    logger.info(f"🌐 Web UI: http://127.0.0.1:8000")
    logger.info(f"📚 API Docs: http://127.0.0.1:8000/docs")
    logger.info(f"🔧 ReDoc: http://127.0.0.1:8000/redoc")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler."""
    logger.info("=" * 70)
    logger.info("🛑 SHUTTING DOWN UNIVERSAL FILE COMPRESSION SUITE")
    logger.info("=" * 70)
    
    try:
        for file_path in TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        logger.info("🧹 Temporary files cleaned up")
    except Exception as e:
        logger.error(f"⚠️ Error cleaning temp: {str(e)}")
    
    try:
        uploads_count = len(list(UPLOAD_DIR.glob("*")))
        downloads_count = len(list(DOWNLOAD_DIR.glob("*")))
        logger.info(f"📊 Final Stats - Uploads: {uploads_count}, Downloads: {downloads_count}")
    except Exception as e:
        logger.error(f"⚠️ Error getting final stats: {str(e)}")
    
    logger.info("✅ Shutdown complete")
    logger.info("=" * 70)

# ============================================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for all unhandled errors."""
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "success": False,
        "error": "An internal server error occurred",
        "detail": str(exc)
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Main entry point for the application."""
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
