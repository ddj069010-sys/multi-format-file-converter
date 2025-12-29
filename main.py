import os
import shutil
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Import utility functions
from file_utils import (
    get_file_type as get_file_type_util,
    safe_filename,
    format_file_size,
    get_mime_type,
    get_compression_ratio
)

# Import conversion libraries with error handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False

try:
    import zipfile
    ZIPFILE_AVAILABLE = True
except ImportError:
    ZIPFILE_AVAILABLE = False

try:
    import tarfile
    TARFILE_AVAILABLE = True
except ImportError:
    TARFILE_AVAILABLE = False

import subprocess

try:
    FFMPEG_AVAILABLE = subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode == 0
except:
    FFMPEG_AVAILABLE = False

try:
    subprocess.run(['pandoc', '--version'], capture_output=True, timeout=5)
    PANDOC_AVAILABLE = True
except:
    PANDOC_AVAILABLE = False

# ============================================================================
# CONFIGURATION (Environment Variables)
# ============================================================================

class Config:
    """Application configuration from environment variables."""
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
    DOWNLOAD_DIR = BASE_DIR / os.getenv("DOWNLOAD_DIR", "downloads")
    TEMP_DIR = BASE_DIR / os.getenv("TEMP_DIR", "temp")
    LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")
    
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))
    
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

config = Config()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create log directory
config.LOG_DIR.mkdir(exist_ok=True)

# Setup logging with file handler and rotation
log_file = config.LOG_DIR / "app.log"
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=config.LOG_FILE_MAX_BYTES,
    backupCount=config.LOG_FILE_BACKUP_COUNT
)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================

for directory in [config.UPLOAD_DIR, config.DOWNLOAD_DIR, config.TEMP_DIR, config.LOG_DIR]:
    directory.mkdir(exist_ok=True)
    logger.info(f"✅ Directory ready: {directory.name}")

# ============================================================================
# RATE LIMITING (Simple in-memory implementation)
# ============================================================================

from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests: int = 60, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < self.window]
        
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True

rate_limiter = RateLimiter(max_requests=config.RATE_LIMIT_PER_MINUTE, window=60)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("=" * 70)
    logger.info("🚀 UNIVERSAL FILE COMPRESSION SUITE - STARTUP")
    logger.info("=" * 70)
    logger.info(f"📁 Base directory: {config.BASE_DIR}")
    logger.info(f"📁 Upload directory: {config.UPLOAD_DIR}")
    logger.info(f"📁 Download directory: {config.DOWNLOAD_DIR}")
    logger.info(f"📁 Temp directory: {config.TEMP_DIR}")
    logger.info(f"📁 Log directory: {config.LOG_DIR}")
    logger.info("=" * 70)
    
    for directory in [config.UPLOAD_DIR, config.DOWNLOAD_DIR, config.TEMP_DIR, config.LOG_DIR]:
        if directory.exists() and os.access(directory, os.W_OK):
            logger.info(f"✅ {directory.name:15} - OK (writable)")
        else:
            logger.error(f"❌ {directory.name:15} - ERROR (not writable)")
    
    logger.info("=" * 70)
    logger.info("📋 DEPENDENCY STATUS:")
    logger.info(f"   PIL: {'✅ Available' if PIL_AVAILABLE else '❌ Not installed'}")
    logger.info(f"   python-docx: {'✅ Available' if DOCX_AVAILABLE else '❌ Not installed'}")
    logger.info(f"   py7zr: {'✅ Available' if PY7ZR_AVAILABLE else '❌ Not installed'}")
    logger.info(f"   FFmpeg: {'✅ Available' if FFMPEG_AVAILABLE else '❌ Not installed'}")
    logger.info(f"   Pandoc: {'✅ Available' if PANDOC_AVAILABLE else '❌ Not installed'}")
    logger.info("=" * 70)
    logger.info("✅ Application started successfully!")
    logger.info(f"🌐 Web UI: http://{config.HOST}:{config.PORT}")
    logger.info(f"📚 API Docs: http://{config.HOST}:{config.PORT}/docs")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown
    logger.info("=" * 70)
    logger.info("🛑 SHUTTING DOWN UNIVERSAL FILE COMPRESSION SUITE")
    logger.info("=" * 70)
    
    try:
        for file_path in config.TEMP_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        logger.info("🧹 Temporary files cleaned up")
    except Exception as e:
        logger.error(f"⚠️ Error cleaning temp: {str(e)}")
    
    logger.info("✅ Shutdown complete")
    logger.info("=" * 70)

app = FastAPI(
    title="Universal File Compression Suite",
    description="Compress and convert files with ease",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
logger.info(f"✅ CORS enabled for origins: {config.CORS_ORIGINS}")

# ============================================================================
# MIDDLEWARE FOR RATE LIMITING AND SECURITY
# ============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"
    
    if request.url.path.startswith("/api/"):
        if not rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"success": False, "error": "Rate limit exceeded. Please try again later."}
            )
    
    response = await call_next(request)
    return response

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

# Blocked file types for security
BLOCKED_EXTENSIONS = {".exe", ".msi", ".bat", ".cmd", ".dll", ".com", ".sh", ".bin", ".scr", ".vbs", ".js", ".jar"}

def get_file_type(filename: str) -> str:
    """Determine file type from extension using file_utils."""
    return get_file_type_util(Path(filename).suffix)

def validate_file_type(filename: str) -> bool:
    """Validate that file type is allowed."""
    ext = Path(filename).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        return False
    # Check if extension is in any supported format
    for extensions in SUPPORTED_FORMATS.values():
        if ext in extensions:
            return True
    return False

# ============================================================================
# PYDANTIC MODELS FOR VALIDATION
# ============================================================================

class CompressRequest(BaseModel):
    quality: int = Field(default=80, ge=30, le=100, description="Compression quality (30-100)")
    format_type: str = Field(default="", max_length=20, description="Target format")
    
    @field_validator('quality')
    @classmethod
    def validate_quality(cls, v):
        if not isinstance(v, int):
            raise ValueError('Quality must be an integer')
        if v < 30 or v > 100:
            raise ValueError('Quality must be between 30 and 100')
        return v

# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

async def convert_image(input_path: Path, output_path: Path, format_type: str, quality: int) -> bool:
    """Convert image between different formats."""
    if not PIL_AVAILABLE:
        logger.warning("⚠️ PIL not available, copying original file")
        shutil.copy2(input_path, output_path)
        return True
    
    try:
        img = Image.open(input_path)
        original_format = img.format
        
        # Convert RGBA to RGB if needed for formats that don't support transparency
        if format_type.lower() in ['jpg', 'jpeg', 'bmp'] and img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                rgb_img.paste(img, mask=img.split()[-1])
            elif img.mode == 'P':
                # Handle palette mode with transparency
                if 'transparency' in img.info:
                    rgb_img.paste(img.convert('RGBA'), mask=img.convert('RGBA').split()[-1])
                else:
                    rgb_img.paste(img.convert('RGB'))
            else:
                rgb_img.paste(img)
            img = rgb_img
        
        # Save with appropriate quality settings
        save_kwargs = {}
        
        if format_type.lower() in ['jpg', 'jpeg']:
            img.save(output_path, 'JPEG', quality=quality, optimize=True, progressive=True)
        elif format_type.lower() == 'png':
            # PNG compression level (0-9, 9 is best compression)
            compress_level = 9 - int((quality / 100) * 8)  # Inverse: higher quality = less compression
            img.save(output_path, 'PNG', optimize=True, compress_level=compress_level)
        elif format_type.lower() == 'webp':
            img.save(output_path, 'WEBP', quality=quality, method=6)  # Best compression method
        elif format_type.lower() == 'bmp':
            img.save(output_path, 'BMP')
        elif format_type.lower() == 'gif':
            # GIF doesn't support quality, but we can optimize
            img.save(output_path, 'GIF', optimize=True, save_all=True if hasattr(img, 'is_animated') and img.is_animated else False)
        elif format_type.lower() in ['tiff', 'tif']:
            # TIFF compression
            img.save(output_path, 'TIFF', compression='lzw')
        elif format_type.lower() == 'ico':
            # ICO format - resize to common icon sizes if too large
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            if img.size[0] > 256 or img.size[1] > 256:
                # Use largest size that fits
                max_size = max(s for s in sizes if s[0] <= img.size[0] and s[1] <= img.size[1])
                img = img.resize(max_size, Image.Resampling.LANCZOS)
            img.save(output_path, 'ICO', sizes=[img.size])
        else:
            img.save(output_path)
        
        logger.info(f"🖼️ Image converted from {original_format or 'unknown'} to {format_type.upper()}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image conversion failed: {str(e)}")
        return False

async def resize_image(input_path: Path, output_path: Path, width: int, height: int, maintain_aspect: bool = True) -> bool:
    """Resize image to specific dimensions."""
    if not PIL_AVAILABLE:
        logger.warning("⚠️ PIL not available")
        return False
    
    try:
        img = Image.open(input_path)
        original_width, original_height = img.size
        
        if maintain_aspect:
            # Calculate aspect ratio
            aspect_ratio = original_width / original_height
            if width and height:
                # Use the smaller dimension to maintain aspect
                if width / aspect_ratio <= height:
                    new_height = int(width / aspect_ratio)
                    new_width = width
                else:
                    new_width = int(height * aspect_ratio)
                    new_height = height
            elif width:
                new_width = width
                new_height = int(width / aspect_ratio)
            elif height:
                new_height = height
                new_width = int(height * aspect_ratio)
            else:
                return False
        else:
            new_width = width or original_width
            new_height = height or original_height
        
        # Resize using high-quality resampling
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, quality=95, optimize=True)
        
        logger.info(f"🖼️ Image resized from {original_width}x{original_height} to {new_width}x{new_height}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image resize failed: {str(e)}")
        return False

async def compress_image_preset(input_path: Path, output_path: Path, preset: str) -> bool:
    """Compress image with preset quality levels."""
    if not PIL_AVAILABLE:
        logger.warning("⚠️ PIL not available")
        return False
    
    try:
        img = Image.open(input_path)
        
        # Preset quality levels
        presets = {
            'low': {'quality': 40, 'optimize': True},
            'medium': {'quality': 70, 'optimize': True},
            'high': {'quality': 85, 'optimize': True},
            'ultra': {'quality': 95, 'optimize': True}
        }
        
        preset_lower = preset.lower()
        if preset_lower not in presets:
            preset_lower = 'medium'
        
        settings = presets[preset_lower]
        
        # Convert to RGB if needed for JPEG
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = rgb_img
        
        # Save as JPEG with preset quality
        img.save(output_path, 'JPEG', **settings)
        
        logger.info(f"🖼️ Image compressed with {preset_lower} preset")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image compression failed: {str(e)}")
        return False

async def merge_pdfs(pdf_files: list[Path], output_path: Path) -> bool:
    """Merge multiple PDF files into one."""
    if not PYPDF_AVAILABLE:
        logger.warning("⚠️ pypdf not available")
        return False
    
    try:
        writer = PdfWriter()
        
        for pdf_file in pdf_files:
            if not pdf_file.exists():
                logger.warning(f"PDF file not found: {pdf_file}")
                continue
            
            reader = PdfReader(str(pdf_file))
            for page in reader.pages:
                writer.add_page(page)
        
        if len(writer.pages) == 0:
            logger.error("No pages to merge")
            return False
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        logger.info(f"📄 Merged {len(pdf_files)} PDF files into one")
        return True
        
    except Exception as e:
        logger.error(f"❌ PDF merge failed: {str(e)}")
        return False

async def convert_document(input_path: Path, output_path: Path, format_type: str) -> bool:
    """Convert document formats using pandoc or fallback methods."""
    try:
        format_lower = format_type.lower()
        
        if format_lower == 'pdf':
            if input_path.suffix.lower() == '.pdf':
                shutil.copy2(input_path, output_path)
                return True
            elif PANDOC_AVAILABLE:
                try:
                    result = subprocess.run(
                        ['pandoc', str(input_path), '-o', str(output_path)],
                        capture_output=True,
                        timeout=30,
                        text=True
                    )
                    if result.returncode == 0:
                        logger.info("📄 Document converted to PDF using pandoc")
                        return True
                except Exception as e:
                    logger.warning(f"Pandoc conversion failed: {e}")
            
            # Fallback: text extraction
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # Simple text to "PDF-like" output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"📄 Document converted to PDF (fallback)")
            return True
            
        elif format_lower in ['txt', 'text']:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"📝 Document converted to TXT")
            return True
        
        elif format_lower == 'docx' and DOCX_AVAILABLE and input_path.suffix.lower() in ['.txt', '.rtf']:
            try:
                doc = docx.Document()
                with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                    doc.add_paragraph(f.read())
                doc.save(output_path)
                logger.info("📝 Document converted to DOCX")
                return True
            except Exception as e:
                logger.warning(f"DOCX conversion failed: {e}")
        
        # Default: copy file
        shutil.copy2(input_path, output_path)
        logger.info(f"📋 Document format: {format_type}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Document conversion failed: {str(e)}")
        return False

async def convert_archive(input_path: Path, output_path: Path, format_type: str) -> bool:
    """Convert archive formats."""
    if not ZIPFILE_AVAILABLE and not TARFILE_AVAILABLE and not PY7ZR_AVAILABLE:
        logger.warning("⚠️ No archive libraries available, copying original file")
        shutil.copy2(input_path, output_path)
        return True
    
    try:
        format_lower = format_type.lower()
        
        # Extract to temp directory
        extract_dir = config.TEMP_DIR / f"archive_{int(datetime.now().timestamp())}"
        extract_dir.mkdir(exist_ok=True)
        
        try:
            # Extract source archive
            if input_path.suffix.lower() == '.zip' and ZIPFILE_AVAILABLE:
                with zipfile.ZipFile(input_path, 'r') as zf:
                    zf.extractall(extract_dir)
            elif input_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz'] and TARFILE_AVAILABLE:
                with tarfile.open(input_path, 'r:*') as tf:
                    tf.extractall(extract_dir)
            elif input_path.suffix.lower() == '.7z' and PY7ZR_AVAILABLE:
                with py7zr.SevenZipFile(input_path, 'r') as szf:
                    szf.extractall(extract_dir)
            else:
                shutil.copy2(input_path, output_path)
                return True
            
            # Create target archive
            if format_lower == 'zip' and ZIPFILE_AVAILABLE:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file in extract_dir.rglob('*'):
                        if file.is_file():
                            zf.write(file, arcname=file.relative_to(extract_dir))
                logger.info("📦 Archive converted to ZIP")
                return True
            
            elif format_lower == 'tar' and TARFILE_AVAILABLE:
                with tarfile.open(output_path, 'w') as tf:
                    for file in extract_dir.rglob('*'):
                        if file.is_file():
                            tf.add(file, arcname=file.relative_to(extract_dir))
                logger.info("📦 Archive converted to TAR")
                return True
            
            else:
                shutil.copy2(input_path, output_path)
                return True
                
        finally:
            # Cleanup extract directory
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
                
    except Exception as e:
        logger.error(f"❌ Archive conversion failed: {str(e)}")
        return False

async def convert_media(input_path: Path, output_path: Path, format_type: str, quality: int) -> bool:
    """Convert audio/video formats using FFmpeg."""
    if not FFMPEG_AVAILABLE:
        logger.warning("⚠️ FFmpeg not available, copying original file")
        shutil.copy2(input_path, output_path)
        return True
    
    try:
        # Determine bitrate from quality
        if format_type.lower() in ['mp3', 'aac', 'ogg']:
            bitrate = f"{32 + int((quality / 100) * 256)}k"  # 32-288 kbps
            cmd = ['ffmpeg', '-i', str(input_path), '-b:a', bitrate, '-y', str(output_path)]
        elif format_type.lower() in ['mp4', 'webm', 'avi']:
            bitrate = f"{500 + int((quality / 100) * 4000)}k"  # 500-4500 kbps
            cmd = ['ffmpeg', '-i', str(input_path), '-b:v', bitrate, '-y', str(output_path)]
        else:
            cmd = ['ffmpeg', '-i', str(input_path), '-y', str(output_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        logger.info(f"🎵 Media converted to {format_type.upper()}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Media conversion timeout")
        return False
    except Exception as e:
        logger.error(f"❌ Media conversion failed: {str(e)}")
        return False

async def convert_file(input_path: Path, output_path: Path, file_type: str, format_type: str, quality: int) -> bool:
    """Route file to appropriate conversion function."""
    if not format_type:
        shutil.copy2(input_path, output_path)
        return True
    
    format_lower = format_type.lower()
    
    if file_type == "images" or format_lower in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'tif']:
        return await convert_image(input_path, output_path, format_type, quality)
    
    elif file_type == "documents" or format_lower in ['pdf', 'txt', 'text', 'docx', 'html']:
        return await convert_document(input_path, output_path, format_type)
    
    elif file_type == "archives" or format_lower in ['zip', 'tar', '7z', 'rar']:
        return await convert_archive(input_path, output_path, format_type)
    
    elif file_type in ["audio", "video"] or format_lower in ['mp3', 'wav', 'flac', 'aac', 'ogg', 'mp4', 'avi', 'mkv', 'webm', 'mov']:
        return await convert_media(input_path, output_path, format_type, quality)
    
    else:
        shutil.copy2(input_path, output_path)
        return True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dir_size(path: Path) -> int:
    """Calculate total size of directory."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        logger.error(f"Error calculating directory size: {e}")
    return total

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": "Validation error", "details": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled errors."""
    logger.error(f"❌ Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal server error occurred",
            "detail": str(exc) if logger.level == logging.DEBUG else "Internal error"
        }
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    try:
        uploads_count = len(list(config.UPLOAD_DIR.glob("*")))
        downloads_count = len(list(config.DOWNLOAD_DIR.glob("*")))
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "Running",
            "uploads_available": uploads_count,
            "downloads_available": downloads_count
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/api/resize-image")
async def resize_image_endpoint(
    request: Request,
    file: UploadFile = File(...),
    width: int = Form(None),
    height: int = Form(None),
    maintain_aspect: bool = Form(True)
):
    """
    Resize image to specific dimensions.
    
    - **file**: Image file to resize
    - **width**: Target width in pixels (optional)
    - **height**: Target height in pixels (optional)
    - **maintain_aspect**: Maintain aspect ratio (default: True)
    """
    upload_path = None
    try:
        # Sanitize filename
        original_filename = file.filename if file.filename else "uploaded_file"
        filename = safe_filename(original_filename)
        
        # Validate it's an image
        ext = Path(filename).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']:
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not width and not height:
            raise HTTPException(status_code=400, detail="Either width or height must be specified")
        
        upload_path = config.UPLOAD_DIR / filename
        
        # Stream file to disk
        file_size = 0
        with open(upload_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > config.MAX_FILE_SIZE_BYTES:
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File size exceeds maximum of {config.MAX_FILE_SIZE_MB}MB")
                buffer.write(chunk)
        
        original_size = upload_path.stat().st_size
        
        # Generate output filename
        file_stem = Path(filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"resized_{file_stem}_{width or 'auto'}x{height or 'auto'}_{timestamp}{ext}"
        output_filename = safe_filename(output_filename)
        download_path = config.DOWNLOAD_DIR / output_filename
        
        # Resize image
        success = await resize_image(upload_path, download_path, width, height, maintain_aspect)
        
        if not success:
            raise HTTPException(status_code=500, detail="Image resize failed")
        
        compressed_size = download_path.stat().st_size
        compression_ratio = get_compression_ratio(original_size, compressed_size)
        
        # Clean up upload
        if upload_path.exists():
            upload_path.unlink()
        
        return {
            "success": True,
            "filename": output_filename,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "download_url": f"/downloads/{output_filename}",
            "width": width,
            "height": height
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resizing image: {str(e)}", exc_info=True)
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error resizing image: {str(e)}")

@app.post("/api/compress-image-preset")
async def compress_image_preset_endpoint(
    request: Request,
    file: UploadFile = File(...),
    preset: str = Form("medium")
):
    """
    Compress image with preset quality levels (low, medium, high, ultra).
    
    - **file**: Image file to compress
    - **preset**: Compression preset - 'low', 'medium', 'high', or 'ultra'
    """
    upload_path = None
    try:
        # Sanitize filename
        original_filename = file.filename if file.filename else "uploaded_file"
        filename = safe_filename(original_filename)
        
        # Validate it's an image
        ext = Path(filename).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif']:
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if preset.lower() not in ['low', 'medium', 'high', 'ultra']:
            preset = 'medium'
        
        upload_path = config.UPLOAD_DIR / filename
        
        # Stream file to disk
        file_size = 0
        with open(upload_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > config.MAX_FILE_SIZE_BYTES:
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File size exceeds maximum of {config.MAX_FILE_SIZE_MB}MB")
                buffer.write(chunk)
        
        original_size = upload_path.stat().st_size
        
        # Generate output filename
        file_stem = Path(filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compressed_{preset}_{file_stem}_{timestamp}.jpg"
        output_filename = safe_filename(output_filename)
        download_path = config.DOWNLOAD_DIR / output_filename
        
        # Compress image
        success = await compress_image_preset(upload_path, download_path, preset)
        
        if not success:
            raise HTTPException(status_code=500, detail="Image compression failed")
        
        compressed_size = download_path.stat().st_size
        compression_ratio = get_compression_ratio(original_size, compressed_size)
        
        # Clean up upload
        if upload_path.exists():
            upload_path.unlink()
        
        return {
            "success": True,
            "filename": output_filename,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "download_url": f"/downloads/{output_filename}",
            "preset": preset
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error compressing image: {str(e)}", exc_info=True)
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error compressing image: {str(e)}")

@app.post("/api/convert-image-format")
async def convert_image_format_endpoint(
    request: Request,
    file: UploadFile = File(...),
    target_format: str = Form(...),
    quality: int = Form(80)
):
    """
    Convert image to a different format with format-specific optimizations.
    
    - **file**: Image file to convert
    - **target_format**: Target format (jpg, png, webp, bmp, gif, tiff, ico)
    - **quality**: Compression quality (30-100) - only affects lossy formats
    """
    upload_path = None
    try:
        # Validate quality parameter
        if quality < 30 or quality > 100:
            raise HTTPException(status_code=400, detail="Quality must be between 30 and 100")
        
        # Validate target format
        valid_formats = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'tiff', 'tif', 'ico']
        if target_format.lower() not in valid_formats:
            raise HTTPException(status_code=400, detail=f"Invalid format. Valid formats: {', '.join(valid_formats)}")
        
        # Sanitize filename
        original_filename = file.filename if file.filename else "uploaded_file"
        filename = safe_filename(original_filename)
        
        # Validate it's an image
        ext = Path(filename).suffix.lower()
        image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.ico', '.svg']
        if ext not in image_exts:
            raise HTTPException(status_code=400, detail="File must be an image")
        
        upload_path = config.UPLOAD_DIR / filename
        
        # Stream file to disk
        file_size = 0
        with open(upload_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > config.MAX_FILE_SIZE_BYTES:
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File size exceeds maximum of {config.MAX_FILE_SIZE_MB}MB")
                buffer.write(chunk)
        
        original_size = upload_path.stat().st_size
        
        # Get original format info
        if PIL_AVAILABLE:
            try:
                with Image.open(upload_path) as img:
                    original_format = img.format or 'unknown'
                    original_mode = img.mode
                    original_size_pixels = img.size
            except:
                original_format = 'unknown'
                original_mode = 'unknown'
                original_size_pixels = (0, 0)
        else:
            original_format = 'unknown'
            original_mode = 'unknown'
            original_size_pixels = (0, 0)
        
        # Generate output filename
        file_stem = Path(filename).stem
        output_ext = f".{target_format.lower()}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"converted_{file_stem}_{target_format.lower()}_{timestamp}{output_ext}"
        output_filename = safe_filename(output_filename)
        download_path = config.DOWNLOAD_DIR / output_filename
        
        # Convert image
        success = await convert_image(upload_path, download_path, target_format, quality)
        
        if not success:
            raise HTTPException(status_code=500, detail="Image format conversion failed")
        
        converted_size = download_path.stat().st_size
        compression_ratio = get_compression_ratio(original_size, converted_size)
        
        # Clean up upload
        if upload_path.exists():
            upload_path.unlink()
        
        # Format information
        format_info = {
            'jpg': {'name': 'JPEG', 'description': 'Best for photos, small file size', 'lossy': True},
            'jpeg': {'name': 'JPEG', 'description': 'Best for photos, small file size', 'lossy': True},
            'png': {'name': 'PNG', 'description': 'Best for graphics with transparency', 'lossy': False},
            'webp': {'name': 'WebP', 'description': 'Modern format, excellent compression', 'lossy': True},
            'bmp': {'name': 'BMP', 'description': 'Uncompressed, large file size', 'lossy': False},
            'gif': {'name': 'GIF', 'description': 'Supports animation, limited colors', 'lossy': True},
            'tiff': {'name': 'TIFF', 'description': 'High quality, preserves detail', 'lossy': False},
            'tif': {'name': 'TIFF', 'description': 'High quality, preserves detail', 'lossy': False},
            'ico': {'name': 'ICO', 'description': 'Windows icon format', 'lossy': False}
        }
        
        info = format_info.get(target_format.lower(), {})
        
        return {
            "success": True,
            "filename": output_filename,
            "original_size": original_size,
            "converted_size": converted_size,
            "compression_ratio": compression_ratio,
            "download_url": f"/downloads/{output_filename}",
            "target_format": target_format.upper(),
            "format_info": info,
            "original_format": original_format,
            "quality": quality,
            "original_dimensions": f"{original_size_pixels[0]}x{original_size_pixels[1]}" if original_size_pixels[0] > 0 else "unknown"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error converting image format: {str(e)}", exc_info=True)
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error converting image format: {str(e)}")

@app.post("/api/merge-pdfs")
async def merge_pdfs_endpoint(
    request: Request,
    files: list[UploadFile] = File(...)
):
    """
    Merge multiple PDF files into one.
    
    - **files**: PDF files to merge (multiple files)
    """
    upload_paths = []
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 PDF files required for merging")
        
        # Save all uploaded PDFs
        pdf_paths = []
        for file in files:
            filename = safe_filename(file.filename if file.filename else "uploaded_file.pdf")
            
            if Path(filename).suffix.lower() != '.pdf':
                raise HTTPException(status_code=400, detail=f"File {filename} is not a PDF")
            
            upload_path = config.UPLOAD_DIR / filename
            upload_paths.append(upload_path)
            
            # Stream file to disk
            file_size = 0
            with open(upload_path, "wb") as buffer:
                while True:
                    chunk = await file.read(8192)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > config.MAX_FILE_SIZE_BYTES:
                        raise HTTPException(status_code=413, detail=f"File size exceeds maximum of {config.MAX_FILE_SIZE_MB}MB")
                    buffer.write(chunk)
            
            pdf_paths.append(upload_path)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"merged_{len(files)}_files_{timestamp}.pdf"
        output_filename = safe_filename(output_filename)
        download_path = config.DOWNLOAD_DIR / output_filename
        
        # Merge PDFs
        success = await merge_pdfs(pdf_paths, download_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="PDF merge failed")
        
        merged_size = download_path.stat().st_size
        total_original_size = sum(p.stat().st_size for p in pdf_paths)
        
        # Clean up uploads
        for upload_path in upload_paths:
            if upload_path.exists():
                upload_path.unlink()
        
        return {
            "success": True,
            "filename": output_filename,
            "files_merged": len(files),
            "total_original_size": total_original_size,
            "merged_size": merged_size,
            "download_url": f"/downloads/{output_filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error merging PDFs: {str(e)}", exc_info=True)
        # Cleanup on error
        for upload_path in upload_paths:
            if upload_path.exists():
                upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error merging PDFs: {str(e)}")

@app.post("/api/compress")
async def compress_file(
    request: Request,
    file: UploadFile = File(...),
    quality: int = Form(80),
    format_type: str = Form("")
):
    """
    Main compression endpoint. Accepts files and compression parameters.
    
    - **file**: File to compress/convert
    - **quality**: Compression quality (30-100)
    - **format_type**: Target format (optional, uses original format if empty)
    """
    upload_path = None
    try:
        # Validate quality parameter
        if quality < 30 or quality > 100:
            raise HTTPException(status_code=400, detail="Quality must be between 30 and 100")
        
        # Sanitize filename
        original_filename = file.filename if file.filename else "uploaded_file"
        filename = safe_filename(original_filename)
        
        # Validate file type
        if not validate_file_type(filename):
            raise HTTPException(status_code=400, detail=f"File type not allowed: {Path(filename).suffix}")
        
        logger.info(f"📥 Received file: {filename}")
        logger.info(f"   Content Type: {file.content_type}")
        logger.info(f"   Quality: {quality}%")
        logger.info(f"   Format: {format_type or 'auto'}")
        
        file_type = get_file_type(filename)
        logger.info(f"   File Type: {file_type}")
        
        # Create upload path
        upload_path = config.UPLOAD_DIR / filename
        
        # Stream file to disk (for large files)
        file_size = 0
        with open(upload_path, "wb") as buffer:
            while True:
                chunk = await file.read(8192)  # Read in 8KB chunks
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > config.MAX_FILE_SIZE_BYTES:
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds maximum of {config.MAX_FILE_SIZE_MB}MB"
                    )
                buffer.write(chunk)
        
        original_size = upload_path.stat().st_size
        logger.info(f"📊 Original size: {format_file_size(original_size)}")
        
        # Generate output filename
        file_stem = Path(filename).stem
        file_ext = f".{format_type}" if format_type else Path(filename).suffix or ".bin"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compressed_{file_stem}_{quality}_{timestamp}{file_ext}"
        output_filename = safe_filename(output_filename)
        download_path = config.DOWNLOAD_DIR / output_filename
        
        # Perform conversion
        conversion_success = await convert_file(upload_path, download_path, file_type, format_type, quality)
        
        if not conversion_success:
            logger.warning("⚠️ Conversion failed, copying original file")
            shutil.copy2(upload_path, download_path)
        
        # Get compressed file size
        compressed_size = download_path.stat().st_size
        compression_ratio = get_compression_ratio(original_size, compressed_size)
        
        logger.info(f"📦 Compressed size: {format_file_size(compressed_size)}")
        logger.info(f"💾 Compression ratio: {compression_ratio:.1f}%")
        logger.info(f"✅ File saved: {output_filename}")
        
        # Clean up upload
        if upload_path.exists():
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error compressing file: {str(e)}", exc_info=True)
        # Cleanup on error
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/downloads/{filename}")
async def download_file(filename: str):
    """Download endpoint to retrieve compressed files."""
    try:
        # Sanitize filename to prevent path traversal
        safe_name = safe_filename(filename)
        file_path = config.DOWNLOAD_DIR / safe_name
        
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"⚠️ File not found: {filename}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Validate file is in download directory (prevent path traversal)
        try:
            file_path.resolve().relative_to(config.DOWNLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        logger.info(f"⬇️ Downloading: {filename}")
        
        mime_type = get_mime_type(filename)
        return FileResponse(
            file_path,
            media_type=mime_type,
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

@app.get("/api/history")
async def get_history():
    """Get list of all compressed files available for download."""
    try:
        files = []
        for file_path in config.DOWNLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_type = get_file_type(file_path.name)
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "size_formatted": format_file_size(stat.st_size),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
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
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

@app.get("/api/formats")
async def get_formats():
    """Get list of all supported file formats."""
    try:
        logger.info("📋 Fetching supported formats")
        return {
            "success": True,
            "formats": SUPPORTED_FORMATS,
            "capabilities": {
                "images": PIL_AVAILABLE,
                "documents": DOCX_AVAILABLE or PANDOC_AVAILABLE,
                "archives": ZIPFILE_AVAILABLE or TARFILE_AVAILABLE or PY7ZR_AVAILABLE,
                "media": FFMPEG_AVAILABLE
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting formats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting formats: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get server statistics and directory information."""
    try:
        uploads_size = get_dir_size(config.UPLOAD_DIR)
        downloads_size = get_dir_size(config.DOWNLOAD_DIR)
        temp_size = get_dir_size(config.TEMP_DIR)
        
        uploads_count = len(list(config.UPLOAD_DIR.glob("*")))
        downloads_count = len(list(config.DOWNLOAD_DIR.glob("*")))
        
        logger.info("📊 Fetching server statistics")
        
        return {
            "success": True,
            "upload_dir": {
                "count": uploads_count,
                "size": uploads_size,
                "size_mb": round(uploads_size / 1024 / 1024, 2),
                "size_formatted": format_file_size(uploads_size)
            },
            "download_dir": {
                "count": downloads_count,
                "size": downloads_size,
                "size_mb": round(downloads_size / 1024 / 1024, 2),
                "size_formatted": format_file_size(downloads_size)
            },
            "temp_dir": {
                "size": temp_size,
                "size_mb": round(temp_size / 1024 / 1024, 2),
                "size_formatted": format_file_size(temp_size)
            },
            "total_storage_mb": round((uploads_size + downloads_size + temp_size) / 1024 / 1024, 2),
            "max_file_size_mb": config.MAX_FILE_SIZE_MB
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/api/cleanup")
async def cleanup_files(days: int = 7):
    """Clean up old files older than specified days."""
    try:
        import time
        current_time = time.time()
        deleted_count = 0
        freed_space = 0
        
        for file_path in config.DOWNLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_age_days = (current_time - file_path.stat().st_ctime) / (24 * 3600)
                if file_age_days > days:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    freed_space += file_size
                    logger.info(f"🗑️ Deleted: {file_path.name}")
        
        logger.info(f"🧹 Cleanup complete: {deleted_count} files removed, {format_file_size(freed_space)} freed")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "freed_space_mb": round(freed_space / 1024 / 1024, 2),
            "freed_space_formatted": format_file_size(freed_space),
            "message": f"Deleted {deleted_count} files older than {days} days"
        }
    
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific compressed file."""
    try:
        safe_name = safe_filename(filename)
        file_path = config.DOWNLOAD_DIR / safe_name
        
        # Validate file is in download directory
        try:
            file_path.resolve().relative_to(config.DOWNLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            logger.warning(f"⚠️ File not found for deletion: {filename}")
            raise HTTPException(status_code=404, detail="File not found")
        
        file_size = file_path.stat().st_size
        file_path.unlink()
        
        logger.info(f"🗑️ Deleted: {filename} ({format_file_size(file_size)})")
        
        return {
            "success": True,
            "message": f"File {filename} deleted successfully",
            "freed_space_mb": round(file_size / 1024 / 1024, 2),
            "freed_space_formatted": format_file_size(file_size)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/api/files/{filename}")
async def get_file_info(filename: str):
    """Get information about a specific file."""
    try:
        safe_name = safe_filename(filename)
        file_path = config.DOWNLOAD_DIR / safe_name
        
        # Validate file is in download directory
        try:
            file_path.resolve().relative_to(config.DOWNLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        stat = file_path.stat()
        file_type = get_file_type(filename)
        
        return {
            "success": True,
            "filename": filename,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "size_formatted": format_file_size(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type": file_type,
            "download_url": f"/downloads/{filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting file info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting file info: {str(e)}")

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
                "uploads": count_files(config.UPLOAD_DIR),
                "downloads": count_files(config.DOWNLOAD_DIR),
                "temp": count_files(config.TEMP_DIR)
            },
            "capabilities": {
                "max_file_size_mb": config.MAX_FILE_SIZE_MB,
                "supported_formats": len([f for formats in SUPPORTED_FORMATS.values() for f in formats]),
                "concurrent_uploads": "unlimited",
                "pil_available": PIL_AVAILABLE,
                "docx_available": DOCX_AVAILABLE,
                "pypdf_available": PYPDF_AVAILABLE,
                "py7zr_available": PY7ZR_AVAILABLE,
                "ffmpeg_available": FFMPEG_AVAILABLE,
                "pandoc_available": PANDOC_AVAILABLE
            },
            "image_formats": {
                "jpg": {"name": "JPEG", "description": "Best for photos, small file size", "lossy": True, "recommended": True},
                "png": {"name": "PNG", "description": "Best for graphics with transparency", "lossy": False, "recommended": True},
                "webp": {"name": "WebP", "description": "Modern format, excellent compression", "lossy": True, "recommended": True},
                "bmp": {"name": "BMP", "description": "Uncompressed, large file size", "lossy": False, "recommended": False},
                "gif": {"name": "GIF", "description": "Supports animation, limited colors", "lossy": True, "recommended": False},
                "tiff": {"name": "TIFF", "description": "High quality, preserves detail", "lossy": False, "recommended": False},
                "ico": {"name": "ICO", "description": "Windows icon format", "lossy": False, "recommended": False}
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Error getting status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

# ============================================================================
# STATIC FILES AND ROOT ROUTE - MUST BE AFTER ALL API ROUTES
# ============================================================================

@app.get("/")
async def read_root():
    """Serve the main index.html file."""
    return FileResponse(config.BASE_DIR / "index.html")

# Mount static files directory for other static assets if needed
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR)), name="static")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """Main entry point for the application."""
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )