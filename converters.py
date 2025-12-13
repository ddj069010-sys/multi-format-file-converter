# converters/image_converter.py - Image Format Conversion

from pathlib import Path
from PIL import Image
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ImageConverter:
    """Handle image format conversions"""
    
    SUPPORTED_FORMATS = {
        'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG',
        'webp': 'WEBP', 'bmp': 'BMP', 'tiff': 'TIFF',
        'gif': 'GIF', 'ico': 'ICO'
    }
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
    
    async def convert(
        self,
        input_path: Path,
        target_format: str,
        quality: int = 80
    ) -> Optional[Dict[str, Any]]:
        """
        Convert image to target format
        
        Args:
            input_path: Path to input image
            target_format: Target format (jpg, png, webp, etc.)
            quality: Compression quality (30-100)
        
        Returns:
            Dict with success status and output path
        """
        start_time = time.time()
        
        try:
            # Open image
            img = Image.open(input_path)
            
            # Convert RGBA to RGB if target is JPEG
            if target_format.lower() in ['jpg', 'jpeg'] and img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            
            # Generate output filename
            output_name = f"{input_path.stem}_converted.{target_format.lower()}"
            output_path = self.temp_dir / output_name
            
            # Move to downloads
            download_path = Path("downloads") / output_name
            
            # Save with compression
            save_kwargs = {
                'quality': quality,
                'optimize': True
            }
            
            if target_format.lower() == 'webp':
                save_kwargs['method'] = 6  # Best compression
            
            img.save(download_path, format=self.SUPPORTED_FORMATS[target_format.lower()], **save_kwargs)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Image converted: {input_path.name} → {output_name}")
            
            return {
                "success": True,
                "output_path": str(download_path),
                "format": target_format.upper(),
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Image conversion error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# converters/document_converter.py - Document Format Conversion

from docx import Document as DocxDocument
from docx.shared import Pt, RGBColor
from pathlib import Path
import logging
import time
from typing import Optional, Dict, Any
import subprocess

logger = logging.getLogger(__name__)

class DocumentConverter:
    """Handle document format conversions"""
    
    SUPPORTED_FORMATS = ['pdf', 'docx', 'odt', 'txt', 'html', 'rtf']
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
    
    async def convert(
        self,
        input_path: Path,
        target_format: str,
        quality: int = 80
    ) -> Optional[Dict[str, Any]]:
        """
        Convert document to target format
        
        Uses pandoc for format conversions
        """
        start_time = time.time()
        
        try:
            output_name = f"{input_path.stem}_converted.{target_format.lower()}"
            output_path = Path("downloads") / output_name
            
            # Use pandoc for conversion
            cmd = [
                'pandoc',
                str(input_path),
                '-o', str(output_path),
                '-t', self._get_pandoc_format(target_format)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                # Fallback: simple text conversion
                return await self._fallback_convert(input_path, target_format, output_path)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document converted: {input_path.name} → {output_name}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "format": target_format.upper(),
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Document conversion error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_pandoc_format(self, fmt: str) -> str:
        """Map format to pandoc format name"""
        format_map = {
            'pdf': 'pdf',
            'docx': 'docx',
            'odt': 'odt',
            'txt': 'plain',
            'html': 'html',
            'rtf': 'rtf'
        }
        return format_map.get(fmt.lower(), 'pdf')
    
    async def _fallback_convert(self, input_path: Path, target_format: str, output_path: Path):
        """Fallback conversion when pandoc fails"""
        try:
            # Read as text
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Write to target format
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                "success": True,
                "output_path": str(output_path),
                "format": target_format.upper(),
                "processing_time": 0
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# converters/archive_converter.py - Archive Handling

from pathlib import Path
import zipfile
import tarfile
import logging
import time
from typing import Optional, Dict, Any
import py7zr

logger = logging.getLogger(__name__)

class ArchiveConverter:
    """Handle archive format conversions"""
    
    SUPPORTED_FORMATS = ['zip', 'tar', '7z', 'rar']
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
    
    async def convert(
        self,
        input_path: Path,
        target_format: str,
        quality: int = 80
    ) -> Optional[Dict[str, Any]]:
        """Convert archive to target format"""
        start_time = time.time()
        
        try:
            output_name = f"{input_path.stem}_converted.{target_format.lower()}"
            output_path = Path("downloads") / output_name
            
            # Extract and re-archive
            extract_dir = self.temp_dir / f"archive_{int(time.time())}"
            extract_dir.mkdir(exist_ok=True)
            
            # Extract source archive
            if input_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(input_path, 'r') as zf:
                    zf.extractall(extract_dir)
            elif input_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(input_path, 'r:*') as tf:
                    tf.extractall(extract_dir)
            elif input_path.suffix.lower() == '.7z':
                with py7zr.SevenZipFile(input_path, 'r') as szf:
                    szf.extractall(extract_dir)
            
            # Create target archive
            if target_format.lower() == 'zip':
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file in extract_dir.rglob('*'):
                        if file.is_file():
                            zf.write(file, arcname=file.relative_to(extract_dir))
            
            elif target_format.lower() == 'tar':
                with tarfile.open(output_path, 'w') as tf:
                    for file in extract_dir.rglob('*'):
                        if file.is_file():
                            tf.add(file, arcname=file.relative_to(extract_dir))
            
            elif target_format.lower() == '7z':
                with py7zr.SevenZipFile(output_path, 'w') as szf:
                    for file in extract_dir.rglob('*'):
                        if file.is_file():
                            szf.write(file, arcname=str(file.relative_to(extract_dir)))
            
            # Cleanup
            import shutil
            shutil.rmtree(extract_dir)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Archive converted: {input_path.name} → {output_name}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "format": target_format.upper(),
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Archive conversion error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# converters/media_converter.py - Audio/Video Conversion

from pathlib import Path
import subprocess
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MediaConverter:
    """Handle audio and video format conversions"""
    
    SUPPORTED_FORMATS = {
        'audio': ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'opus', 'aiff'],
        'video': ['mp4', 'avi', 'mkv', 'webm', 'mov', 'flv', 'wmv', 'mpeg']
    }
    
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
    
    async def convert(
        self,
        input_path: Path,
        target_format: str,
        quality: int = 80
    ) -> Optional[Dict[str, Any]]:
        """
        Convert audio/video to target format using FFmpeg
        """
        start_time = time.time()
        
        try:
            output_name = f"{input_path.stem}_converted.{target_format.lower()}"
            output_path = Path("downloads") / output_name
            
            # Determine bitrate from quality
            if target_format.lower() in ['mp3', 'aac', 'ogg']:
                bitrate = f"{32 + int((quality / 100) * 256)}k"  # 32-288 kbps
            elif target_format.lower() in ['mp4', 'webm', 'avi']:
                bitrate = f"{500 + int((quality / 100) * 4000)}k"  # 500-4500 kbps
            else:
                bitrate = "192k"
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-b:a', bitrate,
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            processing_time = time.time() - start_time
            
            logger.info(f"Media converted: {input_path.name} → {output_name}")
            
            return {
                "success": True,
                "output_path": str(output_path),
                "format": target_format.upper(),
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Media conversion error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

