# utils/file_utils.py - File Utility Functions

from pathlib import Path
import os
import re
from typing import Optional

def get_file_type(file_extension: str) -> str:
    """
    Determine file type from extension
    
    Returns: 'image', 'document', 'archive', 'media', 'data', or 'unknown'
    """
    
    ext = file_extension.lower().lstrip('.')
    
    # Image formats
    image_exts = {
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'webp',
        'svg', 'ico', 'heic', 'heif', 'psd', 'raw', 'cr2', 'nef',
        'arw', 'tga', 'dds', 'kra', 'eps', 'ai', 'cdr', 'wmf', 'emf'
    }
    
    # Document formats
    doc_exts = {
        'doc', 'docx', 'odt', 'pdf', 'txt', 'rtf', 'pages', 'wpd',
        'tex', 'html', 'htm', 'xml', 'md', 'markdown', 'eml', 'msg'
    }
    
    # Spreadsheet formats
    spreadsheet_exts = {
        'xls', 'xlsx', 'ods', 'csv', 'tsv', 'json', 'xml'
    }
    
    # Presentation formats
    presentation_exts = {
        'ppt', 'pptx', 'odp', 'key'
    }
    
    # Archive formats
    archive_exts = {
        'zip', 'rar', '7z', 'tar', 'gz', 'bz2', 'xz', 'iso',
        'cab', 'dmg', 'sit', 'lz', 'lzma'
    }
    
    # Audio formats
    audio_exts = {
        'mp3', 'wav', 'flac', 'aac', 'm4a', 'ogg', 'opus', 'aiff',
        'aif', 'alac', 'ape', 'midi', 'mid', 'au', 'snd', 'wma'
    }
    
    # Video formats
    video_exts = {
        'mp4', 'm4v', 'avi', 'mov', 'qt', 'wmv', 'mkv', 'flv',
        'webm', 'mpg', 'mpeg', 'avchd', 'ogv', 'm2ts', 'mts', 'ts'
    }
    
    # Data formats
    data_exts = {
        'json', 'xml', 'yaml', 'yml', 'toml', 'csv', 'sql', 'dat', 'bin'
    }
    
    # 3D/CAD formats
    cad_exts = {
        'dwg', 'dxf', 'sldprt', 'sldasm', 'prt', 'ipt', 'iam',
        'catpart', 'catproduct', 'step', 'stp', 'iges', 'igs',
        'stl', 'obj', 'fbx', '3mf', 'dae', 'blend', 'max', 'mb', 'ma'
    }
    
    # EBook formats
    ebook_exts = {'epub', 'mobi', 'azw', 'azw3', 'kfx', 'fb2', 'iba'}
    
    # Font formats
    font_exts = {'ttf', 'otf', 'woff', 'woff2', 'eot', 'svg'}
    
    # Program/executable formats (will be rejected)
    exec_exts = {
        'exe', 'msi', 'bat', 'cmd', 'dll', 'com', 'apk', 'ipa',
        'aab', 'app', 'deb', 'rpm', 'sh'
    }
    
    if ext in image_exts:
        return 'image'
    elif ext in doc_exts or ext in spreadsheet_exts or ext in presentation_exts:
        return 'document'
    elif ext in archive_exts:
        return 'archive'
    elif ext in audio_exts:
        return 'media'
    elif ext in video_exts:
        return 'media'
    elif ext in data_exts:
        return 'data'
    elif ext in cad_exts:
        return '3d'
    elif ext in ebook_exts:
        return 'ebook'
    elif ext in font_exts:
        return 'font'
    elif ext in exec_exts:
        return 'executable'
    else:
        return 'unknown'


def get_mime_type(filename: str) -> str:
    """Get MIME type for file"""
    ext = Path(filename).suffix.lower()
    
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.mp4': 'video/mp4',
        '.mp3': 'audio/mpeg',
        '.zip': 'application/zip',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.csv': 'text/csv',
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.svg': 'image/svg+xml',
        '.tar': 'application/x-tar',
        '.gz': 'application/gzip',
        '.7z': 'application/x-7z-compressed',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.aac': 'audio/aac',
        '.ogg': 'audio/ogg',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm'
    }
    
    return mime_types.get(ext, 'application/octet-stream')


def safe_filename(filename: str) -> str:
    """
    Generate safe filename by removing/replacing dangerous characters
    """
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace dangerous characters
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    filename = re.sub(r'[\x00-\x1f]', '', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    max_length = 200 - len(ext)
    name = name[:max_length]
    
    return name + ext


def format_file_size(size_bytes: float) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio percentage"""
    if original_size == 0:
        return 0
    return ((original_size - compressed_size) / original_size) * 100