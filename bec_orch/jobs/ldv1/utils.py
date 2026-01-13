"""
Utility functions for S3 operations, URI handling, and image task creation.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import boto3  # type: ignore
    import botocore  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None
    botocore = None

from .types_common import ImageTask

def _normalize_uri(path_or_uri: str) -> str:
    """Convert local path to file:// URI if needed, otherwise return as-is."""
    if path_or_uri.startswith(("s3://", "file://")):
        return path_or_uri.rstrip('/')
    # Convert to absolute path and then to file:// URI
    abs_path = os.path.abspath(path_or_uri)
    # On Windows, handle backslashes
    if os.name == 'nt':
        abs_path = abs_path.replace('\\', '/')
        # Ensure proper format: file:///C:/...
        if abs_path[1] == ':':
            abs_path = '/' + abs_path
    return f"file://{abs_path}"


def _join_uri(base_uri: str, filename: str) -> str:
    """Join a filename to a base URI (s3:// or file://)."""
    base_uri = base_uri.rstrip('/')
    if base_uri.startswith("s3://"):
        return f"{base_uri}/{filename}"
    elif base_uri.startswith("file://"):
        # For file:// URIs, we need to handle path joining properly
        path_part = base_uri[7:]  # Remove "file://"
        if os.name == 'nt' and path_part.startswith('/') and len(path_part) > 1 and path_part[2] == ':':
            # Windows: file:///C:/path -> C:/path
            path_part = path_part[1:]
        elif os.name == 'nt' and not path_part.startswith('/'):
            # Already a Windows path
            pass
        joined = os.path.join(path_part, filename).replace('\\', '/')
        if os.name == 'nt' and joined[1] == ':':
            # Ensure file:///C:/ format for Windows
            return f"file:///{joined}"
        return f"file://{joined}"
    else:
        # Plain path
        joined = os.path.join(base_uri, filename)
        return _normalize_uri(joined)


def _get_local_image_tasks(input_folder: str) -> List[ImageTask]:
    """Scan input_folder for image files and create ImageTask list."""
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")
    
    image_tasks = []
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            source_uri = _normalize_uri(str(file_path))
            image_tasks.append(ImageTask(
                source_uri=source_uri,
                img_filename=file_path.name
            ))
    
    if not image_tasks:
        raise ValueError(f"No image files found in {input_folder}")
    
    return image_tasks

