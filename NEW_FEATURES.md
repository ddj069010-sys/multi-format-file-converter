# New Converter Features

## 🎉 Added Unique Converters

### 1. Image Resizer (`/api/resize-image`)
Resize images to specific dimensions with optional aspect ratio preservation.

**Features:**
- Set custom width and/or height
- Option to maintain aspect ratio
- High-quality LANCZOS resampling
- Supports: JPG, PNG, GIF, WebP, BMP, TIFF

**Frontend:** "Resize" tab in the navigation

### 2. Image Compressor with Presets (`/api/compress-image-preset`)
Quick compression using preset quality levels - no manual quality adjustment needed.

**Presets:**
- **Low**: 40% quality (smallest file size)
- **Medium**: 70% quality (balanced)
- **High**: 85% quality (larger size)
- **Ultra**: 95% quality (best quality)

**Frontend:** Available in Compress tab as "Quick Compress (Preset)" button

### 3. PDF Merger (`/api/merge-pdfs`)
Combine multiple PDF files into a single document.

**Features:**
- Merge 2 or more PDF files
- Maintains page order
- Preserves all PDF content

**Frontend:** "Merge PDF" tab in the navigation

**Requirements:** pypdf library (included in requirements.txt)

## How to Use

### Image Resizer
1. Click "Resize" tab
2. Upload an image
3. Set width and/or height (pixels)
4. Toggle "Maintain Aspect Ratio" if needed
5. Click "Resize Image"

### Image Compressor Preset
1. Go to "Compress" tab
2. Upload image(s)
3. Click "Quick Compress (Preset)" button
4. Files are compressed with medium preset by default

### PDF Merger
1. Click "Merge PDF" tab
2. Upload 2 or more PDF files (drag & drop or browse)
3. Arrange files in desired order (use Remove to reorder)
4. Click "Merge PDFs"

## Technical Details

- All new endpoints follow the same security and validation patterns
- File size limits apply to all operations
- Rate limiting protects all endpoints
- Proper error handling and logging
- Clean file cleanup on errors

