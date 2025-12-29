# Deployment Guide

## Quick Start

### Local Development

1. **Run the application:**
   ```bash
   ./run.sh
   ```

   Or manually:
   ```bash
   source venv/bin/activate
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Access the application:**
   - Web UI: http://127.0.0.1:8000
   - API Docs: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc

### Production Deployment

#### Using systemd (Linux)

1. **Create systemd service file** `/etc/systemd/system/file-compression.service`:
   ```ini
   [Unit]
   Description=Universal File Compression Suite
   After=network.target

   [Service]
   Type=simple
   User=www-data
   WorkingDirectory=/path/to/DocFormatting
   Environment="PATH=/path/to/DocFormatting/venv/bin"
   ExecStart=/path/to/DocFormatting/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and start the service:**
   ```bash
   sudo systemctl enable file-compression.service
   sudo systemctl start file-compression.service
   ```

#### Using Docker

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   RUN mkdir -p uploads downloads temp logs

   EXPOSE 8000

   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run:**
   ```bash
   docker build -t file-compression-suite .
   docker run -d -p 8000:8000 \
     -v $(pwd)/uploads:/app/uploads \
     -v $(pwd)/downloads:/app/downloads \
     -v $(pwd)/logs:/app/logs \
     file-compression-suite
   ```

#### Using Nginx Reverse Proxy

1. **Install Nginx:**
   ```bash
   sudo apt-get install nginx
   ```

2. **Create Nginx configuration** `/etc/nginx/sites-available/file-compression`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       client_max_body_size 500M;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

3. **Enable and reload:**
   ```bash
   sudo ln -s /etc/nginx/sites-available/file-compression /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## Environment Variables

Create a `.env` file or set environment variables:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000

# File Size Limits
MAX_FILE_SIZE_MB=500

# CORS Configuration
CORS_ORIGINS=http://localhost:8000,http://your-domain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
LOG_FILE_MAX_BYTES=10485760
LOG_FILE_BACKUP_COUNT=5

# Directories (optional, defaults to relative paths)
UPLOAD_DIR=uploads
DOWNLOAD_DIR=downloads
TEMP_DIR=temp
LOG_DIR=logs
```

## Security Considerations

1. **Change default CORS origins** for production
2. **Set appropriate file size limits** based on your server capacity
3. **Use HTTPS** in production (configure reverse proxy with SSL)
4. **Set up firewall rules** to restrict access
5. **Regular cleanup** of old files using `/api/cleanup` endpoint
6. **Monitor logs** for suspicious activity
7. **Keep dependencies updated** regularly

## Monitoring

- Check logs: `tail -f logs/app.log`
- Health check: `curl http://localhost:8000/api/health`
- Status: `curl http://localhost:8000/api/status`
- Stats: `curl http://localhost:8000/api/stats`

## Troubleshooting

### Port already in use
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

### Permission errors
```bash
# Ensure directories are writable
chmod -R 755 uploads downloads temp logs
```

### FFmpeg/Pandoc not found
Install system dependencies:
```bash
sudo apt-get install ffmpeg pandoc  # Ubuntu/Debian
brew install ffmpeg pandoc          # macOS
```

