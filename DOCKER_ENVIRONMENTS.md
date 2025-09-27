# Rare Disease Helper Chatbot - Docker Management Script

## Development Environment

### Start Development Environment (Hot Reload Enabled)
```powershell
# Start development container with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Or run in background
docker-compose -f docker-compose.dev.yml up -d --build

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

## Production Environment

### Start Production Environment
```powershell
# Start production container
docker-compose -f docker-compose.prod.yml up --build

# Or run in background
docker-compose -f docker-compose.prod.yml up -d --build

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop production environment
docker-compose -f docker-compose.prod.yml down
```

## Environment Files

### Development Features:
- **Hot Reload**: Code changes are automatically detected and applied
- **Volume Mounting**: Source code is mounted for live editing
- **Debug Mode**: Enhanced logging and file watching
- **Interactive Shell**: TTY enabled for debugging

### Production Features:
- **Optimized Build**: No source code mounting
- **Resource Limits**: Memory limits for stability
- **Production Settings**: Headless mode, no usage stats
- **Auto Restart**: Container restarts on failure

## Quick Commands

```powershell
# Development
docker-compose -f docker-compose.dev.yml up -d --build

# Production
docker-compose -f docker-compose.prod.yml up -d --build

# Clean up everything
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.prod.yml down -v
```

## Environment Variables

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_api_key_here
```
