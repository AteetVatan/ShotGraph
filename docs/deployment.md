# Production Deployment Guide

This guide covers deploying ShotGraph to production using Docker.

## Quick Start

1. **Copy the production environment template:**
   ```bash
   cp env.prod.example .env.prod
   ```

2. **Edit `.env.prod` with your actual values:**
   - Set `LLM_API_KEY` (required)
   - Set `API_KEY` (required for production)
   - Configure `CORS_ORIGINS` for your domain
   - Adjust resource limits (`CPU_LIMIT`, `MEMORY_LIMIT`, `WORKERS`)

3. **Build and start the production container:**
   ```bash
   docker-compose -f docker-compose.prod.yml build
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Check logs:**
   ```bash
   docker-compose -f docker-compose.prod.yml logs -f
   ```

5. **Verify health:**
   ```bash
   curl http://localhost:8000/health
   ```

## Production Features

### Security Enhancements

- **Non-root user**: Container runs as `appuser` (UID 1000) instead of root
- **API Key Authentication**: Enabled by default in production (`API_KEY_ENABLED=true`)
- **Rate Limiting**: Configurable per-minute limits
- **CORS Protection**: Restrict origins to your domain
- **Resource Limits**: CPU and memory constraints to prevent resource exhaustion

### Performance Optimizations

- **Multi-stage Build**: Smaller final image size
- **Named Volumes**: Better I/O performance than bind mounts
- **Configurable Workers**: Set `WORKERS` based on CPU cores (typically `2 * cores + 1`)
- **Log Rotation**: Automatic log file rotation (10MB max, 3 files)

### Monitoring

- **Health Checks**: Automatic container health monitoring
- **Structured Logging**: JSON-formatted logs for aggregation
- **Resource Metrics**: CPU/memory usage tracked by Docker

## Configuration

### Environment Variables

Key production environment variables (see `env.prod.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `EXECUTION_PROFILE` | Execution mode | `prod_gpu` |
| `GPU_ENABLED` | Enable GPU support | `true` |
| `LLM_API_KEY` | Together.ai API key | **Required** |
| `API_KEY` | API authentication key | **Required** |
| `API_KEY_ENABLED` | Enable API key auth | `true` |
| `WORKERS` | Uvicorn worker count | `1` |
| `CPU_LIMIT` | CPU limit for container | `4.0` |
| `MEMORY_LIMIT` | Memory limit | `8G` |
| `CORS_ORIGINS` | Allowed CORS origins | `[]` |

### Resource Limits

Adjust based on your hardware:

```yaml
# In docker-compose.prod.yml or .env.prod
CPU_LIMIT=4.0        # Number of CPUs
MEMORY_LIMIT=8G      # Memory limit
WORKERS=1            # Uvicorn workers (2 * cores + 1 recommended)
```

### Worker Configuration

For optimal performance:
- **Single worker**: Good for CPU-bound tasks, simpler debugging
- **Multiple workers**: Better for I/O-bound tasks, requires shared state management
- **Formula**: `WORKERS = 2 * CPU_CORES + 1`

Example for 4-core system:
```bash
WORKERS=9
```

## Deployment Options

### RunPod / Cloud GPU Providers

1. **Build and push image:**
   ```bash
   docker build -t your-registry/shotgraph:latest .
   docker push your-registry/shotgraph:latest
   ```

2. **Deploy on RunPod:**
   - Use the pushed image
   - Set environment variables via RunPod UI
   - Mount volumes for persistent storage
   - Configure GPU access (RTX 3090 or A100 recommended)

### Self-Hosted Server

1. **Prerequisites:**
   - Docker 20.10+
   - Docker Compose 2.0+
   - NVIDIA Docker runtime (for GPU support)
   - CUDA 12.1+ compatible GPU

2. **Deploy:**
   ```bash
   # Clone repository
   git clone <repo-url>
   cd ShotGraph
   
   # Configure environment
   cp env.prod.example .env.prod
   # Edit .env.prod with your values
   
   # Start services
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Update deployment:**
   ```bash
   docker-compose -f docker-compose.prod.yml pull
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Volumes

Production uses named volumes for better performance:

- `shotgraph_output`: Generated videos and artifacts
- `shotgraph_assets`: Static assets and mock data
- `shotgraph_models`: Model weights (read-only)

To backup volumes:
```bash
docker run --rm -v shotgraph_output:/data -v $(pwd):/backup ubuntu tar czf /backup/output-backup.tar.gz /data
```

## Troubleshooting

### Container won't start

1. **Check logs:**
   ```bash
   docker-compose -f docker-compose.prod.yml logs
   ```

2. **Verify GPU access:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Check environment variables:**
   ```bash
   docker-compose -f docker-compose.prod.yml config
   ```

### Health check failing

1. **Verify port is accessible:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check container status:**
   ```bash
   docker-compose -f docker-compose.prod.yml ps
   ```

3. **Inspect container:**
   ```bash
   docker exec -it shotgraph-prod sh
   curl http://localhost:8000/health
   ```

### Out of memory

1. **Increase memory limit in docker-compose.prod.yml:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 16G  # Increase as needed
   ```

2. **Reduce worker count:**
   ```bash
   WORKERS=1
   ```

### API key authentication issues

1. **Verify API_KEY is set:**
   ```bash
   docker exec shotgraph-prod env | grep API_KEY
   ```

2. **Check API key in request:**
   ```bash
   curl -H "X-API-Key: your_key" http://localhost:8000/health
   ```

## Security Checklist

- [ ] API key authentication enabled (`API_KEY_ENABLED=true`)
- [ ] Strong API key set (`API_KEY`)
- [ ] CORS origins restricted to your domain
- [ ] Rate limiting configured appropriately
- [ ] Secrets stored securely (not in git)
- [ ] Container runs as non-root user
- [ ] Resource limits set to prevent DoS
- [ ] Health checks enabled
- [ ] Log rotation configured
- [ ] Regular security updates applied

## Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

### Clean Up

```bash
# Remove old containers and images
docker-compose -f docker-compose.prod.yml down
docker system prune -a

# Clean volumes (WARNING: deletes data)
docker volume rm shotgraph_output shotgraph_assets
```

### Monitor Resources

```bash
# Container stats
docker stats shotgraph-prod

# Disk usage
docker system df
docker volume ls
```

## Support

For issues or questions:
- Check logs: `docker-compose -f docker-compose.prod.yml logs -f`
- Review health endpoint: `curl http://localhost:8000/health`
- Check container status: `docker-compose -f docker-compose.prod.yml ps`
