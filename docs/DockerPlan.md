# Docker Containerization Plan

## 🎯 Objective
Containerize the LLM Hallucination Detection RAG Assistant to enable easy deployment, portability, and consistent runtime environment across different systems.

---

## 📐 Architecture Overview

### Container Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (rag-network)              │
│                                                               │
│  ┌──────────────────────────┐   ┌──────────────────────────┐│
│  │  Ollama Service          │   │  RAG App Container       ││
│  │  (ollama/ollama)         │◄──┤  (Streamlit)             ││
│  │                          │   │                          ││
│  │  - nomic-embed-text      │   │  - Python 3.11           ││
│  │  - llama3.2:1b           │   │  - Streamlit             ││
│  │  Port: 11434             │   │  - Dependencies          ││
│  │  Volume: ollama-data     │   │  Port: 8501              ││
│  └──────────────────────────┘   └──────────────────────────┘│
│                                           │                  │
│                                           │ (External APIs)  │
│                                           ▼                  │
│                                  ┌────────────────────┐      │
│                                  │ Pinecone API       │      │
│                                  │ SerpAPI            │      │
│                                  └────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Component Breakdown

### 1. Ollama Service Container
**Purpose:** Provides LLM inference and embeddings
**Base Image:** `ollama/ollama:latest`
**Responsibilities:**
- Run nomic-embed-text model for embeddings
- Run llama3.2:1b model for text generation
- Expose API on port 11434

### 2. RAG Application Container
**Purpose:** Main Streamlit application
**Base Image:** `python:3.11-slim`
**Responsibilities:**
- Run Streamlit web interface
- Connect to Ollama service
- Connect to external APIs (Pinecone, SerpAPI)
- Expose UI on port 8501

---

## 🐳 Dockerfile Specification

### Dockerfile for RAG Application

```dockerfile
# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_scrapper.py .
COPY cleanup.sh .
COPY README.md .

# Create directory for Streamlit config
RUN mkdir -p ~/.streamlit

# Create Streamlit config file
RUN echo "\
[server]\n\
headless = true\n\
port = 8501\n\
address = 0.0.0.0\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "rag_scrapper.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🔧 Docker Compose Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Ollama Service
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-service
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    networks:
      - rag-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # RAG Application
  rag-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-assistant
    ports:
      - "8501:8501"
    environment:
      - SERPAPI_API_KEY=${SERPAPI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - OLLAMA_BASE_URL=http://ollama:11434
    networks:
      - rag-network
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s

  # Model Initialization Service (runs once)
  ollama-init:
    image: ollama/ollama:latest
    container_name: ollama-init
    networks:
      - rag-network
    depends_on:
      ollama:
        condition: service_healthy
    volumes:
      - ./init-ollama.sh:/init-ollama.sh
    entrypoint: ["/bin/bash", "/init-ollama.sh"]
    restart: "no"

networks:
  rag-network:
    driver: bridge

volumes:
  ollama-data:
    driver: local
```

---

## 📝 Supporting Files

### .env File (Environment Variables)

```env
# API Keys
SERPAPI_API_KEY=your_serpapi_key_here
PINECONE_API_KEY=your_pinecone_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434

# Pinecone Configuration
INDEX_NAME=rag-embedds
```

### .dockerignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Documentation
*.md
!README.md

# Logs
*.log

# Environment
.env.local
.env.*.local
```

### init-ollama.sh (Model Initialization Script)

```bash
#!/bin/bash

# Wait for Ollama service to be ready
sleep 10

echo "🔄 Initializing Ollama models..."

# Pull required models
echo "📥 Pulling nomic-embed-text..."
ollama pull nomic-embed-text

echo "📥 Pulling llama3.2:1b..."
ollama pull llama3.2:1b

echo "✅ Models initialized successfully!"
```

---

## 🚀 Deployment Instructions

### Prerequisites
- Docker installed (version 20.10+)
- Docker Compose installed (version 1.29+)
- At least 8GB RAM available
- At least 10GB disk space

### Step 1: Prepare Environment
```bash
# Clone repository
cd /path/to/LLM-Hallucination-Detection-Correction-Using-RAG

# Create .env file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### Step 2: Update rag_scrapper.py for Docker
**Modify connection to use environment variable for Ollama:**

```python
# Before (hardcoded localhost)
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def get_llm():
    return OllamaLLM(model="llama3.2:1b")

# After (use environment variable)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

@st.cache_resource
def get_llm():
    return OllamaLLM(
        model="llama3.2:1b",
        base_url=OLLAMA_BASE_URL
    )
```

**Modify API keys to use environment variables:**
```python
# Before (hardcoded)
SERPAPI_API_KEY = "3b5e8c37d4769cf12f42df01df5baa17f207836ee859d08f62d66607cd06cfb4"
PINECONE_API_KEY = "pcsk_4EeaiW_PxmXpizoWmimbi8q9Cn3NTEMQJK9Xz14epbTWVwJGyWbyRp6cQy5BeEuE3AP9ws"

# After (from environment)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not SERPAPI_API_KEY or not PINECONE_API_KEY:
    st.error("❌ Missing API Keys! Please set SERPAPI_API_KEY and PINECONE_API_KEY environment variables.")
    st.stop()
```

### Step 3: Build and Run

```bash
# Make init script executable
chmod +x init-ollama.sh

# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### Step 4: Initialize Ollama Models (First Run Only)
```bash
# Pull models (happens automatically via ollama-init service)
# Or manually:
docker exec -it ollama-service ollama pull nomic-embed-text
docker exec -it ollama-service ollama pull llama3.2:1b

# Verify models
docker exec -it ollama-service ollama list
```

### Step 5: Access Application
```
Open browser: http://localhost:8501
```

---

## 🔍 Verification & Testing

### Health Checks
```bash
# Check Ollama health
curl http://localhost:11434/api/tags

# Check Streamlit health
curl http://localhost:8501/_stcore/health

# Check all services
docker-compose ps
```

### Container Logs
```bash
# All logs
docker-compose logs

# Specific service
docker-compose logs rag-app
docker-compose logs ollama

# Follow logs
docker-compose logs -f rag-app
```

### Resource Usage
```bash
# Check resource usage
docker stats

# Check specific container
docker stats rag-assistant
```

---

## 🛠️ Management Commands

### Stop Services
```bash
docker-compose down
```

### Stop and Remove Volumes
```bash
docker-compose down -v
```

### Rebuild After Code Changes
```bash
docker-compose up -d --build rag-app
```

### Restart Specific Service
```bash
docker-compose restart rag-app
```

### View Ollama Models
```bash
docker exec -it ollama-service ollama list
```

### Shell Access
```bash
# RAG App container
docker exec -it rag-assistant /bin/bash

# Ollama container
docker exec -it ollama-service /bin/bash
```

---

## 📊 Resource Requirements

### Minimum Requirements
- **CPU:** 4 cores
- **RAM:** 8GB
- **Disk:** 10GB
- **Network:** Stable internet for API calls

### Recommended Requirements
- **CPU:** 8 cores
- **RAM:** 16GB
- **Disk:** 20GB SSD
- **Network:** High-speed internet

### Container Resource Limits (Optional)
Add to docker-compose.yml:
```yaml
services:
  rag-app:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  ollama:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

---

## 🔒 Security Considerations

### 1. API Key Management
- **Never commit .env file to version control**
- Use Docker secrets for production:
  ```bash
  echo "your_api_key" | docker secret create serpapi_key -
  ```

### 2. Network Isolation
- Use custom network (already configured)
- Expose only necessary ports
- Use reverse proxy (Nginx/Traefik) in production

### 3. Image Security
- Use official base images
- Regularly update images
- Scan for vulnerabilities:
  ```bash
  docker scan rag-assistant
  ```

### 4. Volume Permissions
- Set appropriate file permissions
- Use named volumes (already configured)
- Regular backups

---

## 🐛 Troubleshooting

### Issue 1: Ollama Models Not Loading
**Symptoms:** RAG app can't connect to Ollama
**Solution:**
```bash
# Check Ollama is running
docker-compose ps ollama

# Manually pull models
docker exec -it ollama-service ollama pull nomic-embed-text
docker exec -it ollama-service ollama pull llama3.2:1b

# Check logs
docker-compose logs ollama
```

### Issue 2: Connection Refused
**Symptoms:** `Connection refused at http://ollama:11434`
**Solution:**
```bash
# Check network
docker network inspect rag-network

# Restart services in order
docker-compose restart ollama
docker-compose restart rag-app
```

### Issue 3: Out of Memory
**Symptoms:** Container crashes or slow performance
**Solution:**
```bash
# Increase Docker memory limit (Docker Desktop)
# Or add resource limits in docker-compose.yml

# Check memory usage
docker stats
```

### Issue 4: Port Already in Use
**Symptoms:** `Error: port 8501 already allocated`
**Solution:**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use 8502 instead

# Or kill process using port
lsof -ti:8501 | xargs kill -9
```

### Issue 5: Slow Model Loading
**Symptoms:** Long startup time
**Solution:**
- Models are cached in volume (ollama-data)
- First run takes longer to download
- Subsequent runs are fast
- Pre-pull models before deployment

---

## 🚀 Production Deployment

### Using Docker Hub
```bash
# Tag image
docker tag rag-assistant:latest username/rag-assistant:v1.0

# Push to registry
docker push username/rag-assistant:v1.0

# Pull on production server
docker pull username/rag-assistant:v1.0
```

### Using Cloud Platforms

#### AWS ECS
```bash
# Create ECR repository
aws ecr create-repository --repository-name rag-assistant

# Tag and push
docker tag rag-assistant:latest <account-id>.dkr.ecr.region.amazonaws.com/rag-assistant:latest
docker push <account-id>.dkr.ecr.region.amazonaws.com/rag-assistant:latest
```

#### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/project-id/rag-assistant

# Deploy
gcloud run deploy rag-assistant --image gcr.io/project-id/rag-assistant --platform managed
```

#### Azure Container Instances
```bash
# Build and push to ACR
az acr build --registry myregistry --image rag-assistant:v1 .

# Deploy
az container create --resource-group myResourceGroup --name rag-assistant --image myregistry.azurecr.io/rag-assistant:v1
```

---

## 📈 Monitoring & Logging

### Log Aggregation
```yaml
# Add to docker-compose.yml
services:
  rag-app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Prometheus Metrics (Optional)
```yaml
# Add metrics exporter
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Health Monitoring Script
```bash
#!/bin/bash
# health-check.sh

while true; do
    if ! curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "⚠️ Health check failed! Restarting..."
        docker-compose restart rag-app
    fi
    sleep 60
done
```

---

## 🎯 Performance Optimization

### 1. Multi-Stage Build
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["streamlit", "run", "rag_scrapper.py"]
```

### 2. Caching Strategies
- Use BuildKit for better caching
- Layer optimization in Dockerfile
- Mount cache for pip:
  ```dockerfile
  RUN --mount=type=cache,target=/root/.cache/pip \
      pip install -r requirements.txt
  ```

### 3. Reduce Image Size
```bash
# Current size
docker images rag-assistant

# Use alpine base (smaller)
FROM python:3.11-alpine

# Remove unnecessary files
RUN rm -rf /var/cache/apk/*
```

---

## 📚 Additional Resources

### Useful Commands Reference
```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes

# Export container
docker export rag-assistant > rag-assistant.tar

# Import container
docker import rag-assistant.tar
```

### Environment Variables Reference
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| SERPAPI_API_KEY | SerpAPI key for web search | Yes | None |
| PINECONE_API_KEY | Pinecone API key | Yes | None |
| OLLAMA_BASE_URL | Ollama service URL | No | http://ollama:11434 |
| INDEX_NAME | Pinecone index name | No | rag-embedds |

---

## ✅ Implementation Checklist

### Setup Phase
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create .dockerignore
- [ ] Create .env.example
- [ ] Create init-ollama.sh
- [ ] Make init script executable

### Code Modifications
- [ ] Update rag_scrapper.py to use OLLAMA_BASE_URL env var
- [ ] Update API keys to use environment variables
- [ ] Add error handling for missing env vars
- [ ] Test locally with environment variables

### Build & Test
- [ ] Build Docker image: `docker-compose build`
- [ ] Start services: `docker-compose up -d`
- [ ] Initialize Ollama models
- [ ] Verify models loaded: `docker exec -it ollama-service ollama list`
- [ ] Test application at http://localhost:8501
- [ ] Test hallucination detection feature
- [ ] Check logs for errors

### Documentation
- [ ] Update README.md with Docker instructions
- [ ] Document environment variables
- [ ] Add troubleshooting section
- [ ] Add production deployment guide

### Production Prep
- [ ] Set up proper secrets management
- [ ] Configure reverse proxy (if needed)
- [ ] Set up monitoring/logging
- [ ] Configure backups for volumes
- [ ] Load testing
- [ ] Security audit

---

## 🎓 Learning Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama)
- [Streamlit Deployment Guide](https://docs.streamlit.io/deploy)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/dev-best-practices/)

---

## 📝 Summary

This Docker implementation provides:
- ✅ Complete containerization of RAG application
- ✅ Multi-container architecture with Ollama
- ✅ Persistent storage for models
- ✅ Health checks and monitoring
- ✅ Easy deployment and scaling
- ✅ Environment-based configuration
- ✅ Production-ready setup

**Estimated Implementation Time:** 2-3 hours
**Complexity:** Medium
**Benefits:** Portability, consistency, easy deployment

---

*End of Docker Plan Document*
