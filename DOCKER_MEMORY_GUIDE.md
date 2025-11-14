# Docker Memory Configuration Guide

## Issue: SVD (Surprise) Model Requires 8GB+ Memory

The SVD (Surprise) model requires **~4.2GB RAM** during loading (3.8x the 1115MB file size). With Docker Desktop's default **6.645GB limit**, the container cannot allocate sufficient memory.

## Solution: Increase Docker Desktop Memory

### Windows (Docker Desktop):

1. **Open Docker Desktop**
2. Click the **Settings** icon (⚙️) in the top-right
3. Navigate to **Resources** → **Advanced**
4. Increase **Memory** slider to **8GB or higher**
5. Click **Apply & Restart**

### macOS (Docker Desktop):

1. **Open Docker Desktop**
2. Click **Docker Desktop** in menu bar → **Preferences**
3. Go to **Resources** → **Advanced**
4. Increase **Memory** slider to **8GB or higher**
5. Click **Apply & Restart**

### Linux (Docker Engine):

Docker on Linux uses host memory directly. No configuration needed if host has 8GB+ RAM.

To set container-specific limits in `docker-compose.yml`:
```yaml
services:
  cinematch-v2:
    mem_limit: 8g
    memswap_limit: 8g
```

## Verification

After increasing memory, verify the change:

```bash
# Check Docker stats
docker stats cinematch-v2-multi-algorithm --no-stream

# Should show LIMIT of 8GB or higher
```

## Testing All 5 Models

After increasing Docker memory, test all models:

```bash
# Enter container
docker exec -it cinematch-v2-multi-algorithm bash

# Run test script
python test_docker_memory_fix.py
```

Expected output:
```
Models loaded: 5/5 (100%)
✓ SUCCESS! All 5 models loaded successfully!
```

## Memory Requirements by Model

| Model | File Size | RAM Required | Overhead |
|-------|-----------|--------------|----------|
| SVD (Surprise) | 1115 MB | ~4200 MB | 3.8x |
| User-KNN | 1114 MB | ~1700 MB | 1.5x |
| Item-KNN | 1108 MB | ~1700 MB | 1.5x |
| Content-Based | 1060 MB | ~1600 MB | 1.5x |
| SVD (sklearn) | 910 MB | ~1400 MB | 1.5x |
| **TOTAL** | **5307 MB** | **~10600 MB** | **2.0x avg** |

**Recommended Docker Memory:** **8-10GB** for comfortable operation

## Alternative: Deploy Without SVD (Surprise)

If you cannot increase Docker memory, you can deploy with 4/5 algorithms:

```python
# In algorithm_manager.py, skip SVD Surprise loading
MODELS_TO_LOAD = [
    'svd_sklearn',      # ✓ Works with 6.6GB
    'user_knn',         # ✓ Works with 6.6GB
    'item_knn',         # ✓ Works with 6.6GB
    'content_based',    # ✓ Works with 6.6GB
    # 'svd_surprise',   # ✗ Requires 8GB+
]
```

This provides **80% functionality** (4/5 algorithms) within 6.6GB Docker limit.

## Performance Notes

- **Sequential Loading:** Models load one at a time with garbage collection to minimize peak memory
- **Peak Memory Usage:** ~8.8GB when loading all 5 models simultaneously
- **Final Memory Usage:** ~7GB after models are loaded and garbage collected
- **Startup Time:** ~35 seconds to load all 5 models

## Troubleshooting

### Error: "Cannot allocate memory" / Killed

**Cause:** Docker Desktop memory limit too low

**Solution:** Increase Docker memory to 8GB+ following steps above

### Container Crashes During Startup

**Cause:** All models loading simultaneously exhausting memory

**Solution:** Use `load_models_sequential()` from `src/utils/memory_manager.py`

### Docker Stats Shows Lower Limit

**Cause:** Docker Desktop setting not applied

**Solution:** 
1. Restart Docker Desktop completely
2. Verify settings saved
3. Rebuild containers: `docker-compose down && docker-compose up --build`

## System Requirements

**Minimum (4/5 algorithms):**
- Docker Desktop: 6.6GB memory
- Host RAM: 8GB+

**Recommended (5/5 algorithms):**
- Docker Desktop: 8-10GB memory  
- Host RAM: 16GB+

**Optimal (Production):**
- Docker Desktop: 12GB memory
- Host RAM: 16-32GB+
- SSD storage for model files

---

**CineMatch V2.1.1** - Professional multi-algorithm recommendation system
