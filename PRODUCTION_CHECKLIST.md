# CineMatch V2.0.1 - Production Deployment Checklist

## 🎯 Pre-Deployment Checklist

### **System Health Checks**
- [ ] Run regression tests: `docker exec cinematch-v2-multi-algorithm python test_bug_fixes_regression.py`
- [ ] Verify all 7/7 tests pass
- [ ] Check container status: `docker ps --filter "name=cinematch-v2"`
- [ ] Verify container is HEALTHY
- [ ] Check logs for errors: `docker logs cinematch-v2-multi-algorithm --tail 100 | Select-String "error|exception"`

### **Performance Validation**
- [ ] Load time < 15 seconds (target: 11.76s)
- [ ] Memory usage reasonable (< 6GB active)
- [ ] CPU usage low when idle (< 1%)
- [ ] Zero errors in logs
- [ ] Zero deprecation warnings

### **Bug Fixes Verification**
- [ ] Bug #1: Hybrid loads from disk
- [ ] Bug #2: Content-Based in Hybrid state
- [ ] Bug #13: Cache conflicts resolved
- [ ] Bug #14: Content-Based called in all paths (3x)
- [ ] All deprecation warnings cleared

---

## 🚀 Deployment Commands

### **Full Rebuild and Deploy**
```bash
# Stop, rebuild, and restart with fresh deployment
docker-compose down
docker-compose build
docker-compose up -d

# Wait for healthy status
Start-Sleep -Seconds 20

# Verify deployment
docker ps --filter "name=cinematch-v2"
docker logs cinematch-v2-multi-algorithm --tail 50
```

### **Quick Restart (No Rebuild)**
```bash
docker-compose restart
Start-Sleep -Seconds 15
docker ps --filter "name=cinematch-v2"
```

### **Status Check**
```bash
# Container status
docker ps --filter "name=cinematch-v2" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Resource usage
docker stats cinematch-v2-multi-algorithm --no-stream

# Recent logs
docker logs cinematch-v2-multi-algorithm --tail 50
```

---

## 🧪 Testing Protocol

### **Regression Test Suite**
```bash
# Run full test suite (7 tests)
docker exec cinematch-v2-multi-algorithm python test_bug_fixes_regression.py

# Expected: ALL 7/7 TESTS PASSED
```

### **Manual Testing**
1. **Access Application**: http://localhost:8501
2. **Test User 10 (Dense Profile)**:
   - Navigate to Recommend page
   - Select: Hybrid (Best of All)
   - Enter User ID: 10
   - Click "Get Recommendations"
   - Expected: 10 recommendations in <10 seconds
   
3. **Verify Enhanced Logs** (check Docker logs):
   ```
   • Using weights: SVD=0.XX, UserKNN=0.XX, ItemKNN=0.XX, CBF=0.XX
   ✓ Algorithm timings: SVD=X.Xs, UserKNN=X.Xs, ItemKNN=X.Xs, CBF=X.Xs
   ✓ Recommendations collected: SVD=10, UserKNN=10, ItemKNN=10, CBF=10
   • Aggregating 40 recommendations from 4 algorithms
   ```

4. **Test Different User Types**:
   - **New User** (ID: 999999): Tests cold-start (SVD + ItemKNN + CBF)
   - **Sparse User** (ID: 100): Tests 3-algorithm blend
   - **Dense User** (ID: 10): Tests full 4-algorithm ensemble

---

## 🔍 Troubleshooting Guide

### **Container Won't Start**
```bash
# Check logs for errors
docker logs cinematch-v2-multi-algorithm

# Remove and rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### **Slow Performance (>15s load)**
1. Check container resources: `docker stats cinematch-v2-multi-algorithm --no-stream`
2. Verify models are pre-trained (not training on-demand)
3. Check log for "Loading pre-trained model" messages
4. Run regression tests to verify performance benchmark

### **Recommendations Not Generated**
1. Check logs: `docker logs cinematch-v2-multi-algorithm --tail 100`
2. Verify all 4 algorithms loaded
3. Test with known good user (ID: 10)
4. Check for "N/A" metrics in UI

### **Hybrid Shows "N/A"**
- **Issue**: Hybrid model not loading from disk
- **Fix**: Verify `models/hybrid_model.pkl` exists
- **Test**: Run Bug #1 regression test

### **Content-Based Not Called**
- **Issue**: Bug #14 regression
- **Fix**: Verify commit efeae2a is deployed
- **Test**: Run Bug #14 regression test
- **Check**: Search logs for "content_based_recs"

---

## 📊 Monitoring Metrics

### **Key Performance Indicators**
- **Load Time**: < 15 seconds (actual: 11.76s)
- **Recommendation Time**: < 10 seconds
- **Memory Usage**: < 6.5GB active
- **CPU Idle**: < 1%
- **Error Rate**: 0%
- **Test Pass Rate**: 100% (7/7)

### **Log Monitoring Commands**
```bash
# Check for errors
docker logs cinematch-v2-multi-algorithm --tail 100 | Select-String "error|exception|traceback" -CaseSensitive:$false

# Check for warnings
docker logs cinematch-v2-multi-algorithm --tail 100 | Select-String "warning|deprecated" -CaseSensitive:$false

# Verify 4-algorithm ensemble
docker logs cinematch-v2-multi-algorithm --tail 100 | Select-String "Algorithm timings"

# Check load times
docker logs cinematch-v2-multi-algorithm --tail 100 | Select-String "loaded in"
```

---

## 🔄 Maintenance Schedule

### **Daily**
- [ ] Check container health status
- [ ] Monitor memory usage
- [ ] Review error logs

### **Weekly**
- [ ] Run full regression test suite
- [ ] Check performance benchmarks
- [ ] Review and rotate logs

### **Monthly**
- [ ] Review algorithm weights (may need retraining)
- [ ] Analyze recommendation quality metrics
- [ ] Check for framework updates (Streamlit, pandas, etc.)
- [ ] Update documentation if needed

### **Quarterly**
- [ ] Retrain models with fresh data
- [ ] Benchmark against new algorithms
- [ ] Review and optimize weights
- [ ] Performance audit

---

## 📚 Quick Reference

### **Important Files**
- `src/algorithms/hybrid_recommender.py` - Hybrid algorithm (Bug #14 fix)
- `src/algorithms/user_knn_recommender.py` - User KNN (Bug #13 fix)
- `src/algorithms/item_knn_recommender.py` - Item KNN (Bug #13 fix)
- `src/algorithms/algorithm_manager.py` - Algorithm loading (Bug #1 fix)
- `test_bug_fixes_regression.py` - Automated test suite
- `BUGFIXES.md` - Complete bug documentation (v2.1)

### **Key Commits**
- **efeae2a**: Bug #14 fix - Content-Based integration
- **52a3e87**: Documentation - BUGFIXES.md v2.1
- **28dae7b**: Enhanced monitoring - Algorithm logging
- **622f0aa**: Regression tests - Automated validation

### **URLs**
- **Application**: http://localhost:8501
- **GitHub**: https://github.com/M7MDRAUF/DataSets

### **Port Mapping**
- **Host**: 8501
- **Container**: 8501
- **Protocol**: TCP

---

## ✅ Production Ready Criteria

### **All Must Pass**
- [x] All 14 bugs fixed and tested
- [x] 7/7 regression tests passing
- [x] Load time < 15 seconds
- [x] Zero errors in logs
- [x] Zero deprecation warnings
- [x] Container status: HEALTHY
- [x] All 4 algorithms active (SVD, UserKNN, ItemKNN, ContentBased)
- [x] Enhanced monitoring operational
- [x] Documentation complete (v2.1)
- [x] Version control (GitHub)

---

## 🚨 Emergency Procedures

### **System Down**
1. Check container: `docker ps -a`
2. Start container: `docker start cinematch-v2-multi-algorithm`
3. If fails, rebuild: `docker-compose up -d --build`

### **Data Corruption**
1. Stop container: `docker-compose down`
2. Restore from backup (if available)
3. Rebuild: `docker-compose build --no-cache`
4. Deploy: `docker-compose up -d`
5. Verify: Run regression tests

### **Performance Degradation**
1. Check resources: `docker stats cinematch-v2-multi-algorithm --no-stream`
2. Review logs for bottlenecks
3. Consider increasing Docker memory limit
4. Restart container: `docker-compose restart`

---

**Document Version**: 1.0  
**Last Updated**: November 12, 2025  
**Status**: Production Ready ✅
