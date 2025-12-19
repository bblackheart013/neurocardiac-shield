# NeuroCardiac Shield — Release Checklist

**Version**: 2.0.0
**Date**: December 2025
**Authors**: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir

---

## Pre-Release Verification

### 1. Environment Setup

```bash
# Clone/navigate to repository
cd neurocardiac-shield

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r cloud/requirements.txt
pip install -r ml/requirements.txt
```

### 2. System Verification

```bash
# Run full verification suite
python verify_system.py

# Expected output:
# ============================================================
#   STATUS: ALL CHECKS PASSED (67/67)
# ============================================================
```

- [ ] All 67 checks pass
- [ ] No import errors
- [ ] No missing dependencies

### 3. AI Mentions Check

```bash
# Run dedicated AI mentions scanner
python tools/no_ai_mentions_check.py

# Expected output:
# ============================================================
#   PASSED: No AI mentions found
# ============================================================
```

- [ ] Zero AI/assistant mentions in codebase
- [ ] No banned terms (see tools/no_ai_mentions_check.py for list)

### 4. Demo Execution

```bash
# Run complete demonstration
./run_complete_demo.sh

# This will:
# 1. Start API server on localhost:8000
# 2. Start Dashboard on localhost:8501
```

- [ ] API server starts without errors
- [ ] Dashboard loads and displays data
- [ ] Real-time updates work

### 5. Frontend Build (Optional - for Netlify)

```bash
cd web
npm install
npm run build

# Build output in: web/out/
```

- [ ] No build errors
- [ ] Static export generated
- [ ] All pages render correctly

---

## Code Quality Checks

### 6. Documentation Completeness

- [ ] README.md is current and accurate
- [ ] REPORT.md ready for PDF export
- [ ] EVALUATION.md provides reviewer guidance
- [ ] docs/ARCHITECTURE.md complete
- [ ] docs/DATA_FLOW.md complete
- [ ] docs/ML_PIPELINE.md complete
- [ ] docs/SIMULATION_SCOPE.md complete
- [ ] docs/DEVICE_INTEGRATION.md complete
- [ ] docs/DATA_BIBLE.md complete

### 7. Repository Hygiene

- [ ] .gitignore includes AI tool dirs, venv/, node_modules/
- [ ] No large binary files committed
- [ ] No credentials or secrets in code
- [ ] No TODO or FIXME comments in main code
- [ ] No debug print statements

### 8. Author Attribution

- [ ] Both authors listed in all documentation
- [ ] Dr. Matthew Campisi credited as advisor
- [ ] NYU Tandon and ECE-GY 9953 mentioned
- [ ] Fall 2025 term specified

---

## Final Git Commands

### Stage All Changes

```bash
# Review what will be committed
git status

# Add all changes
git add -A

# Verify staging
git status
```

### Create Release Commit

```bash
git commit -m "Release v2.0.0: Complete submission with device adapters, Next.js frontend, and comprehensive documentation

- Add pluggable device adapter architecture (SimulatedAdapter, BLEAdapter, SerialAdapter)
- Add Gold Schema for unified data format
- Add DATA_BIBLE.md with complete data specification
- Add DEVICE_INTEGRATION.md with hardware setup guide
- Add Next.js Netlify-deployable frontend
- Add AI mentions check tool
- Update verify_system.py to 67 checks
- Update REPORT.md with device integration section
- Add RELEASE_CHECKLIST.md

Authors: Mohd Sarfaraz Faiyaz, Vaibhav D. Chandgir
NYU Tandon School of Engineering | ECE-GY 9953 | Fall 2025
Advisor: Dr. Matthew Campisi"
```

### Push to Remote

```bash
# Push to main branch
git push origin main

# Optional: Create release tag
git tag -a v2.0.0 -m "Release v2.0.0: Final Academic Submission"
git push origin v2.0.0
```

---

## Netlify Deployment (Optional)

### Deploy Frontend

```bash
cd web

# Install Netlify CLI if needed
npm install -g netlify-cli

# Deploy to Netlify
netlify deploy --prod --dir=out
```

---

## Post-Release Verification

- [ ] GitHub repository accessible
- [ ] README displays correctly
- [ ] All documentation links work
- [ ] Netlify site loads (if deployed)
- [ ] verify_system.py passes on fresh clone

---

## Contact

**Mohd Sarfaraz Faiyaz**
NYU Tandon School of Engineering

**Vaibhav D. Chandgir**
NYU Tandon School of Engineering

---

*NeuroCardiac Shield — ECE-GY 9953 Advanced Project — Fall 2025*
