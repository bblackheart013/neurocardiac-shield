# Installation Guide

## Prerequisites

- macOS or Linux
- Python 3.9+
- GCC compiler
- 4GB RAM minimum

## Steps

### 1. Clone Repository

```bash
git clone https://github.com/bblackheart013/neurocardiac-shield.git
cd neurocardiac-shield
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environments
- Install all dependencies
- Train ML models
- Compile firmware

### 3. Start System

```bash
./run_complete_demo.sh
```

### 4. Access Services

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Verification

```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy", ...}
```

## Common Issues

See [Troubleshooting](Troubleshooting) for solutions.
