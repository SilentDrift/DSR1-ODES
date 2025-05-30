# DeepSeek‑ODE‑Tutor – **Production-Ready Setup**

A complete, production‑ready skeleton that you can deploy on a **Windows GPU server** and run either as a command‑line tutor or a REST API.

---

## 🚀 **Quick Start**

1. **Clone and setup**
```powershell
git clone <your-repo-url> DeepSeek-ODE-Tutor
cd DeepSeek-ODE-Tutor
scripts\setup.ps1
```

2. **Test your setup**
```powershell
python test_setup.py
```

3. **Run interactive tutor**
```powershell
scripts\run_cli.ps1
```

4. **Or run as API server**
```powershell
scripts\run_api.ps1
```

---

## 📁 **Project Structure**

```
DeepSeek-ODE-Tutor/
├── README.md                 ← Setup & usage instructions
├── environment.yml           ← Conda environment (GPU-optimized)
├── requirements.txt          ← Pip fallback dependencies
├── test_setup.py             ← Verify installation
├── download_weights.py       ← Download model weights (~3.6 GB)
├── .gitignore               
├── config/
│   └── settings.yaml         ← Model & generation parameters
├── models/                   ← Model weights (auto-created)
├── src/
│   ├── __init__.py
│   ├── chatbot.py            ← Core inference engine
│   ├── cli.py                ← Interactive console
│   └── api.py                ← FastAPI web service
└── scripts/
    ├── setup.ps1             ← One-click setup
    ├── run_cli.ps1           ← Launch CLI
    └── run_api.ps1           ← Launch API server
```

---

## ⚙️ **Prerequisites**

### Required Software
- **Git for Windows** 
- **Miniconda ≥ 23.11** ([Download](https://docs.conda.io/en/latest/miniconda.html))
- **NVIDIA Driver + CUDA 12.1+** (for GPU acceleration)

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥4 GB VRAM (recommended)
- **RAM**: ≥8 GB system RAM  
- **Storage**: ≥5 GB free space for model weights

---

## 🔧 **Detailed Setup**

### 1. Clone Repository
```powershell
git clone <your-repo-url> DeepSeek-ODE-Tutor
cd DeepSeek-ODE-Tutor
```

### 2. Run Automated Setup
```powershell
scripts\setup.ps1
```

This script will:
- Create `deepseek-ode` conda environment
- Install PyTorch with CUDA support
- Install all Python dependencies  
- Download DeepSeek-R1-1.5B model weights (~3.6 GB)

### 3. Verify Installation
```powershell
python test_setup.py
```

Expected output:
```
🧪 DeepSeek ODE Tutor Setup Test

🔧 Testing imports...
  ✅ torch
  ✅ transformers
  ✅ accelerate
  ...
✅ All packages available

🔧 Testing CUDA...
  ✅ CUDA available: NVIDIA GeForce RTX 4090
  📊 GPU Memory: 24.0 GB

📊 Test Results: 5/5 passed
🎉 All tests passed! Your setup looks good.
```

---

## 🖥️ **Usage**

### Interactive CLI Mode
```powershell
scripts\run_cli.ps1
```

Example session:
```
✅ DeepSeek ODE Tutor ready! (type 'exit' to quit)

You ➜ Solve dy/dx = xy, y(0) = 1

Tutor ➜ I'll solve this separable differential equation step by step.

Given: $\frac{dy}{dx} = xy$ with initial condition $y(0) = 1$

**Step 1: Separate variables**
$\frac{dy}{y} = x \, dx$

**Step 2: Integrate both sides**
$\int \frac{dy}{y} = \int x \, dx$
$\ln|y| = \frac{x^2}{2} + C$

**Step 3: Solve for y**
$y = Ae^{\frac{x^2}{2}}$ where $A = e^C$

**Step 4: Apply initial condition**
$y(0) = 1 = Ae^0 = A$
Therefore $A = 1$

**Final Solution:**
$$y = e^{\frac{x^2}{2}}$$

**Verification:** $\frac{dy}{dx} = xe^{\frac{x^2}{2}} = xy$ ✓
```

### REST API Mode
```powershell
scripts\run_api.ps1
```

The API will be available at `http://localhost:8000`

**Endpoints:**
- `GET /` - API status
- `GET /health` - Health check  
- `POST /generate` - Generate ODE solution

**Example API usage:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "Solve y'' + 4y = 0"}'
```

---

## 🛠️ **Customization**

| What to change | Where to edit |
|----------------|---------------|
| Model (e.g., use 7B variant) | `config/settings.yaml` → `model.local_path` |
| Generation parameters | `config/settings.yaml` → `generation.*` |
| System prompt | `config/settings.yaml` → `prompts.system` |
| Enable quantization | `src/chatbot.py` → add `load_in_4bit=True` |
| API port | `scripts/run_api.ps1` → change `--port 8000` |

---

## 🔍 **Troubleshooting**

### Common Issues

**❌ "CUDA out of memory"**
```yaml
# In config/settings.yaml, reduce batch size:
generation:
  max_new_tokens: 256  # Reduce from 512
```

**❌ "conda: command not found"**
- Install Miniconda and restart PowerShell
- Or use pip: `pip install -r requirements.txt`

**❌ "Model not found"**
```powershell
# Re-download model weights:
python download_weights.py
```

**❌ "Import errors"**
```powershell
# Recreate environment:
conda env remove -n deepseek-ode
scripts\setup.ps1
```

**❌ API won't start**
- Check if port 8000 is in use: `netstat -an | findstr :8000`
- Use different port: Edit `scripts\run_api.ps1`

### Performance Optimization

**For slower GPUs (< 8GB VRAM):**
```python
# In src/chatbot.py, enable quantization:
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype="auto", 
    device_map="auto",
    load_in_4bit=True,  # Add this line
    trust_remote_code=True,
)
```

**For CPU-only systems:**
```python
# In src/chatbot.py, force CPU:
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype="float32",
    device_map="cpu",  # Change this
    trust_remote_code=True,
)
```

---

## 📊 **System Requirements Met**

✅ **Windows GPU server ready**  
✅ **One-click setup with PowerShell**  
✅ **GPU acceleration with CUDA**  
✅ **Production-grade error handling**  
✅ **Both CLI and API interfaces**  
✅ **Configurable and extensible**  

https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
