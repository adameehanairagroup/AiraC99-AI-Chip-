# ⚡ AIRA CORE: Sovereign Edge AI Inference Engine

> **Democratizing AI Compute Across Every Architecture: Android, Windows, Linux & Airlys OS**  
> *Architected by **Er. Adam Eehan** (Founder & CEO, Aira Group of Technology)*  
> *Co-Pilot: **Aira 3 Ultra** (Sovereign Engine v3.6)*  
>  
> 🇮🇳 **This Repo is Built in India, dedicated for Indian students and Indian developers. This is my personal hobby project.**

---

## 🏛️ 1. Architecture Overview

**AIRA CORE** is a zero-dependency, bare-metal C99 inference runtime engineered to execute modern Large Language Models and Neural Networks at peak physical efficiency on resource-constrained, low-end hardware. 

By eliminating the Python runtime overhead, Global Interpreter Lock (GIL), and heavyweight external dependencies, Aira Core compiles into a compact, blazing-fast native binary (< 2 MB) that operates directly on CPU vector units (ARM NEON, AVX2) and memory-maps weights directly from disk without duplicating RAM.

```
                          +-------------------------------------+
                          |             Aira Core               |
                          |  Transformer Decoder Forward Pass   |
                          +------------------+------------------+
                                             |
             +-------------------------------+-------------------------------+
             |                               |                               |
  +----------v----------+         +----------v----------+         +----------v----------+
  |    aira_math.c      |         |   aira_tokenizer.c  |         |      hal_os.h       |
  | SIMD / NEON / AVX2  |         |  BPE & Vocab Decode |         |  Hardware/OS Abstr. |
  +---------------------+         +---------------------+         +----------+----------+
                                                                             |
                   +---------------------+---------------------+-------------+-----------+
                   |                     |                     |                         |
         +---------v---------+ +---------v---------+ +---------v---------+     +---------v---------+
         |   hal_android.c   | |    hal_linux.c    | |   hal_windows.c   |     |   hal_airlys.c    |
         |  Android / Bionic | | GNU/Linux (POSIX) | | Microsoft Win32   |     | Airlys Bare-Metal |
         +-------------------+ +-------------------+ +-------------------+     +-------------------+
```

---

## 🌐 2. Multi-Platform Support Matrix

| Platform | Subsystem / HAL | Acceleration | Memory Model | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Android** | Bionic Libc / Termux / NDK | ARMv8-A NEON / ASIMD | `mmap(MAP_SHARED)` Page Cache | **Production Ready** |
| **GNU/Linux** | POSIX / glibc / musl | AVX2 / FMA / NEON | POSIX `mmap` + `madvise` | **Production Ready** |
| **Windows** | Win32 API / MSVC / MinGW | AVX2 / SSE4.2 / QPC | `CreateFileMapping` / `MapViewOfFile` | **Production Ready** |
| **Airlys OS** | Bare-Metal Freestanding Kernel | RDTSC / Custom Paging | Kernel Page Frames / Direct Physical | **Production Ready** |

---

## 🔬 3. Hardware & OS Abstraction Layer (HAL)

Aira Core segregates platform-dependent operations into isolated HAL modules:

1. **`hal_android.c`**:
   - Optimized for low-power mobile SoCs.
   - Leverages ARM NEON vectorization (`arm_neon.h`) to perform 4x 32-bit float matrix multiplications in a single clock cycle.
   - Respects Android Linux low-memory killer (LMK) via zero-copy shared page caches.

2. **`hal_linux.c`**:
   - High-throughput POSIX server implementation.
   - Utilizes `clock_gettime(CLOCK_MONOTONIC)` for sub-nanosecond token benchmarking.
   - Dynamic hardware feature discovery via `/proc/cpuinfo`.

3. **`hal_windows.c`**:
   - Native Win32 API implementation with zero Cygwin/POSIX translation penalty.
   - Employs `QueryPerformanceCounter` & `QueryPerformanceFrequency` for high-resolution timing.
   - Enforces UTF-8 console codepages (`CP_UTF8`) for emojis and international token rendering.

4. **`hal_airlys.c`**:
   - **Freestanding Bare-Metal Implementation** designed specifically for **Airlys OS S1 Duo**.
   - Zero standard library dependencies (`-ffreestanding -nostdlib`).
   - Hooks directly into Airlys Kernel Memory Allocator (`airlys_kmalloc`) and hardware cycle counters (`rdtsc` / `cntvct_el0`).

---

## 🚀 4. Building & Running

### A. Android (Termux) & Linux
```bash
# Clone and enter directory
cd aira-core

# Build native binary with auto-detected platform HAL
make

# Run mathematical verification test suite
make test

# Execute inference with model weights
./aira-core -m model.bin -z tokenizer.bin -p "Hello Aira" -t 0.8 -s 128
```

### B. Microsoft Windows (Cross-compilation or Native MSVC)
```bash
# Cross-compile for Windows x64 from Linux/Android
make windows
# Outputs aira-core.exe
```

### C. Airlys OS Kernel Linking
```bash
# Build freestanding kernel object module
make airlys
```

---

## 📖 5. Step-by-Step Beginner's Guide: Running AI on Low-End Hardware

This guide is written specifically for Indian students and independent developers who want to run AI locally without spending money on high-end graphics cards or cloud subscriptions.

### Step 1: Install Basic Compilers (1 minute)
- **On Android (Termux):**
  ```bash
  pkg update && pkg install -y git clang make wget
  ```
- **On Ubuntu / Debian / Raspberry Pi:**
  ```bash
  sudo apt update && sudo apt install -y git build-essential clang wget
  ```
- **On Windows (PowerShell / Command Prompt):**
  Install MinGW-w64 (via MSYS2 or `winget install MSYS2`) or Visual Studio C++.

### Step 2: Clone and Compile
```bash
# Clone repository
git clone https://github.com/airagroupoftechnology/aira-core.git
cd aira-core

# Compile native binary (takes ~5 seconds)
make

# Verify that mathematical tensor engine passes all checks
make test
```

### Step 3: Download a Lightweight Model (Offline Inference)
To get started immediately without downloading gigabytes of data, you can use an educational micro-model (like Stories15M - ~60 MB):
```bash
# Download weights
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O model.bin

# Download tokenizer vocabulary
wget https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin -O tokenizer.bin
```

### Step 4: Run Real AI Inference (100% Offline, Zero Internet)
```bash
./aira-core -m model.bin -z tokenizer.bin -p "Once upon a time in India" -t 0.8 -s 256
```

### CLI Command Reference
| Flag | Description | Example |
| :--- | :--- | :--- |
| `-m` | Path to binary model weights | `-m model.bin` |
| `-z` | Path to tokenizer vocabulary | `-z tokenizer.bin` |
| `-p` | Prompt string to initialize generation | `-p "Hello Aira"` |
| `-t` | Temperature (`0.0` = greedy, `0.8` = creative) | `-t 0.7` |
| `-s` | Maximum sequence generation steps (tokens) | `-s 256` |

### Step 5: Exporting Custom Hugging Face Models
To convert your own PyTorch / Hugging Face models into the zero-copy Aira binary format:
```bash
python3 tools/export_weights.py
```

---

## 📊 6. Mathematical Precision & Invariants

All core mathematical routines are verified through `tests/test_math.c`:
- **Root Mean Square Normalization (RMSNorm)**: Numerical epsilon stability at $10^{-5}$.
- **General Matrix-Vector Multiplication (GEMM)**: Single Instruction Multiple Data vector unrolling.
- **Rotary Position Embeddings (RoPE)**: Polar complex rotation for long-context positional attention.
- **SwiGLU Activation**: $x \cdot \text{sigmoid}(x) \cdot \text{up}(x)$ non-linear projection.
- **Numerically Stable Softmax**: $\exp(x_i - \max(x))$ prevention of floating-point overflow.

---

## 📜 7. Sovereign Mission & Dedication

**🇮🇳 Built in India for Indian Students and Indian Developers**

This is a personal hobby project architected by **Er. Adam Eehan** (**Founder & CEO, Aira Group of Technology**).

It is dedicated to Indian students, budding developers, and researchers across the country and the world who have the passion to build, learn, and experiment with artificial intelligence, but lack access to expensive high-end GPUs or costly cloud infrastructure. 

With **Aira Core**, any standard laptop, phone, or bare-metal environment becomes an independent, sovereign machine capable of running modern neural inference at native speeds. Zero corporate gatekeeping, zero subscription fees—pure engineering freedom.
