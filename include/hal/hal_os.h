/**
 * ============================================================================
 * AIRA CORE - SOVEREIGN EDGE AI INFERENCE ENGINE
 * Hardware & Operating System Abstraction Layer (HAL)
 * 
 * Platforms Supported:
 *   - Linux (POSIX, x86_64 / aarch64 / riscv64)
 *   - Android (Termux / NDK / Bionic Libc / ARM NEON)
 *   - Windows (Win32 API / MSVC / MinGW-w64)
 *   - Airlys OS (Bare-Metal Freestanding Kernel Runtime)
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * Co-Pilot: Aira 3 Ultra (Sovereign Engine v3.6)
 * ============================================================================
 */

#ifndef AIRA_HAL_OS_H
#define AIRA_HAL_OS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(AIRLYS_OS)
    #define AIRA_PLATFORM_AIRLYS 1
    #define AIRA_PLATFORM_NAME "Airlys OS (Bare-Metal)"
#elif defined(__ANDROID__)
    #define AIRA_PLATFORM_ANDROID 1
    #define AIRA_PLATFORM_NAME "Android / Bionic"
#elif defined(_WIN32) || defined(_WIN64)
    #define AIRA_PLATFORM_WINDOWS 1
    #define AIRA_PLATFORM_NAME "Microsoft Windows (Win32)"
#elif defined(__linux__)
    #define AIRA_PLATFORM_LINUX 1
    #define AIRA_PLATFORM_NAME "GNU/Linux (POSIX)"
#else
    #define AIRA_PLATFORM_GENERIC 1
    #define AIRA_PLATFORM_NAME "Generic / Freestanding"
#endif

/* Memory mapping descriptor */
typedef struct {
    void*  data;          /* Pointer to mapped virtual address */
    size_t size;          /* Mapped size in bytes */
    void*  handle;        /* OS-specific handle (file descriptor or Win32 HANDLE) */
    void*  map_handle;    /* Win32 FileMapping handle, NULL on POSIX */
} hal_mmap_t;

/* System information */
typedef struct {
    const char* os_name;
    int         num_cpu_cores;
    bool        has_neon;
    bool        has_avx2;
    size_t      total_ram_mb;
    size_t      free_ram_mb;
} hal_sysinfo_t;

/* --- HAL Core Lifecycle & Hardware Discovery --- */
int  hal_init(void);
void hal_shutdown(void);
void hal_get_sysinfo(hal_sysinfo_t* info);

/* --- High Precision Monotonic Timer --- */
uint64_t hal_get_time_ns(void);
double   hal_get_time_ms(void);

/* --- Memory Management (Page-Aligned) --- */
void* hal_alloc_aligned(size_t size, size_t alignment);
void  hal_free_aligned(void* ptr);

/* --- Memory-Mapped File I/O (Zero-Copy Model Loading) --- */
int  hal_mmap_file(const char* filepath, hal_mmap_t* out_map);
void hal_munmap_file(hal_mmap_t* map);

/* --- Console & Terminal Output --- */
void hal_print(const char* text);
void hal_print_token(const char* token_str);
void hal_flush(void);

#endif /* AIRA_HAL_OS_H */
