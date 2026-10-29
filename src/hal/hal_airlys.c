/**
 * ============================================================================
 * AIRA CORE - HAL AIRLYS OS IMPLEMENTATION (Bare-Metal Sovereign Kernel)
 * 
 * Target: Airlys OS S1 (Freestanding Kernel Architecture)
 * Zero standard C library dependencies (no glibc, no POSIX).
 * Directly interfaces with Airlys Kernel Memory Allocator, Page Frame Tables,
 * RDTSC / CNTVCT hardware timers, and Framebuffer / VGA Console.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * Co-Pilot: Aira 3 Ultra (Sovereign Engine v3.6)
 * ============================================================================
 */

#if defined(AIRLYS_OS)

#include "hal/hal_os.h"

/* Airlys OS Kernel Syscall & Hardware Interface */
extern void* airlys_kmalloc(size_t size);
extern void  airlys_kfree(void* ptr);
extern void* airlys_map_physical(uint64_t phys_addr, size_t size);
extern void  airlys_unmap_physical(void* virt_addr, size_t size);
extern void  airlys_console_puts(const char* s);
extern void  airlys_console_putc(char c);

/* Architecture-specific cycle counter */
static inline uint64_t airlys_read_cycles(void) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ __volatile__ ("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return 0;
#endif
}

static hal_sysinfo_t s_airlys_sysinfo;
static uint64_t s_boot_cycles = 0;
static uint64_t s_cpu_freq_hz = 2000000000ULL; /* 2.0 GHz nominal base */

int hal_init(void) {
    s_airlys_sysinfo.os_name = "Airlys OS S1 Duo (Bare-Metal)";
    s_airlys_sysinfo.num_cpu_cores = 4;
    s_airlys_sysinfo.has_neon = false;
    s_airlys_sysinfo.has_avx2 = true;
    s_airlys_sysinfo.total_ram_mb = 4096;
    s_airlys_sysinfo.free_ram_mb = 3584;

    s_boot_cycles = airlys_read_cycles();
    airlys_console_puts("\n[Airlys OS] Aira Sovereign AI Inference Core Initialized.\n");
    return 0;
}

void hal_shutdown(void) {
    airlys_console_puts("[Airlys OS] Aira Core Offline.\n");
}

void hal_get_sysinfo(hal_sysinfo_t* info) {
    if (info) *info = s_airlys_sysinfo;
}

uint64_t hal_get_time_ns(void) {
    uint64_t delta_cycles = airlys_read_cycles() - s_boot_cycles;
    return (delta_cycles * 1000000000ULL) / s_cpu_freq_hz;
}

double hal_get_time_ms(void) {
    return (double)hal_get_time_ns() / 1000000.0;
}

void* hal_alloc_aligned(size_t size, size_t alignment) {
    (void)alignment;
    /* Airlys kernel page frame allocator naturally aligns to 4096-byte pages */
    return airlys_kmalloc(size);
}

void hal_free_aligned(void* ptr) {
    if (ptr) {
        airlys_kfree(ptr);
    }
}

int hal_mmap_file(const char* filepath, hal_mmap_t* out_map) {
    (void)filepath;
    if (!out_map) return -1;
    
    /* In Airlys OS Bare-Metal, the model is embedded in RAM disk or mapped at 0x10000000 */
    uint64_t model_phys_addr = 0x10000000;
    size_t   model_size = 64 * 1024 * 1024; /* 64 MB region */

    void* virt_addr = airlys_map_physical(model_phys_addr, model_size);
    if (!virt_addr) return -1;

    out_map->data = virt_addr;
    out_map->size = model_size;
    out_map->handle = (void*)model_phys_addr;
    out_map->map_handle = 0;
    return 0;
}

void hal_munmap_file(hal_mmap_t* map) {
    if (!map || !map->data) return;
    airlys_unmap_physical(map->data, map->size);
    map->data = 0;
}

void hal_print(const char* text) {
    if (text) {
        airlys_console_puts(text);
    }
}

void hal_print_token(const char* token_str) {
    if (token_str) {
        airlys_console_puts(token_str);
    }
}

void hal_flush(void) {
    /* Direct hardware write, always flushed */
}

#endif /* AIRLYS_OS */
