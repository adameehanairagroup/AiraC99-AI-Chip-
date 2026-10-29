/**
 * ============================================================================
 * AIRA CORE - HAL LINUX IMPLEMENTATION (POSIX)
 * 
 * Target: GNU/Linux, POSIX compliant systems (x86_64, aarch64, riscv64)
 * Uses native Linux mmap, clock_gettime, and /proc/cpuinfo discovery.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * ============================================================================
 */

#if defined(__linux__) && !defined(__ANDROID__) && !defined(AIRLYS_OS)

#define _GNU_SOURCE
#include "hal/hal_os.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

static hal_sysinfo_t s_sysinfo;

int hal_init(void) {
    memset(&s_sysinfo, 0, sizeof(s_sysinfo));
    s_sysinfo.os_name = "GNU/Linux (POSIX)";
    s_sysinfo.num_cpu_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (s_sysinfo.num_cpu_cores < 1) s_sysinfo.num_cpu_cores = 1;

    /* Detect CPU features via /proc/cpuinfo */
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strstr(line, "avx2")) s_sysinfo.has_avx2 = true;
            if (strstr(line, "asimd") || strstr(line, "neon")) s_sysinfo.has_neon = true;
        }
        fclose(f);
    }

    /* RAM estimation */
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        s_sysinfo.total_ram_mb = (size_t)((pages * page_size) / (1024 * 1024));
    }
    return 0;
}

void hal_shutdown(void) {
    /* No cleanup needed for Linux POSIX HAL */
}

void hal_get_sysinfo(hal_sysinfo_t* info) {
    if (info) *info = s_sysinfo;
}

uint64_t hal_get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

double hal_get_time_ms(void) {
    return (double)hal_get_time_ns() / 1000000.0;
}

void* hal_alloc_aligned(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

void hal_free_aligned(void* ptr) {
    free(ptr);
}

int hal_mmap_file(const char* filepath, hal_mmap_t* out_map) {
    if (!filepath || !out_map) return -1;
    memset(out_map, 0, sizeof(hal_mmap_t));

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return -1;

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        close(fd);
        return -1;
    }

    size_t size = (size_t)sb.st_size;
    void* mapped = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return -1;
    }

    /* Advise OS for sequential read performance */
    #ifdef MADV_WILLNEED
    madvise(mapped, size, MADV_WILLNEED);
    #endif

    out_map->data = mapped;
    out_map->size = size;
    out_map->handle = (void*)(intptr_t)fd;
    out_map->map_handle = NULL;
    return 0;
}

void hal_munmap_file(hal_mmap_t* map) {
    if (!map || !map->data) return;
    munmap(map->data, map->size);
    if (map->handle) {
        close((int)(intptr_t)map->handle);
    }
    memset(map, 0, sizeof(hal_mmap_t));
}

void hal_print(const char* text) {
    if (text) {
        fputs(text, stdout);
    }
}

void hal_print_token(const char* token_str) {
    if (token_str) {
        fputs(token_str, stdout);
        fflush(stdout);
    }
}

void hal_flush(void) {
    fflush(stdout);
}

#endif /* __linux__ && !__ANDROID__ && !AIRLYS_OS */
