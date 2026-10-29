/**
 * ============================================================================
 * AIRA CORE - HAL ANDROID IMPLEMENTATION (Bionic / Termux / NDK)
 * 
 * Target: Android OS (Termux, Bionic libc, ARM64 / ARMv7)
 * Optimized for ARM NEON SIMD, mobile thermal limits, and low RAM footprints.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * ============================================================================
 */

#if defined(__ANDROID__)

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
#include <sys/auxv.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static hal_sysinfo_t s_android_sysinfo;

int hal_init(void) {
    memset(&s_android_sysinfo, 0, sizeof(s_android_sysinfo));
    s_android_sysinfo.os_name = "Android (Bionic / Termux)";
    s_android_sysinfo.num_cpu_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (s_android_sysinfo.num_cpu_cores < 1) s_android_sysinfo.num_cpu_cores = 1;

    /* On ARM64 Android, NEON / ASIMD is mandatory in the ARMv8-A architecture */
    #if defined(__aarch64__)
    s_android_sysinfo.has_neon = true;
    #else
    /* Check via AT_HWCAP on 32-bit ARM */
    unsigned long hwcap = getauxval(AT_HWCAP);
    #ifdef HWCAP_ARM_NEON
    if (hwcap & HWCAP_ARM_NEON) s_android_sysinfo.has_neon = true;
    #endif
    #endif

    /* Detect RAM capacity from sysconf */
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        s_android_sysinfo.total_ram_mb = (size_t)((pages * page_size) / (1024 * 1024));
    }
    return 0;
}

void hal_shutdown(void) {
    /* Cleanup */
}

void hal_get_sysinfo(hal_sysinfo_t* info) {
    if (info) *info = s_android_sysinfo;
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
    /* MAP_SHARED enables Android kernel page cache sharing without RAM duplication */
    void* mapped = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return -1;
    }

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

#endif /* __ANDROID__ */
