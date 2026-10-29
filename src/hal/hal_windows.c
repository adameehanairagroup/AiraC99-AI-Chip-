/**
 * ============================================================================
 * AIRA CORE - HAL WINDOWS IMPLEMENTATION (Win32 API)
 * 
 * Target: Microsoft Windows (MSVC, MinGW-w64, Clang-cl)
 * Uses native Win32 Memory Mapped Files (CreateFileMapping, MapViewOfFile)
 * and QueryPerformanceCounter for sub-microsecond timing.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * ============================================================================
 */

#if defined(_WIN32) || defined(_WIN64)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "hal/hal_os.h"

static hal_sysinfo_t s_win_sysinfo;
static LARGE_INTEGER s_timer_freq;

int hal_init(void) {
    memset(&s_win_sysinfo, 0, sizeof(s_win_sysinfo));
    s_win_sysinfo.os_name = "Microsoft Windows (Win32)";

    /* Ensure UTF-8 output encoding for emojis and multi-byte tokens */
    SetConsoleOutputCP(CP_UTF8);

    /* Get CPU Core Count */
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    s_win_sysinfo.num_cpu_cores = (int)sys_info.dwNumberOfProcessors;
    if (s_win_sysinfo.num_cpu_cores < 1) s_win_sysinfo.num_cpu_cores = 1;

    /* Get Total RAM */
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        s_win_sysinfo.total_ram_mb = (size_t)(mem_status.ullTotalPhys / (1024 * 1024));
        s_win_sysinfo.free_ram_mb = (size_t)(mem_status.ullAvailPhys / (1024 * 1024));
    }

    /* Check AVX2 feature on x86/x64 */
    #if defined(_M_X64) || defined(__x86_64__)
    s_win_sysinfo.has_avx2 = (IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0);
    #endif

    /* Check ARM64 Neon if running on Windows on ARM */
    #if defined(_M_ARM64) || defined(__aarch64__)
    s_win_sysinfo.has_neon = (IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE) != 0);
    #endif

    /* Setup high-resolution performance counter */
    QueryPerformanceFrequency(&s_timer_freq);
    return 0;
}

void hal_shutdown(void) {
    /* Cleanup */
}

void hal_get_sysinfo(hal_sysinfo_t* info) {
    if (info) *info = s_win_sysinfo;
}

uint64_t hal_get_time_ns(void) {
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    if (s_timer_freq.QuadPart == 0) return 0;
    return (uint64_t)((count.QuadPart * 1000000000ULL) / s_timer_freq.QuadPart);
}

double hal_get_time_ms(void) {
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    if (s_timer_freq.QuadPart == 0) return 0.0;
    return ((double)count.QuadPart * 1000.0) / (double)s_timer_freq.QuadPart;
}

void* hal_alloc_aligned(size_t size, size_t alignment) {
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    #if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, alignment);
    #else
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    #endif
}

void hal_free_aligned(void* ptr) {
    if (!ptr) return;
    #if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
    #else
    VirtualFree(ptr, 0, MEM_RELEASE);
    #endif
}

int hal_mmap_file(const char* filepath, hal_mmap_t* out_map) {
    if (!filepath || !out_map) return -1;
    memset(out_map, 0, sizeof(hal_mmap_t));

    HANDLE file_handle = CreateFileA(
        filepath,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (file_handle == INVALID_HANDLE_VALUE) {
        return -1;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        CloseHandle(file_handle);
        return -1;
    }

    HANDLE map_handle = CreateFileMappingA(
        file_handle,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (!map_handle) {
        CloseHandle(file_handle);
        return -1;
    }

    void* mapped_view = MapViewOfFile(
        map_handle,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (!mapped_view) {
        CloseHandle(map_handle);
        CloseHandle(file_handle);
        return -1;
    }

    out_map->data = mapped_view;
    out_map->size = (size_t)file_size.QuadPart;
    out_map->handle = (void*)file_handle;
    out_map->map_handle = (void*)map_handle;
    return 0;
}

void hal_munmap_file(hal_mmap_t* map) {
    if (!map || !map->data) return;
    UnmapViewOfFile(map->data);
    if (map->map_handle) {
        CloseHandle((HANDLE)map->map_handle);
    }
    if (map->handle) {
        CloseHandle((HANDLE)map->handle);
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

#endif /* _WIN32 || _WIN64 */
