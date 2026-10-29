# ============================================================================
# AIRA CORE - MULTI-PLATFORM BUILD AUTOMATION SYSTEM
# 
# Targets:
#   - native (Linux / Android Termux)
#   - test (Verification Suite)
#   - windows (Win32 target)
#   - airlys (Airlys OS Freestanding Bare-Metal kernel module)
# 
# Architect: Er. Adam Eehan (Founder & CEO, Aira Group of Technology)
# ============================================================================

CC ?= clang
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -Iinclude -funroll-loops
LDFLAGS ?= -lm

# Detect Platform
UNAME_S := $(shell uname -s)
ifeq ($(findstring Android, $(shell uname -a)), Android)
    PLATFORM := android
    HAL_SRC := src/hal/hal_android.c
    CFLAGS += -march=armv8-a+simd
else ifeq ($(UNAME_S), Linux)
    PLATFORM := linux
    HAL_SRC := src/hal/hal_linux.c
    CFLAGS += -march=native
else ifeq ($(findstring MINGW, $(UNAME_S)), MINGW)
    PLATFORM := windows
    HAL_SRC := src/hal/hal_windows.c
else
    PLATFORM := generic
    HAL_SRC := src/hal/hal_linux.c
endif

SRCS := src/aira_core.c src/aira_math.c src/aira_tokenizer.c $(HAL_SRC)
OBJS := $(SRCS:.c=.o)
TARGET := aira-core

.PHONY: all clean test windows airlys help

all: $(TARGET)
	@echo "=========================================================="
	@echo " [SUCCESS] Aira Core Built for Platform: $(PLATFORM)"
	@echo " Binary: ./$(TARGET)"
	@echo "=========================================================="

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: tests/test_math.o src/aira_math.o $(HAL_SRC)
	$(CC) $(CFLAGS) $^ -o test_math $(LDFLAGS)
	./test_math

windows:
	x86_64-w64-mingw32-gcc -O3 -std=c99 -Iinclude -D_WIN32 \
		src/aira_core.c src/aira_math.c src/aira_tokenizer.c src/hal/hal_windows.c \
		-o aira-core.exe -lm

airlys:
	$(CC) -O3 -std=c99 -Iinclude -DAIRLYS_OS -ffreestanding -nostdlib -fno-builtin \
		-c src/aira_core.c src/aira_math.c src/hal/hal_airlys.c
	@echo "Airlys OS Bare-Metal Kernel Object files generated!"

clean:
	rm -f src/*.o src/hal/*.o tests/*.o $(TARGET) test_math aira-core.exe
