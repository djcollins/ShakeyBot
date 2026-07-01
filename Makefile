# ShakeyBot Makefile (GNU make, MinGW on Windows)

# Toolchain. Override from CLI when needed, e.g. make CXX=clang++.
CXX ?= g++

# Build mode: debug or release.
MODE ?= debug

# CPU target: auto, baseline, avx2, or avx512. Auto uses runtime dispatch.
CPU ?= auto

# Project layout.
SRC_DIR     := src
APP_DIR     := apps
INC_DIRS    := include external
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := $(BUILD_DIR)/bin

ifeq ($(OS),Windows_NT)
  # Auto-detect MinGW bin dir for runtime DLL copy.
  MINGW_BIN ?= $(dir $(shell where x86_64-w64-mingw32-g++.exe 2>NUL))
  ifeq ($(strip $(MINGW_BIN)),)
    MINGW_BIN := $(dir $(shell where g++.exe 2>NUL))
  endif
endif

# Target name.
ifeq ($(OS),Windows_NT)
  EXE_EXT := .exe
else
  EXE_EXT :=
endif
TARGET_NAME := ShakeyBot
TARGET      := $(BIN_DIR)/$(TARGET_NAME)$(EXE_EXT)
HALFKP_PREPROCESS_TARGET := $(BIN_DIR)/halfkp_preprocess$(EXE_EXT)
HALFKP_PREPROCESS_SOURCE := tools/halfkp_preprocess_cpp/main.cpp

# Compiler and linker flags.
CPPFLAGS := $(addprefix -I,$(INC_DIRS))
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -MMD -MP
LDFLAGS  :=
LDLIBS   :=

ifeq ($(MODE),release)
  CXXFLAGS += -O3 -DNDEBUG
else
  CXXFLAGS += -O0 -g
endif

ifeq ($(CPU),auto)
  CPPFLAGS += -DSHAKEYBOT_ENABLE_AVX2_DISPATCH=1 -DSHAKEYBOT_ENABLE_AVX512_DISPATCH=1
else ifeq ($(CPU),avx2)
  CXXFLAGS += -mavx2
else ifeq ($(CPU),avx512)
  CXXFLAGS += -mavx512f -mavx512bw -mavx512dq
else ifneq ($(CPU),baseline)
  $(error Unsupported CPU target '$(CPU)'. Use CPU=auto, CPU=baseline, CPU=avx2, or CPU=avx512)
endif

# Linux needs pthread for std::thread linkage.
ifneq ($(OS),Windows_NT)
  CXXFLAGS += -pthread
  LDFLAGS  += -pthread
endif

# Source files.
CORE_SOURCES := \
  $(SRC_DIR)/config.cpp \
  $(SRC_DIR)/engine.cpp \
  $(SRC_DIR)/evaluation.cpp \
  $(SRC_DIR)/fathom_tbprobe.cpp \
  $(SRC_DIR)/path_utils.cpp \
  $(SRC_DIR)/search.cpp \
  $(SRC_DIR)/tablebase.cpp \
  $(SRC_DIR)/transposition.cpp

APP_SOURCES := \
  $(APP_DIR)/fast_engine_uci.cpp

SOURCES := $(CORE_SOURCES) $(APP_SOURCES)
OBJECTS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d)

.PHONY: all clean run dirs help halfkp_preprocess

all: $(TARGET)

halfkp_preprocess: $(HALFKP_PREPROCESS_TARGET)

$(TARGET): $(OBJECTS) | dirs
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@
ifeq ($(OS),Windows_NT)
	@for %%f in (libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll) do @if exist "$(MINGW_BIN)%%f" copy /Y "$(MINGW_BIN)%%f" "$(BIN_DIR)\\" >NUL
endif

$(HALFKP_PREPROCESS_TARGET): $(HALFKP_PREPROCESS_SOURCE) | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
ifeq ($(OS),Windows_NT)
	@for %%f in (libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll) do @if exist "$(MINGW_BIN)%%f" copy /Y "$(MINGW_BIN)%%f" "$(BIN_DIR)\\" >NUL
endif

$(OBJ_DIR)/%.o: %.cpp | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

ifeq ($(OS),Windows_NT)
dirs:
	@if not exist "$(OBJ_DIR)\\src" mkdir "$(OBJ_DIR)\\src"
	@if not exist "$(OBJ_DIR)\\apps" mkdir "$(OBJ_DIR)\\apps"
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"

clean:
	@if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)"
else
dirs:
	@mkdir -p $(OBJ_DIR)/src $(OBJ_DIR)/apps $(BIN_DIR)

clean:
	@rm -rf $(BUILD_DIR)
endif

run: $(TARGET)
	$(TARGET)

help:
	@echo "Targets:"
	@echo "  make MODE=release              Build release binary with safe CPU dispatch"
	@echo "  make MODE=release CPU=baseline Build portable baseline release binary"
	@echo "  make MODE=release CPU=avx2     Build AVX2-only release binary"
	@echo "  make MODE=release CPU=avx512   Build AVX512-only release binary"
	@echo "  make halfkp_preprocess MODE=release Build C++ HalfKP preprocessing tool"
	@echo "  make MODE=debug                Build debug binary"
	@echo "  make run                       Build and run binary"
	@echo "  make clean                     Remove build artifacts"

-include $(DEPS)
