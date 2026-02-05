# ShakeyBot Makefile (GNU make / MinGW)

# Toolchain (override from CLI if needed, e.g. make CXX=clang++)
CXX ?= g++

# Build mode: debug or release (make MODE=release)
MODE ?= debug

# Project layout
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

# Target naming
ifeq ($(OS),Windows_NT)
  EXE_EXT := .exe
else
  EXE_EXT :=
endif
TARGET_NAME := ShakeyBot
TARGET      := $(BIN_DIR)/$(TARGET_NAME)$(EXE_EXT)

# Flags
CPPFLAGS := $(addprefix -I,$(INC_DIRS))
CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -MMD -MP
LDFLAGS  :=
LDLIBS   :=

ifeq ($(MODE),release)
  CXXFLAGS += -O3 -DNDEBUG
else
  CXXFLAGS += -O0 -g
endif

# Linux needs pthread for std::thread linkage.
ifneq ($(OS),Windows_NT)
  CXXFLAGS += -pthread
  LDFLAGS  += -pthread
endif

# Sources
CORE_SOURCES := \
  $(SRC_DIR)/config.cpp \
  $(SRC_DIR)/engine.cpp \
  $(SRC_DIR)/evaluation.cpp \
  $(SRC_DIR)/search.cpp \
  $(SRC_DIR)/transposition.cpp

APP_SOURCES := \
  $(APP_DIR)/fast_engine_uci.cpp

SOURCES := $(CORE_SOURCES) $(APP_SOURCES)
OBJECTS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d)

.PHONY: all clean run dirs help

all: $(TARGET)

$(TARGET): $(OBJECTS) | dirs
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@
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
	@echo "  make [MODE=debug|release]   Build $(TARGET)"
	@echo "  make run                    Build and run"
	@echo "  make clean                  Remove build artifacts"

-include $(DEPS)
