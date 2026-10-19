# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved. This file is part of RabbitCT.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.

#CONFIGURE BUILD SYSTEM
TARGET	   = rabbitRunner-$(TOOLCHAIN)
BUILD_DIR  = ./build/$(TOOLCHAIN)
SRC_DIR    = ./src
MAKE_DIR   = ./mk
Q         ?= @

#DO NOT EDIT BELOW
ifeq (,$(wildcard config.mk))
$(info )
$(info ====================================================================)
$(info config.mk does not exist!)
$(info Creating config.mk from ./mk/config-default.mk)
$(info Please adapt config.mk to your needs and run make again.)
$(info ====================================================================)
$(info )
$(shell cp ./mk/config-default.mk config.mk)
$(error Stopping after creating config.mk - please review and run make again)
endif
include config.mk
include $(MAKE_DIR)/include_$(TOOLCHAIN).mk
include $(MAKE_DIR)/include_LIKWID.mk
INCLUDES  += -I$(SRC_DIR)/includes -I$(SRC_DIR) -I$(BUILD_DIR)

VPATH     = $(SRC_DIR)
ASM       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.s,$(wildcard $(SRC_DIR)/*.c))
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
SRC       =  $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/*.c)
CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(OPTIONS) $(INCLUDES)

# CUDA support
ifneq (,$(filter $(TOOLCHAIN),NVCC))
  CPPFLAGS += -D_GPU
  CU_SRC    = $(wildcard $(SRC_DIR)/*.cu)
  CU_OBJ    = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o,$(CU_SRC))
  OBJ      += $(CU_OBJ)
endif

c := ,
clist = $(subst $(eval) ,$c,$(strip $1))

define CLANGD_TEMPLATE
CompileFlags:
  Add: [$(call clist,$(CPPFLAGS)), $(call clist,$(CFLAGS)), -xc]
  Compiler: clang
endef

${TARGET}: $(BUILD_DIR) .clangd $(OBJ) $(DATA_DIR)
	$(info ===>  LINKING  $(TARGET))
	$(Q)${LD} ${LFLAGS} -o $(TARGET) $(OBJ) $(LIBS)

$(BUILD_DIR)/%.o:  %.c $(MAKE_DIR)/include_$(TOOLCHAIN).mk config.mk
	$(info ===>  COMPILE  $@)
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $< -o $@
	$(Q)$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.o: %.cu $(MAKE_DIR)/include_$(TOOLCHAIN).mk config.mk
	$(info ===>  COMPILE CUDA $@)
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $< -o $@
	$(Q)$(CC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.s:  %.c
	$(info ===>  GENERATE ASM  $@)
	$(CC) -S $(CPPFLAGS) $(CFLAGS) $< -o $@

$(BUILD_DIR)/%.o:  %.S $(MAKE_DIR)/include_$(TOOLCHAIN).mk config.mk
	$(info ===>  ASSEMBLE  $@)
	$(Q)$(CC) -c $(CPPFLAGS) $< -o $@


.PHONY: clean distclean info asm format

clean:
	$(info ===>  CLEAN)
	@rm -rf $(BUILD_DIR)

distclean:
	$(info ===>  DIST CLEAN)
	@rm -rf build
	@rm -f rabbitRunner-*
	@rm -f .clangd compile_commands.json

info:
	$(info $(CFLAGS))
	$(Q)$(CC) $(VERSION)

asm:  $(BUILD_DIR) $(ASM)

format:
	@for src in $(SRC) ; do \
		echo "Formatting $$src" ; \
		clang-format -i $$src ; \
	done
	@echo "Done"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

.clangd:
	$(file > .clangd,$(CLANGD_TEMPLATE))

-include $(OBJ:.o=.d)
