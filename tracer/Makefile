.RECIPEPREFIX = >
CLANG ?= clang
LIBBPF ?= $(firstword $(wildcard $(CURDIR)/../libbpf-install $(HOME)/libbpf-install))
LIBBPF_A ?= $(firstword $(wildcard $(LIBBPF)/lib64/libbpf.a $(LIBBPF)/lib/libbpf.a /usr/lib/x86_64-linux-gnu/libbpf.a /usr/lib64/libbpf.a /usr/lib/libbpf.a))
ifeq ($(LIBBPF_A),)
LIBBPF_LD := -lbpf
else
LIBBPF_LD := $(LIBBPF_A)
endif
INC := -I. -Ibpf $(if $(wildcard $(LIBBPF)/include),-I$(LIBBPF)/include,)
all: collector tests/test_direct_io
vmlinux.h:
> bpftool btf dump file /sys/kernel/btf/vmlinux format c > $@
trace.bpf.o: bpf/trace.bpf.c vmlinux.h bpf/common.h
> $(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_x86 $(INC) -c $< -o $@
trace.skel.h: trace.bpf.o
> bpftool gen skeleton $< > $@
collector: src/collector.c trace.skel.h bpf/common.h
> $(CC) -g -O2 -Wall $(INC) $< -o $@ $(LIBBPF_LD) -lelf -lz
tests/test_direct_io: tests/test_direct_io.c
> $(CC) -g -O2 -Wall $< -o $@
clean:
> rm -f collector trace.bpf.o trace.skel.h tests/test_direct_io
