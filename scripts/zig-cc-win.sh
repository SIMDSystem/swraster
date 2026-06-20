#!/bin/sh
# Rust linker shim for the x86_64-pc-windows-gnullvm target: drive the link
# through `zig cc` (LLVM-mingw). rustc still emits GCC/mingw runtime -l flags
# (msvcrt/mingwex/mingw32/gcc/pthread); zig provides the CRT via -lc plus its own
# compiler-rt, so we strip those and let zig manage it. ZIG and ZIG_MCPU come
# from the Makefile recipe environment.
saved=$#
i=0
while [ $i -lt $saved ]; do
	a=$1
	shift
	case "$a" in
	-lmsvcrt | -lmingwex | -lmingw32 | -lgcc | -lgcc_eh | -lpthread | -l:libpthread.a) : ;;
	*) set -- "$@" "$a" ;;
	esac
	i=$((i + 1))
done
exec "$ZIG" cc -target x86_64-windows-gnu -mcpu="${ZIG_MCPU:-x86_64_v3}" -lc "$@"
