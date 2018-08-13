#!/bin/bash -e

ninja check-llvm-codegen-riscv
ninja check-llvm-debuginfo-riscv
ninja check-llvm-mc-riscv
ninja check-llvm-object-riscv
ninja check-llvm-transforms-simplifycfg-riscv
