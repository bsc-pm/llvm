; RUN: llc -mtriple riscv32 -relocation-model=pic < %s \
; RUN: | FileCheck --check-prefix=PIC %s
; RUN: llc -mtriple riscv32 < %s \
; RUN: | FileCheck --check-prefix=NOPIC %s

; PIC: .option pic
; NOPIC: .option nopic
