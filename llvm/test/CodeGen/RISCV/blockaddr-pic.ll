; RUN: llc -mtriple riscv32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32-NOPIC
; RUN: llc -mtriple riscv64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64-NOPIC
; RUN: llc -mtriple riscv32 -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV32-PIC
; RUN: llc -mtriple riscv64 -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s --check-prefix=RV64-PIC

define dso_local signext i32 @foo(i32 signext %w) nounwind {
; RV32-NOPIC-LABEL: foo:
; RV32-NOPIC:    lui a1, %hi(.Ltmp0)
; RV32-NOPIC:    addi a1, a1, %lo(.Ltmp0)
; RV32-NOPIC:    sw a1, 8(sp)
; RV32-NOPIC:    addi a1, zero, 101
; RV32-NOPIC:    blt a0, a1,
; RV32-NOPIC:    lw a0, 8(sp)
; RV32-NOPIC:    jr a0
; RV32-NOPIC:  .Ltmp0:
;
; RV64-NOPIC-LABEL: foo:
; RV64-NOPIC:    lui a1, %hi(.Ltmp0)
; RV64-NOPIC:    addi a1, a1, %lo(.Ltmp0)
; RV64-NOPIC:    sd a1, 0(sp)
; RV64-NOPIC:    addi  a1, zero, 101
; RV64-NOPIC:    blt a0, a1,
; RV64-NOPIC:    ld a0, 0(sp)
; RV64-NOPIC:    jr a0
; RV64-NOPIC:  .Ltmp0:
;
; RV32-PIC-LABEL: foo:
; RV32-PIC:    addi sp, sp, -16
; RV32-PIC:    sw ra, 12(sp)
; RV32-PIC:    lla a1, .Ltmp0
; RV32-PIC:    sw a1, 8(sp)
; RV32-PIC:    addi a1, zero, 101
; RV32-PIC:    blt a0, a1,
; RV32-PIC:    lw a0, 8(sp)
; RV32-PIC:    jr a0
; RV32-PIC:  .Ltmp0:
;
; RV64-PIC-LABEL: foo:
; RV64-PIC:    addi sp, sp, -16
; RV64-PIC:    sd ra, 8(sp)
; RV64-PIC:    lla a1, .Ltmp0
; RV64-PIC:    sd a1, 0(sp)
; RV64-PIC:    addi a1, zero, 101
; RV64-PIC:    blt  a0, a1
; RV64-PIC:    ld a0, 0(sp)
; RV64-PIC:    jr a0
; RV64-PIC:  .Ltmp0:

entry:
  %x = alloca i8*, align 8
  store i8* blockaddress(@foo, %test_block), i8** %x, align 8
  %cmp = icmp sgt i32 %w, 100
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %addr = load i8*, i8** %x, align 8
  br label %indirectgoto

if.end:
  br label %return

test_block:
  br label %return

return:
  %retval = phi i32 [ 3, %if.end ], [ 4, %test_block ]
  ret i32 %retval

indirectgoto:
  %indirect.goto.dest = phi i8* [ %addr, %if.then ]
  indirectbr i8* %addr, [ label %test_block ]
}

