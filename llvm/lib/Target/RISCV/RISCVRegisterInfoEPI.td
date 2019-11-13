//===-- RISCVRegisterInfo.td - RISC-V EPI Register defs ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Represents the current vector length
def VL : RISCVReg<0, "vl">;
def VLR : RegisterClass<"RISCV", [XLenVT], 32, (add VL)> {
  let CopyCost = -1; // Don't allow copying
  let isAllocatable = 0; // Not allocatable
  let RegInfos = RegInfoByHwMode<
      [RV32,              RV64,              DefaultMode],
      [RegInfo<32,32,32>, RegInfo<64,64,64>, RegInfo<32,32,32>]>;
}

// Represents the current vector type. This has been decoupled because
// some instructions are not polymorphic so they don't use VTYPE.
def VTYPE : RISCVReg<0, "vtype">;
def VTypeR : RegisterClass<"RISCV", [XLenVT], 32, (add VTYPE)> {
  let CopyCost = -1; // Don't allow copying
  let isAllocatable = 0; // Not allocatable
  let RegInfos = RegInfoByHwMode<
      [RV32,              RV64,              DefaultMode],
      [RegInfo<32,32,32>, RegInfo<64,64,64>, RegInfo<32,32,32>]>;
}
