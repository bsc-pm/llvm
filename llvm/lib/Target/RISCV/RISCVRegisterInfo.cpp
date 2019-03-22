//===-- RISCVRegisterInfo.cpp - RISCV Register Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVRegisterInfo.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_REGINFO_TARGET_DESC
#include "RISCVGenRegisterInfo.inc"

using namespace llvm;

RISCVRegisterInfo::RISCVRegisterInfo(unsigned HwMode)
    : RISCVGenRegisterInfo(RISCV::X1, /*DwarfFlavour*/0, /*EHFlavor*/0,
                           /*PC*/0, HwMode) {}

const MCPhysReg *
RISCVRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  auto &Subtarget = MF->getSubtarget<RISCVSubtarget>();
  if (MF->getFunction().hasFnAttribute("interrupt")) {
    if (Subtarget.hasStdExtD())
      return CSR_XLEN_F64_Interrupt_SaveList;
    if (Subtarget.hasStdExtF())
      return CSR_XLEN_F32_Interrupt_SaveList;
    return CSR_Interrupt_SaveList;
  }

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    return CSR_ILP32_LP64_SaveList;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_SaveList;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    if (Subtarget.hasExtEPI())
      return CSR_ILP32D_LP64D_EPI_SaveList;
    return CSR_ILP32D_LP64D_SaveList;
  }
}

BitVector RISCVRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  BitVector Reserved(getNumRegs());


  // Use markSuperRegs to ensure any register aliases are also reserved
  markSuperRegs(Reserved, RISCV::X0); // zero
  markSuperRegs(Reserved, RISCV::X1); // ra
  markSuperRegs(Reserved, RISCV::X2); // sp
  markSuperRegs(Reserved, RISCV::X3); // gp
  markSuperRegs(Reserved, RISCV::X4); // tp
  if (TFI->hasFP(MF))
    markSuperRegs(Reserved, RISCV::X8); // fp
  if (hasBasePointer(MF))
    markSuperRegs(Reserved, RISCV::X9); // bp

  // EPI registers
  markSuperRegs(Reserved, RISCV::VL);
  markSuperRegs(Reserved, RISCV::VTYPE);

  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool RISCVRegisterInfo::isConstantPhysReg(unsigned PhysReg) const {
  return PhysReg == RISCV::X0;
}

const uint32_t *RISCVRegisterInfo::getNoPreservedMask() const {
  return CSR_NoRegs_RegMask;
}

void RISCVRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                            int SPAdj, unsigned FIOperandNum,
                                            RegScavenger *RS) const {
  assert(SPAdj == 0 && "Unexpected non-zero SPAdj value");

  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVInstrInfo *TII = MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  unsigned FrameReg;
  int Offset =
      getFrameLowering(MF)->getFrameIndexReference(MF, FrameIndex, FrameReg);

  bool OffsetFits = false;
  int OffsetIndex = -1;
  // FIXME: Improve this to make it more robust.
  switch (MI.getOpcode()) {
  case RISCV::VLE_V:
  case RISCV::VSE_V:
    // These two are handled later in this function
  case RISCV::PseudoVSPILL:
  case RISCV::PseudoVRELOAD:
    break;
  default:
    OffsetIndex = FIOperandNum + 1;
    Offset += MI.getOperand(OffsetIndex).getImm();
    OffsetFits = isInt<12>(Offset);
    break;
  }

  if (!isInt<32>(Offset)) {
    report_fatal_error(
        "Frame offsets outside of the signed 32-bit range not supported");
  }

  MachineBasicBlock &MBB = *MI.getParent();
  bool FrameRegIsKill = false;

  if (!OffsetFits) {
    assert(isInt<32>(Offset) && "Int32 expected");
    // The offset won't fit in an immediate, so use a scratch register instead
    // Modify Offset and FrameReg appropriately
    unsigned ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    TII->movImm32(MBB, II, DL, ScratchReg, Offset);
    BuildMI(MBB, II, DL, TII->get(RISCV::ADD), ScratchReg)
        .addReg(FrameReg)
        .addReg(ScratchReg, RegState::Kill);
    Offset = 0;
    FrameReg = ScratchReg;
    FrameRegIsKill = true;
  }

  MI.getOperand(FIOperandNum)
      .ChangeToRegister(FrameReg, false, false, FrameRegIsKill);
  if (OffsetIndex >= 0) {
    MI.getOperand(OffsetIndex).ChangeToImmediate(Offset);
  }

  // Handle vector spills here
  if (MI.getOpcode() == RISCV::PseudoVSPILL ||
      MI.getOpcode() == RISCV::PseudoVRELOAD) {
    unsigned SlotAddrReg = MI.getOperand(1).getReg();

    // Save VTYPE
    unsigned OldVTypeReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, II, DL, TII->get(RISCV::CSRRS), OldVTypeReg)
        .addImm(EPICSR::VTYPE)
        .addReg(RISCV::X0);

    // TODO: Consider using loadRegFromStackSlot but this has to be before
    // replacing the FI above.
    unsigned LoadHandleOpcode =
        getRegSizeInBits(RISCV::GPRRegClass) == 32 ? RISCV::LW : RISCV::LD;
    unsigned HandleReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    BuildMI(MBB, II, DL, TII->get(LoadHandleOpcode), HandleReg)
        .addReg(SlotAddrReg)
        .addImm(0);

    // Make sure we spill/reload all the bits
    BuildMI(MBB, II, DL, TII->get(RISCV::VSETVLI), RISCV::X0)
        .addReg(RISCV::X0)
        // FIXME - Hardcoded to SEW=64
        .addImm(3)
        // VLMUL=1
        .addImm(0);

    MachineOperand &OpReg = MI.getOperand(0);
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected instruction");
    case RISCV::PseudoVSPILL: {
      BuildMI(MBB, II, DL, TII->get(RISCV::VSE_V))
          .addReg(OpReg.getReg(), getKillRegState(OpReg.isKill()))
          .addReg(HandleReg);
      break;
    }
    case RISCV::PseudoVRELOAD: {
      BuildMI(MBB, II, DL, TII->get(RISCV::VLE_V), OpReg.getReg())
          .addReg(HandleReg);
      break;
    }
    }

    // Restore VTYPE
    BuildMI(MBB, II, DL, TII->get(RISCV::CSRRW), RISCV::X0)
        .addImm(EPICSR::VTYPE)
        .addReg(OldVTypeReg);

    // Remove the pseudo
    MI.eraseFromParent();
  }
}

unsigned RISCVRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = getFrameLowering(MF);
  return TFI->hasFP(MF) ? RISCV::X8 : RISCV::X2;
}

const uint32_t *
RISCVRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                        CallingConv::ID /*CC*/) const {
  const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();
  if (MF.getFunction().hasFnAttribute("interrupt")) {
    if (Subtarget.hasStdExtD())
      return CSR_XLEN_F64_Interrupt_RegMask;
    if (Subtarget.hasStdExtF())
      return CSR_XLEN_F32_Interrupt_RegMask;
    return CSR_Interrupt_RegMask;
  }

  switch (Subtarget.getTargetABI()) {
  default:
    llvm_unreachable("Unrecognized ABI");
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    return CSR_ILP32_LP64_RegMask;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    return CSR_ILP32F_LP64F_RegMask;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    if (Subtarget.hasExtEPI())
      return CSR_ILP32D_LP64D_EPI_RegMask;
    return CSR_ILP32D_LP64D_RegMask;
  }
}

bool RISCVRegisterInfo::hasBasePointer(const MachineFunction &MF) const {
  // We use a BP when all of the following are true:
  // - the stack needs realignment (due to overaligned local objects)
  // - the stack has VLAs
  // Note that when we need a BP the conditions also imply a FP.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return needsStackRealignment(MF) &&
         (MFI.hasVarSizedObjects() || MFI.hasDynamicSpillObjects());
}

const TargetRegisterClass *
RISCVRegisterInfo::getPointerRegClass(const MachineFunction &MF,
                                        unsigned Kind) const {
  return &RISCV::GPRRegClass;
}
