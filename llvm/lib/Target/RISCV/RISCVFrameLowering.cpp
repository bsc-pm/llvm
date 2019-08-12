//===-- RISCVFrameLowering.cpp - RISCV Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "RISCVFrameLowering.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MCDwarf.h"

using namespace llvm;

bool RISCVFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  // We use a FP when any of the following is true:
  // - we are told to have one
  // - the stack needs realignment (due to overaligned local objects)
  // - the stack has VLAs
  // - the function uses @llvm.frameaddress
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         RegInfo->needsStackRealignment(MF) || MFI.hasVarSizedObjects() ||
         MFI.isFrameAddressTaken();
}

// Determines the size of the frame and maximum call frame size.
void RISCVFrameLowering::determineFrameLayout(MachineFunction &MF) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t FrameSize = MFI.getStackSize();

  // Get the alignment.
  uint64_t StackAlign = RI->needsStackRealignment(MF) ? MFI.getMaxAlignment()
                                                      : getStackAlignment();
#if 0
  // Get the alignment.
  unsigned StackAlign = getStackAlignment();
  if (RI->needsStackRealignment(MF)) {
    unsigned MaxStackAlign = std::max(StackAlign, MFI.getMaxAlignment());
    FrameSize += (MaxStackAlign - StackAlign);
    StackAlign = MaxStackAlign;
  }

  // Set Max Call Frame Size
  uint64_t MaxCallSize = alignTo(MFI.getMaxCallFrameSize(), StackAlign);
  MFI.setMaxCallFrameSize(MaxCallSize);
#endif

// rferrer: This seems not used at the moment and makes a test fail
//          because the stack is overaligned.
#if 0
  // Get the maximum call frame size of all the calls.
  uint64_t MaxCallFrameSize = MFI.getMaxCallFrameSize();

  // If we have dynamic alloca then MaxCallFrameSize needs to be aligned so
  // that allocations will be aligned.
  if (MFI.hasVarSizedObjects())
    MaxCallFrameSize = alignTo(MaxCallFrameSize, StackAlign);

  // Update maximum call frame size.
  MFI.setMaxCallFrameSize(MaxCallFrameSize);

  // Include call frame size in total.
  if (!(hasReservedCallFrame(MF) && MFI.adjustsStack()))
    FrameSize += MaxCallFrameSize;
#endif

  // Make sure the frame is aligned.
  FrameSize = alignTo(FrameSize, StackAlign);

  // Update frame info.
  MFI.setStackSize(FrameSize);
}

void RISCVFrameLowering::adjustReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL, unsigned DestReg,
                                   unsigned SrcReg, int64_t Val,
                                   MachineInstr::MIFlag Flag) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const RISCVInstrInfo *TII = STI.getInstrInfo();

  if (DestReg == SrcReg && Val == 0)
    return;

  if (isInt<12>(Val)) {
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DestReg)
        .addReg(SrcReg)
        .addImm(Val)
        .setMIFlag(Flag);
  } else if (isInt<32>(Val)) {
    unsigned Opc = RISCV::ADD;
    bool isSub = Val < 0;
    if (isSub) {
      Val = -Val;
      Opc = RISCV::SUB;
    }

    Register ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    TII->movImm32(MBB, MBBI, DL, ScratchReg, Val, Flag);
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
        .addReg(SrcReg)
        .addReg(ScratchReg, RegState::Kill)
        .setMIFlag(Flag);
  } else {
    report_fatal_error("adjustReg cannot yet handle adjustments >32 bits");
  }
}

// Returns the register used to hold the frame pointer.
static unsigned getFPReg(const RISCVSubtarget &STI) { return RISCV::X8; }

// Returns the register used to hold the base pointer.
static unsigned getBPReg(const RISCVSubtarget &STI) { return RISCV::X9; }

// Returns the register used to hold the stack pointer.
static unsigned getSPReg(const RISCVSubtarget &STI) { return RISCV::X2; }

void RISCVFrameLowering::alignSP(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 const DebugLoc &DL, int64_t Alignment) const {
  assert(isPowerOf2_64(Alignment) && "Alignment must be a power of 2");

  const RISCVInstrInfo *TII = STI.getInstrInfo();

  if (isInt<12>(Alignment)) {
    //  ANDI SP, SP, -Alignment
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::ANDI), getSPReg(STI))
        .addReg(getSPReg(STI))
        .addImm(-Alignment)
        .setMIFlag(MachineInstr::FrameSetup);
  } else if (isInt<32>(Alignment)) {
    // Use shifts to avoid using a virtual register
    // SRLI SP, SP, Log2(Alignment)
    // SLLI SP, SP, Log2(Alignment)
    uint64_t Log2Alignment = Log2_64(Alignment);
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::SRLI), getSPReg(STI))
        .addReg(getSPReg(STI))
        .addImm(Log2Alignment)
        .setMIFlag(MachineInstr::FrameSetup);
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::SLLI), getSPReg(STI))
        .addReg(getSPReg(STI))
        .addImm(Log2Alignment)
        .setMIFlag(MachineInstr::FrameSetup);
  } else {
    report_fatal_error(
        "adjustReg cannot yet handle stack realignment >32 bits");
  }
}

void RISCVFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();
  const RISCVInstrInfo *TII = STI.getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  const RISCVRegisterInfo *RegInfo =
      MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();
  bool NeedsStackRealignment = RegInfo->needsStackRealignment(MF);

  unsigned FPReg = getFPReg(STI);
  unsigned SPReg = getSPReg(STI);
  unsigned BPReg = getBPReg(STI);

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  // Determine the correct frame layout
  determineFrameLayout(MF);

  // FIXME (note copied from Lanai): This appears to be overallocating.  Needs
  // investigation. Get the number of bytes to allocate from the FrameInfo.
  uint64_t StackSize = MFI.getStackSize();

  // Early exit if there is no need to allocate on the stack
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  // Allocate space on the stack if necessary.
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, -StackSize, MachineInstr::FrameSetup);

  // Emit ".cfi_def_cfa_offset StackSize"
  unsigned CFIIndex = MF.addFrameInst(
      MCCFIInstruction::createDefCfaOffset(nullptr, -StackSize));
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlag(MachineInstr::FrameSetup);

  // The frame pointer is callee-saved, and code has been generated for us to
  // save it to the stack. We need to skip over the storing of callee-saved
  // registers as the frame pointer must be modified after it has been saved
  // to the stack, not before.
  // FIXME: assumes exactly one instruction is used to save each callee-saved
  // register.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  std::advance(MBBI, CSI.size());

  // Iterate over list of callee-saved registers and emit .cfi_offset
  // directives.
  for (const auto &Entry : CSI) {
    int64_t Offset = MFI.getObjectOffset(Entry.getFrameIdx());
    unsigned Reg = Entry.getReg();
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
        nullptr, RI->getDwarfRegNum(Reg, true), Offset));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // Generate new FP.
  if (hasFP(MF)) {
    adjustReg(MBB, MBBI, DL, FPReg, SPReg,
              StackSize - RVFI->getVarArgsSaveSize(), MachineInstr::FrameSetup);

    // Emit ".cfi_def_cfa $fp, 0"
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createDefCfa(
        nullptr, RI->getDwarfRegNum(FPReg, true), 0));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  if (NeedsStackRealignment) {
    // Realign the stack now.
    alignSP(MBB, MBBI, DL, MFI.getMaxAlignment());

    assert(hasFP(MF) && "we need an FP to properly realign the stack");

    if (RegInfo->hasBasePointer(MF)) {
      // Set BP to be the current SP
      adjustReg(MBB, MBBI, DL, BPReg, SPReg, 0, MachineInstr::FrameSetup);
    }
  }
}

void RISCVFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  DebugLoc DL = MBBI->getDebugLoc();
  const RISCVInstrInfo *TII = STI.getInstrInfo();
  unsigned FPReg = getFPReg(STI);
  unsigned SPReg = getSPReg(STI);

  // Skip to before the restores of callee-saved registers
  // FIXME: assumes exactly one instruction is used to restore each
  // callee-saved register.
  auto LastFrameDestroy = std::prev(MBBI, MFI.getCalleeSavedInfo().size());

  uint64_t StackSize = MFI.getStackSize();
  uint64_t FPOffset = StackSize - RVFI->getVarArgsSaveSize();

  // Restore the stack pointer using the value of the frame pointer. Only
  // necessary if the stack pointer was modified, meaning the stack size is
  // unknown.
  if (RI->needsStackRealignment(MF) || MFI.hasVarSizedObjects()) {
    assert(hasFP(MF) && "frame pointer should not have been eliminated");
    adjustReg(MBB, LastFrameDestroy, DL, SPReg, FPReg, -FPOffset,
              MachineInstr::FrameDestroy);
  }

  if (hasFP(MF)) {
    // To find the instruction restoring FP from stack.
    for (auto &I = LastFrameDestroy; I != MBBI; ++I) {
      if (I->mayLoad() && I->getOperand(0).isReg()) {
        Register DestReg = I->getOperand(0).getReg();
        if (DestReg == FPReg) {
          // If there is frame pointer, after restoring $fp registers, we
          // need adjust CFA to ($sp - FPOffset).
          // Emit ".cfi_def_cfa $sp, -FPOffset"
          unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createDefCfa(
              nullptr, RI->getDwarfRegNum(SPReg, true), -FPOffset));
          BuildMI(MBB, std::next(I), DL,
                  TII->get(TargetOpcode::CFI_INSTRUCTION))
              .addCFIIndex(CFIIndex)
              .setMIFlag(MachineInstr::FrameSetup);
          break;
        }
      }
    }
  }

  // Add CFI directives for callee-saved registers.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  // Iterate over list of callee-saved registers and emit .cfi_restore
  // directives.
  for (const auto &Entry : CSI) {
    unsigned Reg = Entry.getReg();
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createRestore(
        nullptr, RI->getDwarfRegNum(Reg, true)));
    BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // Deallocate stack
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, StackSize, MachineInstr::FrameDestroy);

  // After restoring $sp, we need to adjust CFA to $(sp + 0)
  // Emit ".cfi_def_cfa_offset 0"
  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::createDefCfaOffset(nullptr, 0));
  BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex)
      .setMIFlag(MachineInstr::FrameSetup);
}

int RISCVFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                               int FI,
                                               unsigned &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const RISCVRegisterInfo *RI =
      MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

  // Callee-saved registers should be referenced relative to the stack pointer
  // (positive offset), otherwise use the frame pointer (negative offset)
  // unless the offset from FP is not known at constant time, as it happens
  // when we have to align the stack or we have variably sized data.
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  int Offset = MFI.getObjectOffset(FI) - getOffsetOfLocalArea() +
               MFI.getOffsetAdjustment();

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }
  bool IsCSR = FI >= MinCSFI && FI <= MaxCSFI;
  bool isFixed = !IsCSR && MFI.isFixedObjectIndex(FI);

  if (IsCSR) {
    FrameReg = getSPReg(STI);
    Offset += MF.getFrameInfo().getStackSize();
  } else {
    if (hasFP(MF) && (isFixed || (!RI->needsStackRealignment(MF) &&
                                  !MFI.isVariableSizedObjectIndex(FI)))) {
      FrameReg = getFPReg(STI);
      Offset += RVFI->getVarArgsSaveSize();
    } else {
      FrameReg = RI->hasBasePointer(MF) ? getBPReg(STI) : getSPReg(STI);
      Offset += MF.getFrameInfo().getStackSize();
    }
  }
  return Offset;
}

void RISCVFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                              BitVector &SavedRegs,
                                              RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const RISCVRegisterInfo *RI =
      MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();
  // Unconditionally spill RA and FP only if the function uses a frame
  // pointer.
  if (hasFP(MF)) {
    SavedRegs.set(RISCV::X1);
    SavedRegs.set(getFPReg(STI));
  }
  if (RI->hasBasePointer(MF)) {
    SavedRegs.set(getBPReg(STI));
  }

  // If interrupt is enabled and there are calls in the handler,
  // unconditionally save all Caller-saved registers and
  // all FP registers, regardless whether they are used.
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (MF.getFunction().hasFnAttribute("interrupt") && MFI.hasCalls()) {

    static const MCPhysReg CSRegs[] = { RISCV::X1,      /* ra */
      RISCV::X5, RISCV::X6, RISCV::X7,                  /* t0-t2 */
      RISCV::X10, RISCV::X11,                           /* a0-a1, a2-a7 */
      RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17,
      RISCV::X28, RISCV::X29, RISCV::X30, RISCV::X31, 0 /* t3-t6 */
    };

    for (unsigned i = 0; CSRegs[i]; ++i)
      SavedRegs.set(CSRegs[i]);

    if (MF.getSubtarget<RISCVSubtarget>().hasStdExtD() ||
        MF.getSubtarget<RISCVSubtarget>().hasStdExtF()) {

      // If interrupt is enabled, this list contains all FP registers.
      const MCPhysReg * Regs = MF.getRegInfo().getCalleeSavedRegs();

      for (unsigned i = 0; Regs[i]; ++i)
        if (RISCV::FPR32RegClass.contains(Regs[i]) ||
            RISCV::FPR64RegClass.contains(Regs[i]))
          SavedRegs.set(Regs[i]);
    }
  }
}

void RISCVFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterClass *RC = &RISCV::GPRRegClass;
  // estimateStackSize has been observed to under-estimate the final stack
  // size, so give ourselves wiggle-room by checking for stack size
  // representable an 11-bit signed field rather than 12-bits.
  // FIXME: It may be possible to craft a function with a small stack that
  // still needs an emergency spill slot for branch relaxation. This case
  // would currently be missed.
  if (!isInt<11>(MFI.estimateStackSize(MF))) {
    int RegScavFI = MFI.CreateStackObject(
        RegInfo->getSpillSize(*RC), RegInfo->getSpillAlignment(*RC), false);
    RS->addScavengingFrameIndex(RegScavFI);
  }
}

// Not preserve stack space within prologue for outgoing variables when the
// function contains variable size objects and let eliminateCallFramePseudoInstr
// preserve stack space for it.
bool RISCVFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo().hasVarSizedObjects();
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions.
MachineBasicBlock::iterator RISCVFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  unsigned SPReg = RISCV::X2;
  DebugLoc DL = MI->getDebugLoc();

  if (!hasReservedCallFrame(MF)) {
    // If space has not been reserved for a call frame, ADJCALLSTACKDOWN and
    // ADJCALLSTACKUP must be converted to instructions manipulating the stack
    // pointer. This is necessary when there is a variable length stack
    // allocation (e.g. alloca), which means it's not possible to allocate
    // space for outgoing arguments from within the function prologue.
    int64_t Amount = MI->getOperand(0).getImm();

    if (Amount != 0) {
      // Ensure the stack remains aligned after adjustment.
      Amount = alignSPAdjust(Amount);

      if (MI->getOpcode() == RISCV::ADJCALLSTACKDOWN)
        Amount = -Amount;

      adjustReg(MBB, MI, DL, SPReg, SPReg, Amount, MachineInstr::NoFlags);
    }
  }

  return MBB.erase(MI);
}
