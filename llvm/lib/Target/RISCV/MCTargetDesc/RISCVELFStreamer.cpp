//===-- RISCVELFStreamer.cpp - RISCV ELF Target Streamer Methods ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVELFStreamer.h"
#include "RISCVMCExpr.h"
#include "RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVAsmBackend.h"
#include "Utils/RISCVBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// This part is for ELF object output.
RISCVTargetELFStreamer::RISCVTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : RISCVTargetStreamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<RISCVAsmBackend &>(MCA.getBackend());
  RISCVABI::ABI ABI = MAB.getTargetABI();
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialised target ABI");

  unsigned EFlags = MCA.getELFHeaderEFlags();

  if (Features[RISCV::FeatureStdExtC])
    EFlags |= ELF::EF_RISCV_RVC;

  switch (ABI) {
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_SINGLE;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_DOUBLE;
    break;
  case RISCVABI::ABI_ILP32E:
    EFlags |= ELF::EF_RISCV_RVE;
    break;
  case RISCVABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

  MCA.setELFHeaderEFlags(EFlags);
}

RISCVELFStreamer &RISCVTargetELFStreamer::getStreamer() {
  return static_cast<RISCVELFStreamer &>(Streamer);
}

void RISCVTargetELFStreamer::emitDirectiveOptionPush() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPop() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRelax() {}

void RISCVELFStreamer::EmitInstruction(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  // Lower pseudo-instructions that we can't lower any later
  // because they require labels.
  if (EmitPseudoInstruction(Inst, STI))
    return;

  MCELFStreamer::EmitInstruction(Inst, STI);
}

bool RISCVELFStreamer::EmitPseudoInstruction(const MCInst &Inst,
                                             const MCSubtargetInfo &STI) {
  switch (Inst.getOpcode()) {
  default:
    return false;
  // FIXME this should go away
  // case RISCV::PseudoLLA: {
  //   // PC-rel addressing
  //   MCContext &Ctx = getContext();

  //   // TmpLabel: AUIPC rdest, %pcrel_hi(symbol)
  //   //           ADDI rdest, %pcrel_lo(TmpLabel)
  //   MCSymbol *TmpLabel = Ctx.createTempSymbol(
  //       "pcrel_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
  //   EmitLabel(TmpLabel);

  //   MCOperand DestReg = Inst.getOperand(0);
  //   const RISCVMCExpr *Symbol = RISCVMCExpr::create(
  //       Inst.getOperand(1).getExpr(), RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx);

  //   MCInst AUIPC =
  //       MCInstBuilder(RISCV::AUIPC).addOperand(DestReg).addExpr(Symbol);
  //   EmitInstruction(AUIPC, STI);

  //   const MCExpr *RefToLinkTmpLabel =
  //       RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
  //                           RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

  //   MCInst Addi = MCInstBuilder(RISCV::ADDI)
  //                     .addOperand(DestReg)
  //                     .addOperand(DestReg)
  //                     .addExpr(RefToLinkTmpLabel);
  //   EmitInstruction(Addi, STI);
  //   break;
  // }
  // case RISCV::PseudoLA: {
  //   // GOT addressing
  //   MCContext &Ctx = getContext();

  //   // TmpLabel: AUIPC rdest, %got_pcrel_hi(symbol)
  //   //         L{W,D} rdest, rdest, %pcrel_lo(TmpLabel)
  //   MCSymbol *TmpLabel = Ctx.createTempSymbol(
  //       "got_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
  //   EmitLabel(TmpLabel);

  //   MCOperand DestReg = Inst.getOperand(0);
  //   const RISCVMCExpr *Symbol = RISCVMCExpr::create(
  //       Inst.getOperand(1).getExpr(), RISCVMCExpr::VK_RISCV_GOT_HI, Ctx);

  //   MCInst AUIPC =
  //       MCInstBuilder(RISCV::AUIPC).addOperand(DestReg).addExpr(Symbol);
  //   EmitInstruction(AUIPC, STI);

  //   const MCExpr *RefToLinkTmpLabel =
  //       RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
  //                           RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

  //   bool is64Bit = STI.getTargetTriple().getArch() == Triple::riscv64;
  //   unsigned int LoadOpCode = is64Bit ? RISCV::LD : RISCV::LW;
  //   MCInst Load = MCInstBuilder(LoadOpCode)
  //                     .addOperand(DestReg)
  //                     .addOperand(DestReg)
  //                     .addExpr(RefToLinkTmpLabel);
  //   EmitInstruction(Load, STI);
  //   break;
  // }
  case RISCV::PseudoLATLSIE: {
    // GOT addressing
    MCContext &Ctx = getContext();

    // TmpLabel: AUIPC rdest, %got_hi(symbol)
    //         L{W,D} rdest, rdest, %pcrel_lo(TmpLabel)
    // Note: there is not such thing as %got_hi, yet
    MCSymbol *TmpLabel = Ctx.createTempSymbol(
        "tls_got_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
    EmitLabel(TmpLabel);

    MCOperand DestReg = Inst.getOperand(0);
    const RISCVMCExpr *Symbol =
        RISCVMCExpr::create(Inst.getOperand(1).getExpr(),
                            RISCVMCExpr::VK_RISCV_TLS_GOT_HI_Pseudo, Ctx);

    MCInst AUIPC =
        MCInstBuilder(RISCV::AUIPC).addOperand(DestReg).addExpr(Symbol);
    EmitInstruction(AUIPC, STI);

    const MCExpr *RefToLinkTmpLabel =
        RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

    bool is64Bit = STI.getTargetTriple().getArch() == Triple::riscv64;
    unsigned int LoadOpCode = is64Bit ? RISCV::LD : RISCV::LW;
    MCInst Load = MCInstBuilder(LoadOpCode)
                      .addOperand(DestReg)
                      .addOperand(DestReg)
                      .addExpr(RefToLinkTmpLabel);
    EmitInstruction(Load, STI);
    break;
  }
  case RISCV::PseudoLATLSGD: {
    // PC-rel addressing for TLS General / Local Dynamic
    MCContext &Ctx = getContext();

    // TmpLabel: AUIPC rdest, %tls_gd_hi(symbol)
    //           ADDI rdest, %pcrel_lo(TmpLabel)
    // Note: there is not such thing as %tls_gd_hi, yet
    MCSymbol *TmpLabel = Ctx.createTempSymbol(
        "tls_gd_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
    EmitLabel(TmpLabel);

    MCOperand DestReg = Inst.getOperand(0);
    const RISCVMCExpr *Symbol =
        RISCVMCExpr::create(Inst.getOperand(1).getExpr(),
                            RISCVMCExpr::VK_RISCV_TLS_GD_HI_Pseudo, Ctx);

    MCInst AUIPC =
        MCInstBuilder(RISCV::AUIPC).addOperand(DestReg).addExpr(Symbol);
    EmitInstruction(AUIPC, STI);

    const MCExpr *RefToLinkTmpLabel =
        RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

    MCInst Addi = MCInstBuilder(RISCV::ADDI)
                      .addOperand(DestReg)
                      .addOperand(DestReg)
                      .addExpr(RefToLinkTmpLabel);
    EmitInstruction(Addi, STI);
    break;
  }
  }

  return true;
}
