//===-- RISCVELFStreamer.cpp - RISCV ELF Target Streamer Methods ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVELFStreamer.h"
#include "RISCVMCExpr.h"
#include "RISCVMCTargetDesc.h"
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

  unsigned EFlags = MCA.getELFHeaderEFlags();

  if (Features[RISCV::FeatureStdExtC])
    EFlags |= ELF::EF_RISCV_RVC;

  if (Features[RISCV::HardFloatSingle])
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_SINGLE;
  else if (Features[RISCV::HardFloatDouble])
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_DOUBLE;

  MCA.setELFHeaderEFlags(EFlags);
}

RISCVELFStreamer &RISCVTargetELFStreamer::getStreamer() {
  return static_cast<RISCVELFStreamer &>(Streamer);
}

void RISCVTargetELFStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoPIC() {}

void RISCVELFStreamer::EmitInstruction(const MCInst &Inst,
                                       const MCSubtargetInfo &STI,
                                       bool PrintSchedInfo) {
  // Lower pseudo-instructions that we can't lower any later
  // because they require labels.
  if (EmitPseudoInstruction(Inst, STI, PrintSchedInfo))
    return;

  MCELFStreamer::EmitInstruction(Inst, STI, PrintSchedInfo);
}

bool RISCVELFStreamer::EmitPseudoInstruction(const MCInst &Inst,
                                             const MCSubtargetInfo &STI,
                                             bool PrintSchedInfo) {
  switch (Inst.getOpcode()) {
  default:
    return false;
  // FIXME this should go away
  case RISCV::PseudoLLA: {
    // PC-rel addressing
    MCContext &Ctx = getContext();

    // TmpLabel: AUIPC rdest, %pcrel_hi(symbol)
    //           ADDI rdest, %pcrel_lo(TmpLabel)
    MCSymbol *TmpLabel = Ctx.createTempSymbol(
        "pcrel_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
    EmitLabel(TmpLabel);

    MCOperand DestReg = Inst.getOperand(0);
    const RISCVMCExpr *Symbol = RISCVMCExpr::create(
        Inst.getOperand(1).getExpr(), RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx);

    MCInst AUIPC =
        MCInstBuilder(RISCV::AUIPC).addOperand(DestReg).addExpr(Symbol);
    EmitInstruction(AUIPC, STI, PrintSchedInfo);

    const MCExpr *RefToLinkTmpLabel =
        RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

    MCInst Addi = MCInstBuilder(RISCV::ADDI)
                      .addOperand(DestReg)
                      .addOperand(DestReg)
                      .addExpr(RefToLinkTmpLabel);
    EmitInstruction(Addi, STI, PrintSchedInfo);
    break;
  }
  case RISCV::PseudoLA: {
    // GOT addressing
    MCContext &Ctx = getContext();

    // TmpLabel: AUIPC rdest, %got_hi(symbol)
    //         L{W,D} rdest, rdest, %pcrel_lo(TmpLabel)
    // Note: there is not such thing as %got_hi, yet
    MCSymbol *TmpLabel = Ctx.createTempSymbol(
        "got_hi", /* AlwaysAddSuffix */ true, /* CanBeUnnamed */ false);
    EmitLabel(TmpLabel);

    MCOperand DestReg = Inst.getOperand(0);
    const RISCVMCExpr *Symbol = RISCVMCExpr::create(
        Inst.getOperand(1).getExpr(), RISCVMCExpr::VK_RISCV_GOT_HI_Pseudo, Ctx);

    MCInst AUIPC =
        MCInstBuilder(RISCV::AUIPC).addOperand(DestReg).addExpr(Symbol);
    EmitInstruction(AUIPC, STI, PrintSchedInfo);

    const MCExpr *RefToLinkTmpLabel =
        RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

    bool is64Bit = STI.getTargetTriple().getArch() == Triple::riscv64;
    unsigned int LoadOpCode = is64Bit ? RISCV::LD : RISCV::LW;
    MCInst Load = MCInstBuilder(LoadOpCode)
                      .addOperand(DestReg)
                      .addOperand(DestReg)
                      .addExpr(RefToLinkTmpLabel);
    EmitInstruction(Load, STI, PrintSchedInfo);
    break;
  }
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
    EmitInstruction(AUIPC, STI, PrintSchedInfo);

    const MCExpr *RefToLinkTmpLabel =
        RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

    bool is64Bit = STI.getTargetTriple().getArch() == Triple::riscv64;
    unsigned int LoadOpCode = is64Bit ? RISCV::LD : RISCV::LW;
    MCInst Load = MCInstBuilder(LoadOpCode)
                      .addOperand(DestReg)
                      .addOperand(DestReg)
                      .addExpr(RefToLinkTmpLabel);
    EmitInstruction(Load, STI, PrintSchedInfo);
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
    EmitInstruction(AUIPC, STI, PrintSchedInfo);

    const MCExpr *RefToLinkTmpLabel =
        RISCVMCExpr::create(MCSymbolRefExpr::create(TmpLabel, Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);

    MCInst Addi = MCInstBuilder(RISCV::ADDI)
                      .addOperand(DestReg)
                      .addOperand(DestReg)
                      .addExpr(RefToLinkTmpLabel);
    EmitInstruction(Addi, STI, PrintSchedInfo);
    break;
  }
  }

  return true;
}
