//===-- RISCVELFStreamer.h - RISCV ELF Target Streamer ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVELFSTREAMER_H
#define LLVM_LIB_TARGET_RISCV_RISCVELFSTREAMER_H

#include "RISCVTargetStreamer.h"
#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

class RISCVELFStreamer;
class RISCVTargetELFStreamer : public RISCVTargetStreamer {
public:
  RISCVELFStreamer &getStreamer();
  RISCVTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  virtual void emitDirectiveOptionRVC();
  virtual void emitDirectiveOptionNoRVC();
  virtual void emitDirectiveOptionPIC();
  virtual void emitDirectiveOptionNoPIC();
};

class RISCVELFStreamer : public MCELFStreamer {
private:
  bool EmitPseudoInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                             bool PrintSchedInfo);

public:
  using MCELFStreamer::MCELFStreamer;

  void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                       bool PrintSchedInfo) override;
};
}
#endif
