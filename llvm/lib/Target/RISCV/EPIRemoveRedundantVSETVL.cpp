//===- EPIRemoveRedundantVSETVL.cpp - Remove redundant VSETVL instructions ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that removes the 'vsetvl' instructions
// that are known to have no effect, and thus are redundant.
//
// In particular, given a pair of 'vsetvli' instructions that specify the same
// SEW and VLMUL and have no instruction modifying the VL inbetween, the
// later can be safely removed in the following scenarios:
//
// - If we detect that the value passed as the requested vector length (AVL) is
//   found in the same virtual register in both instructions.
//
// - If we detect that the value passed as the requested vector length (AVL)
//   for the posterior 'vsetvli' instruction is actually defined by the prior
//   'vsetvli' (i.e. it is a granted vector length (GVL)).
//
// Currently, this phase requires SSA form, and the analysis is limited within
// a basic block.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Function.h"

using namespace llvm;

#define DEBUG_TYPE "EPIRemoveRedundantVSETVL"

namespace {

class EPIRemoveRedundantVSETVL : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  EPIRemoveRedundantVSETVL() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  // This pass modifies the program, but does not modify the CFG
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

char EPIRemoveRedundantVSETVL::ID = 0;

bool EPIRemoveRedundantVSETVL::runOnMachineFunction(MachineFunction &F) {
  bool IsFunctionModified = false;

  // This map binds the output operand of an instruction to the output operand
  // of another one. In particular, it holds a mapping between a VL definition
  // from a redundant 'vsetvli' instruction and the VL definition from the
  // previous 'vsetvli' instruction that makes the first instruction redundant.
  // This map is used to replace all uses of the redundant 'vsetvli's
  // definition with the corresponding definition of a (non-removable)
  // 'vsetvli' that preceedes it.
  typedef llvm::DenseMap<unsigned, unsigned> OutputOpMap_t;
  OutputOpMap_t OutputOpMap;

  for (MachineBasicBlock &MBB : F) {
    const MachineInstr *RefInstr = nullptr;
    for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                           IIEnd = MBB.instr_end();
         II != IIEnd;) {
      MachineInstr &MI(*II++);

      if (MI.getOpcode() != RISCV::VSETVLI) {
        // Check if the current intruction defines VL (e.g. 'vsetvl').
        // If it does, we should not remove a posterior 'vsetvli', even when
        // its operands match the reference 'vsetvli's. To force this we clear
        // 'RefInstr'.
        for (auto const &Def : MI.defs()) {
          assert(Def.isReg());
          if (Def.getReg() == RISCV::VL) {
            RefInstr = nullptr;
            continue;
          }
        }

        // Implicit defs are not included in MachineInstruction::defs()
        for (auto const &ImplOp : MI.implicit_operands()) {
          if (ImplOp.isReg() && (ImplOp.getReg() == RISCV::VL) &&
              ImplOp.isDef()) {
            RefInstr = nullptr;
            continue;
          }
        }

        continue;
      }

      assert(MI.getNumExplicitOperands() == 4);

      if (RefInstr == nullptr) {
        RefInstr = &MI;
        continue;
      }

      assert(&MI != RefInstr);

      const MachineOperand &GVLOp = MI.getOperand(0);
      const MachineOperand &RefInstrGVLOp = RefInstr->getOperand(0);

      assert(GVLOp.isReg() && RefInstrGVLOp.isReg());

      const MachineOperand &AVLOp = MI.getOperand(1);
      const MachineOperand &RefInstrAVLOp = RefInstr->getOperand(1);

      assert(AVLOp.isReg() && RefInstrAVLOp.isReg());

      // If a mapping exists, get the mapped operand
      unsigned AVLOpReg = AVLOp.getReg();
      OutputOpMap_t::iterator MappedOp = OutputOpMap.find(AVLOpReg);
      if (MappedOp != OutputOpMap.end())
        AVLOpReg = MappedOp->getSecond();

      // If succesor's AVL corresponds to predecessor's GVL, and all other
      // arguments are equal, we may remove the successor
      if ((AVLOpReg != RefInstrGVLOp.getReg()) &&
          (AVLOpReg != RefInstrAVLOp.getReg())) {
        RefInstr = &MI;
        continue;
      }

      const MachineOperand &SEWOp = MI.getOperand(2);
      const MachineOperand &RefInstrSEWOp = RefInstr->getOperand(2);

      assert(SEWOp.isImm() && RefInstrSEWOp.isImm());

      if (SEWOp.getImm() != RefInstrSEWOp.getImm()) {
        RefInstr = &MI;
        continue;
      }

      const MachineOperand &VLMulOp = MI.getOperand(3);
      const MachineOperand &RefInstrVLMulOp = RefInstr->getOperand(3);

      assert(VLMulOp.isImm() && RefInstrVLMulOp.isImm());

      if (VLMulOp.getImm() != RefInstrVLMulOp.getImm()) {
        RefInstr = &MI;
        continue;
      }

      // Before removing the successor we need to make sure its output is
      // dead or replace the output uses by the predecessor's output
      if (!GVLOp.isDead()) {
        OutputOpMap_t::iterator Op = OutputOpMap.find(RefInstrGVLOp.getReg());
        if (Op != OutputOpMap.end())
          OutputOpMap[GVLOp.getReg()] = Op->getSecond();
        else
          OutputOpMap[GVLOp.getReg()] = RefInstrGVLOp.getReg();
      }

      MI.eraseFromParent();
      IsFunctionModified = true;
    }
  }

  // Now replace the removed output operands according to the map
  for (MachineBasicBlock &MBB : F) {
    for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                           IIEnd = MBB.instr_end();
         II != IIEnd; ++II) {
      MachineInstr &MI(*II);
      for (auto &Use : MI.uses()) {
        if (Use.isReg()) {
          OutputOpMap_t::iterator Op = OutputOpMap.find(Use.getReg());
          if (Op != OutputOpMap.end())
            Use.setReg(Op->getSecond());
        }
      }
    }
  }

  return IsFunctionModified;
}

} // namespace

INITIALIZE_PASS(EPIRemoveRedundantVSETVL, "epi-remove-redundant-vsetvl",
                "EPI Remove Redundant VSETVL pass", false, false)
namespace llvm {

FunctionPass *createEPIRemoveRedundantVSETVLPass() {
  return new EPIRemoveRedundantVSETVL();
}

} // end of namespace llvm
