//===- EPIRemoveRedundantVSETVLGlobal.cpp - Remove redundant VSETVL
//     instructions globally                                              -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that removes the first VSETVLI
// instruction of a basic block when it is known to have no effect, and thus is
// redundant.
//
// A data-flow algorithm is used to determine which VSETVLI instruction (if any)
// will be available across all predecessors of a given basic block. Then, the
// first VSETVLI instruction in the block can be removed if it matches the
// VSETVLI state available through all predecessors of the block.
//
// Currently, this phase requires SSA form.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "epi-remove-redundant-vsetvl-global"

static cl::opt<bool> DisableRemoveVSETVLGlobal(
    "no-epi-remove-redundant-vsetvl-global", cl::init(false), cl::Hidden,
    cl::desc("Disable removing redundant vsetvl global"));

namespace {

class EPIRemoveRedundantVSETVLGlobal : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  EPIRemoveRedundantVSETVLGlobal() : MachineFunctionPass(ID) {}

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

char EPIRemoveRedundantVSETVLGlobal::ID = 0;

namespace {

// This class holds information related to the operands of a VSETVLI
// instruction. In particular it contains the (AVL, SEW, VLMul) triplet.
struct VSETVLInstr : public std::tuple<Register, unsigned, unsigned> {
  using Base = std::tuple<Register, unsigned, unsigned>;

  VSETVLInstr() : Base(std::make_tuple(Register(), ~0U, ~0U)) {}

  VSETVLInstr(Register GVLReg, Register AVLReg, unsigned SEW, unsigned VLMul)
      : Base(std::make_tuple(AVLReg, SEW, VLMul)), GVLReg(GVLReg) {}

  Register getGVLReg() { return GVLReg; }

  Register getAVLReg() { return std::get<0>(*this); }

  Register getSEW() { return std::get<1>(*this); }

  Register getVLMul() { return std::get<2>(*this); }

  static VSETVLInstr createFromMI(const MachineInstr &MI) {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI);

    assert(MI.getNumExplicitOperands() == 3);
    assert(MI.getNumOperands() == 5);

    const MachineOperand &DefOp = MI.getOperand(0);
    assert(DefOp.isReg());
    Register GVLReg = DefOp.getReg();

    const MachineOperand &AVLOp = MI.getOperand(1);
    assert(AVLOp.isReg());

    Register AVLReg = AVLOp.getReg();

    const MachineOperand &VTypeIOp = MI.getOperand(2);
    assert(VTypeIOp.isImm());

    unsigned VTypeI = VTypeIOp.getImm();

    unsigned SEWBits = (VTypeI >> 2) & 0x7;
    unsigned VMulBits = VTypeI & 0x3;

    unsigned SEW = (1 << SEWBits) * 8;
    unsigned VLMul = 1 << VMulBits;

    return VSETVLInstr(GVLReg, AVLReg, SEW, VLMul);
  }

private:
  Register GVLReg;
};

} // namespace

bool EPIRemoveRedundantVSETVLGlobal::runOnMachineFunction(MachineFunction &F) {
  if (skipFunction(F.getFunction()) || DisableRemoveVSETVLGlobal)
    return false;

  LLVM_DEBUG(dbgs() << "********** Begin remove redundant VSETVLI global "
                       "phase on function '"
                    << F.getName() << "' **********\n\n");

  const MachineRegisterInfo &MRI = F.getRegInfo();
  assert(MRI.isSSA());

  bool IsFunctionModified = false;

  // Map MachineBasicBlock to the last VSETVLInstr found in the block.
  typedef DenseMap<const MachineBasicBlock*, VSETVLInstr> BBVSETVLInstrMap_t;
  BBVSETVLInstrMap_t BBLastVSETVLInstrMap;
  BBVSETVLInstrMap_t BBVSETVLInstrOutMap;
  BBVSETVLInstrMap_t BBVSETVLInstrInMap;

  // Instantiate 'Unmatched' value (as per the default constructor).
  // Unmatched value means that we may observe the effects of different VSETVLI
  // instructions along different paths that lead to the same program point.
  VSETVLInstr Unmatched;

  // Compute the last VSETVLI instruction found in each basic blocks (if any).
  for (const MachineBasicBlock &MBB : F) {
    MachineBasicBlock::const_reverse_instr_iterator LastVSETVLI = std::find_if(
        MBB.instr_rbegin(), MBB.instr_rend(), [](const MachineInstr &MI) {
          return MI.getOpcode() == RISCV::PseudoVSETVLI;
        });

    // No VSETVLI instructions in this MBB.
    if (LastVSETVLI == MBB.instr_rend())
      continue;

    VSETVLInstr VI = VSETVLInstr::createFromMI(*LastVSETVLI);
    assert(!BBLastVSETVLInstrMap.count(&MBB) &&
           "Should only map one VSETVLInstr per MBB");
    BBLastVSETVLInstrMap[&MBB] = VI;
  }

  // Initialize output map for basic blocks that have an explicit VSETVLI.
  BBVSETVLInstrOutMap = BBLastVSETVLInstrMap;

  bool HasChanged;

  do {
    HasChanged = false;

    for (const MachineBasicBlock &MBB : F) {
      // Compute In[MBB].

      // In[MBB], initialized to Undef value (no value set).
      // Undef value means that no VSETVLI instruction is yet known to reach
      // the program point in question.
      Optional<VSETVLInstr> VI;

      if (MBB.pred_size() == 0) {
        // If no predecessors (entry MBB), In[MBB] := Unmatched.
        // We cannot claim anything about the incoming VSETVLI instruction, so
        // the only sensible value here is Umatched. This is the most
        // conservative value.
        VI = Unmatched;
      }
      for (const MachineBasicBlock *Predecessor : MBB.predecessors()) {
        assert(Predecessor != nullptr && "Unexpected NULL basic block");
        BBVSETVLInstrMap_t::iterator VII =
            BBVSETVLInstrOutMap.find(Predecessor);

        // VI ∧ Undef = VI.
        if (VII == BBVSETVLInstrOutMap.end())
          continue;

        if (!VI.hasValue()) {
          // Undef ∧ VI = VI
          VI = VII->getSecond();
        } else if (VI.getValue() != VII->getSecond()) {
          // VI ∧ different VI = Unmatched.
          VI = Unmatched;

          // Unmatched ∧ * = Unmatched.
          break;
        }
        // else VI ∧ equal VI = VI.
      }

      // In[MBB] == Undef, no updates.
      if (!VI.hasValue())
        continue;

      // Update the solution map.
      BBVSETVLInstrInMap[&MBB] = VI.getValue();

      // Apply transfer function over MBB.

      // If there is an explicit VSETVLI instruction in MBB, Out[MBB] is fixed.
      if (BBLastVSETVLInstrMap.count(&MBB))
        continue;

      // Out[MBB] hasn't changed.
      if (BBVSETVLInstrOutMap.count(&MBB) &&
          (BBVSETVLInstrOutMap[&MBB] == VI.getValue()))
        continue;

      // Update Out[MBB] and account for changes.
      BBVSETVLInstrOutMap[&MBB] = VI.getValue();
      HasChanged = true;
    }
  } while (HasChanged);

  LLVM_DEBUG(
    for (const MachineBasicBlock &MBB : F) {
      BBVSETVLInstrMap_t::const_iterator I = BBVSETVLInstrInMap.find(&MBB);
      // Undef shouldn't appear in the solution.
      assert(I != BBVSETVLInstrInMap.end() &&
             "Unexpected 'Undef' after data flow convergence");
      VSETVLInstr VI = I->getSecond();
      if (VI == Unmatched) {
        dbgs() << "BB: '" << MBB.getNumber() << "." << MBB.getName()
               << "'\tUnmatchedVIs\n";
      } else {
        dbgs() << "BB: '" << MBB.getNumber() << "." << MBB.getName() << "'\t"
               << "(AVL: '" << printReg(VI.getAVLReg()) << "', SEW: e"
               << VI.getSEW() << ", LMUL: m" << VI.getVLMul() << ")\n";
      }
    }
    dbgs() << "\n"
  );

  for (MachineBasicBlock &MBB : F) {
    BBVSETVLInstrMap_t::const_iterator I = BBVSETVLInstrInMap.find(&MBB);
    // Undef shouldn't appear in the solution.
    assert(I != BBVSETVLInstrInMap.end() &&
           "Unexpected 'Undef' after data flow convergence");
    VSETVLInstr InVI = I->getSecond();

    MachineBasicBlock::instr_iterator FirstVSETVLI = std::find_if(
        MBB.instr_begin(), MBB.instr_end(), [](const MachineInstr &MI) {
          return MI.getOpcode() == RISCV::PseudoVSETVLI;
        });

    // There is nothing to remove.
    if (FirstVSETVLI == MBB.instr_end())
      continue;

    LLVM_DEBUG(dbgs() << "Considering removal of \n"; FirstVSETVLI->dump(););

    VSETVLInstr FirstVI = VSETVLInstr::createFromMI(*FirstVSETVLI);
    Register GVLReg = FirstVI.getGVLReg();
    assert((GVLReg.isVirtual() || GVLReg == RISCV::X0) &&
           "GVL must be a virtual register or X0!");

    // Don't remove VSETVLI whose GVL is being used.
    // FIXME: Replace uses with predecessor's GVL, using a phi if necessary.
    if (GVLReg.isVirtual() && !MRI.use_empty(GVLReg)) {
      LLVM_DEBUG(dbgs() << "Not removing because its value is still used.\n";
                 dbgs() << "Printing uses of defined register "
                        << printReg(GVLReg) << "\n";
                 for (auto UIt = MRI.use_begin(GVLReg), UEnd = MRI.use_end();
                      UIt != UEnd; UIt++) {
                   MachineOperand &MO = *UIt;
                   MachineInstr *MI = MO.getParent();
                   MI->dump();
                 });
      continue;
    }

    if (InVI == FirstVI) {
      LLVM_DEBUG(dbgs() << "Remove redundant VSETVLI instruction leading BB '"
                        << MBB.getNumber() << "." << MBB.getName() << "':\n";
                 FirstVSETVLI->dump(); dbgs() << "\n");
      FirstVSETVLI->eraseFromParent();
      IsFunctionModified = true;
    }
  }

  LLVM_DEBUG(dbgs() << "*********** End remove redundant VSETVLI global "
                       "phase on function '"
                    << F.getName() << "' ***********\n");

  return IsFunctionModified;
}

} // namespace

INITIALIZE_PASS(EPIRemoveRedundantVSETVLGlobal,
                "epi-remove-redundant-vsetvl-global",
                "EPI Remove Redundant VSETVL global pass", false, false)
namespace llvm {

FunctionPass *createEPIRemoveRedundantVSETVLGlobalPass() {
  return new EPIRemoveRedundantVSETVLGlobal();
}

} // end of namespace llvm
