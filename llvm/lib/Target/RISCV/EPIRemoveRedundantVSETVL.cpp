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
//   for the later 'vsetvli' instruction is actually defined by the prior
//   'vsetvli' (i.e. it is a granted vector length (GVL)).
//
// Currently, this phase requires SSA form, and the analysis is limited within
// a basic block.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"
#include "RISCVRegisterInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "epi-remove-redundant-vsetvl"

static cl::opt<bool>
    DisableRemoveVSETVL("no-epi-remove-redundant-vsetvl", cl::init(false),
                        cl::Hidden,
                        cl::desc("Disable removing redundant vsetvl"));

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
    AU.addRequired<LiveVariables>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
};

char EPIRemoveRedundantVSETVL::ID = 0;

namespace {

// This class holds information related to the operands of a VSETVLI
// instruction. It provides mechanisms to compare such instructions.
struct VSETVLInfo {
  Register AVLReg;
  unsigned SEW;
  unsigned VLMul;

  VSETVLInfo(Register AVLReg, unsigned SEW, unsigned VLMul)
      : AVLReg(AVLReg), SEW(SEW), VLMul(VLMul) {}

  VSETVLInfo(const MachineInstr &MI) {
    assert(MI.getOpcode() == RISCV::VSETVLI);

    const MachineOperand &AVLOp = MI.getOperand(1);
    assert(AVLOp.isReg());

    AVLReg = AVLOp.getReg();

    const MachineOperand &SEWOp = MI.getOperand(2);

    assert(SEWOp.isImm());

    const MachineOperand &VLMulOp = MI.getOperand(3);

    assert(VLMulOp.isImm());

    SEW = (1 << SEWOp.getImm()) * 8;
    VLMul = 1 << VLMulOp.getImm();
  }

  // A VSETVLI instruction has a 'more restrictive VType' if it entails a
  // smaller VLMAX. This is independent of the AVL operand.
  bool hasMoreRestrictiveVType(const VSETVLInfo &other) const {
    return (SEW / VLMul) > (other.SEW / other.VLMul);
  }

  bool computesSameGVL(const VSETVLInfo &other) const {
    return AVLReg == other.AVLReg && (SEW / VLMul) == (other.SEW / other.VLMul);
  }

  bool operator==(const VSETVLInfo &other) const {
    return AVLReg == other.AVLReg && SEW == other.SEW && VLMul == other.VLMul;
  }
};

bool removeDeadVSETVLInstructions(MachineBasicBlock &MBB,
                                  const MachineRegisterInfo &MRI) {
  bool IsMBBModified = false;

  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr *MI(&*II++);

    bool RemovedLastUse;
    do {
      RemovedLastUse = false;

      if (MI->getOpcode() != RISCV::VSETVLI)
        continue;

      assert(MI->getNumExplicitOperands() == 4);
      assert(MI->getNumOperands() == 6);

      const MachineOperand &GVLOp = MI->getOperand(0);
      assert(GVLOp.isReg());

      const MachineOperand &AVLOp = MI->getOperand(1);
      assert(AVLOp.isReg());

      const MachineOperand &ImplVLOp = MI->getOperand(4);
      const MachineOperand &ImplVTypeOp = MI->getOperand(5);

      assert(ImplVLOp.isImplicit() && ImplVLOp.isReg());
      assert(ImplVTypeOp.isImplicit() && ImplVTypeOp.isReg());

      if (GVLOp.isDead() && ImplVLOp.isDead() && ImplVTypeOp.isDead()) {
        LLVM_DEBUG(dbgs() << "Erase trivially dead VSETVLI instruction:\n";
                   MI->dump(); dbgs() << "\n");
        MI->eraseFromParent();
        IsMBBModified = true;

        if (Register::isVirtualRegister(AVLOp.getReg())) {
          // Check if by removing this instruction another def can be set dead.
          MachineInstr *DefMI = MRI.getUniqueVRegDef(AVLOp.getReg());
          assert(DefMI != nullptr && "Expected MachineInstr defining AVLOp");
          MachineOperand *DefMO = DefMI->findRegisterDefOperand(AVLOp.getReg());
          assert(DefMO != nullptr && "Expected MachineOperand defining AVLOp");

          if (MRI.use_nodbg_empty(AVLOp.getReg())) {
            DefMO->setIsDead();
            RemovedLastUse = true;
            MI = DefMI;
          }
        } else {
          // VSETVLMAX instruction.
          assert(AVLOp.getReg() == RISCV::X0);
        }
      }
    } while (RemovedLastUse);
  }

  return IsMBBModified;
}

bool forwardCompatibleAVLToGVLUses(const MachineRegisterInfo &MRI,
                                   const MachineInstr &OriginalMI,
                                   const Register &GVLReg,
                                   const Register &AVLReg) {
  bool Modified = false;
  assert(MRI.hasOneDef(GVLReg));
  for (MachineRegisterInfo::use_nodbg_iterator UI = MRI.use_nodbg_begin(GVLReg),
                                               UIEnd = MRI.use_nodbg_end();
       UI != UIEnd;) {
    MachineOperand &Use(*UI++);
    assert(Use.getParent() != nullptr);
    const MachineInstr &UseInstr = *Use.getParent();

    if (UseInstr.getOpcode() != RISCV::VSETVLI)
      continue;

    // Ensure use is AVL operand
    assert(UseInstr.getOperandNo(&Use) == 1);

    if (!VSETVLInfo(OriginalMI).hasMoreRestrictiveVType(VSETVLInfo(UseInstr))) {
      LLVM_DEBUG(dbgs() << "Forward AVL from VSETVLI instruction:\n";
                 OriginalMI.dump(); dbgs() << "to VSETVLI instruction:\n";
                 UseInstr.dump(); dbgs() << "\n");
      Use.setReg(AVLReg);
      Modified = true;

      // Stop forwarding AVL when we find a more restrictive VSETVLI. Otherwise
      // the outcome GVL can be greater than in the original code. Eg.
      // gvl  = vsetvli avl,  e32, m1
      // gvl2 = vsetvli gvl,  e16, m1
      // gvl3 = vsetvli gvl2, e64, m1 // Can't forward AVL past this instruction
      //        vsetvli gvl3, e32, m1
      continue;
    }

    const MachineOperand &NewGVLOp = UseInstr.getOperand(0);
    assert(NewGVLOp.isReg());
    if (!NewGVLOp.isDead())
      Modified |= forwardCompatibleAVLToGVLUses(MRI, OriginalMI,
                                                NewGVLOp.getReg(), AVLReg);
  }

  // Update liveness.
  if (MRI.use_nodbg_empty(GVLReg)) {
    assert(Register::isVirtualRegister(GVLReg));
    MachineRegisterInfo::def_iterator GVLOpIt = MRI.def_begin(GVLReg);
    assert(GVLOpIt != MRI.def_end());
    MachineOperand &GVLOp = *GVLOpIt;
    GVLOp.setIsDead();
  }

  return Modified;
}

bool forwardCompatibleAVL(MachineBasicBlock &MBB,
                          const MachineRegisterInfo &MRI) {
  bool IsMBBModified = false;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() != RISCV::VSETVLI)
      continue;

    assert(MI.getNumExplicitOperands() == 4);
    assert(MI.getNumOperands() == 6);

    const MachineOperand &GVLOp = MI.getOperand(0);
    assert(GVLOp.isReg());

    const MachineOperand &AVLOp = MI.getOperand(1);
    assert(AVLOp.isReg());

    IsMBBModified |=
        forwardCompatibleAVLToGVLUses(MRI, MI, GVLOp.getReg(), AVLOp.getReg());
  }

  return IsMBBModified;
}

// Class used to compare VSETVLInfo keys based on the outcome GVL. In its terms,
// if a pair of (AVL, VType) tuples compute the same GVL, they are considered
// equal.
struct SameGVLKeyInfo {
  using DenseMapInfoReg = DenseMapInfo<Register>;
  using DenseMapInfoUnsigned = DenseMapInfo<unsigned>;

  static inline VSETVLInfo getEmptyKey() {
    return {DenseMapInfoReg::getEmptyKey(), DenseMapInfoUnsigned::getEmptyKey(),
            DenseMapInfoUnsigned::getEmptyKey()};
  }

  static inline VSETVLInfo getTombstoneKey() {
    return {DenseMapInfoReg::getTombstoneKey(),
            DenseMapInfoUnsigned::getTombstoneKey(),
            DenseMapInfoUnsigned::getTombstoneKey()};
  }

  static unsigned getHashValue(const VSETVLInfo &Val) {
    return DenseMapInfoReg::getHashValue(Val.AVLReg) ^
           (DenseMapInfoUnsigned::getHashValue(Val.SEW/Val.VLMul) << 1);
  }

  static bool isEqual(const VSETVLInfo &LHS, const VSETVLInfo &RHS) {
    return LHS.computesSameGVL(RHS);
  }
};

bool forwardCompatibleGVL(MachineBasicBlock &MBB,
                          const MachineRegisterInfo &MRI) {

  // Map VSETVLInfo (representing VSETVLI input parameters) to the
  // corresponding computed GVL GPR.
  typedef DenseMap<VSETVLInfo, Register, SameGVLKeyInfo> VSETVLInfoMap_t;
  VSETVLInfoMap_t VSETVLInfoMap;

  bool IsMBBModified = false;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() != RISCV::VSETVLI)
      continue;

    assert(MI.getNumExplicitOperands() == 4);
    assert(MI.getNumOperands() == 6);

    MachineOperand &GVLOp = MI.getOperand(0);
    assert(GVLOp.isReg());

    const MachineOperand *AVLOp = &MI.getOperand(1);
    assert(AVLOp->isReg());

    const MachineOperand &ImplVLOp = MI.getOperand(4);
    const MachineOperand &ImplVTypeOp = MI.getOperand(5);

    assert(ImplVLOp.isImplicit() && ImplVLOp.isReg());
    assert(ImplVTypeOp.isImplicit() && ImplVTypeOp.isReg());

    assert(ImplVTypeOp.isDead() == ImplVLOp.isDead());

    VSETVLInfo VI = VSETVLInfo(MI);
    // Find the VSETVLI instruction up in the AVL - GVL chain that actually
    // determines the GVL. That is, the VSETVLI with the most restrictive VType
    // (and thus the VSETVLI that produces the smallest GVL).
    if (Register::isVirtualRegister(AVLOp->getReg())) {
      MachineInstr *ParentMI = MRI.getUniqueVRegDef(AVLOp->getReg());
      assert(ParentMI != nullptr);
      // Given that forwardCompatibleGVL is run after forwardCompatibleAVL, we
      // should find the most restrictive VSETVLI instruction within one jump in
      // the AVL - GVL chain. Otherwise we would need a loop here.
      if (ParentMI->getOpcode() == RISCV::VSETVLI &&
          !VI.hasMoreRestrictiveVType(VSETVLInfo(*ParentMI))) {
        // Ensure is GVL op.
        assert(ParentMI->getOperand(0).isReg() &&
               ParentMI->getOperand(0).getReg() == AVLOp->getReg());

        VI = VSETVLInfo(*ParentMI);
      }
    } else {
      // VSETVLMAX instruction.
      assert(AVLOp->getReg() == RISCV::X0);
    }

    VSETVLInfoMap_t::const_iterator I = VSETVLInfoMap.find(VI);
    if (I == VSETVLInfoMap.end()) {
      VSETVLInfoMap[VI] = GVLOp.getReg();
    } else if (ImplVTypeOp.isDead()) {
      Register PrevGVLReg = I->second;
      // Replace all uses.
      if (!MRI.use_nodbg_empty(GVLOp.getReg())) {
        assert(!GVLOp.isDead());
        for (MachineRegisterInfo::use_nodbg_iterator
                 UI = MRI.use_nodbg_begin(GVLOp.getReg()),
                 UIEnd = MRI.use_nodbg_end();
             UI != UIEnd;) {
          MachineOperand &Use(*UI++);
          Use.setReg(PrevGVLReg);
          IsMBBModified = true;
        }

        assert(Register::isVirtualRegister(PrevGVLReg));
        assert(MRI.hasOneDef(PrevGVLReg));
        MachineOperand &PrevGVLOp = *MRI.def_begin(PrevGVLReg);
        LLVM_DEBUG(dbgs() << "Forward GVL from VSETVLI instruction:\n";
                   PrevGVLOp.getParent()->dump();
                   dbgs() << "to VSETVLI instruction:\n"; MI.dump();
                   dbgs() << "\n");
        if (PrevGVLOp.isDead()) {
          // Now it has become alive.
          PrevGVLOp.setIsDead(false);
        }
        // No uses left, thus dead.
        GVLOp.setIsDead();
      }
    }
  }

  return IsMBBModified;
}

bool removeDuplicateVSETVLI(MachineBasicBlock &MBB) {
  bool IsMBBModified = false;

  MachineInstr *RefInstr = nullptr;
  for (MachineBasicBlock::instr_iterator II = MBB.instr_begin(),
                                         IIEnd = MBB.instr_end();
       II != IIEnd;) {
    MachineInstr &MI(*II++);

    if (MI.getOpcode() != RISCV::VSETVLI) {
      // Check if the current intruction defines VL (e.g. 'vsetvl', (but not
      // 'vsetvli')). If it does, we should not remove a subsequent 'vsetvli',
      // even when its vtype matches the reference 'vsetvli's. To force this
      // we clear 'RefInstr'.
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

      // VL may be changed within functions, we can't reuse defs through calls
      if (MI.isCall()) {
        RefInstr = nullptr;
        continue;
      }

      continue;
    }

    assert(MI.getNumExplicitOperands() == 4);
    assert(MI.getNumOperands() == 6);

    if (RefInstr == nullptr) {
      RefInstr = &MI;
      continue;
    }

    assert(&MI != RefInstr);

    const MachineOperand &GVLOp = MI.getOperand(0);
    assert(GVLOp.isReg());

    if (GVLOp.isDead() && VSETVLInfo(MI) == VSETVLInfo(*RefInstr)) {
      LLVM_DEBUG(dbgs() << "Remove duplicate VSETVLI instruction:\n"; MI.dump();
                 dbgs() << "in favour of:\n"; RefInstr->dump(); dbgs() << "\n");
      MI.eraseFromParent();

      MachineOperand &RefInstrImplVLOp = RefInstr->getOperand(4);
      MachineOperand &RefInstrImplVTypeOp = RefInstr->getOperand(5);

      assert(RefInstrImplVLOp.isImplicit() && RefInstrImplVLOp.isReg());
      assert(RefInstrImplVTypeOp.isImplicit() && RefInstrImplVTypeOp.isReg());
      assert(RefInstrImplVTypeOp.isDead() == RefInstrImplVLOp.isDead());

      const MachineOperand &ImplVLOp = MI.getOperand(4);
      const MachineOperand &ImplVTypeOp = MI.getOperand(5);

      assert(ImplVLOp.isImplicit() && ImplVLOp.isReg());
      assert(ImplVTypeOp.isImplicit() && ImplVTypeOp.isReg());
      assert(ImplVTypeOp.isDead() == ImplVLOp.isDead());

      if (RefInstrImplVTypeOp.isDead() && !ImplVTypeOp.isDead()) {
        // Now it has become alive.
        RefInstrImplVTypeOp.setIsDead(false);
        RefInstrImplVLOp.setIsDead(false);
      }

      IsMBBModified = true;

      // MI has been removed, do not update RefInstr.
      continue;
    }

    RefInstr = &MI;
  }

  return IsMBBModified;
}

} // namespace

bool EPIRemoveRedundantVSETVL::runOnMachineFunction(MachineFunction &F) {

  LLVM_DEBUG(
      dbgs() << "********** Begin remove redundant VSETVLI phase on function '"
             << F.getName() << "' **********\n\n");

  if (skipFunction(F.getFunction()) || DisableRemoveVSETVL)
    return false;

  const MachineRegisterInfo &MRI = F.getRegInfo();
  assert(MRI.isSSA());
  assert(MRI.tracksLiveness());

  bool IsFunctionModified = false;

  for (MachineBasicBlock &MBB : F) {
    IsFunctionModified |= forwardCompatibleAVL(MBB, MRI);
    LLVM_DEBUG(dbgs() << "--- BB dump after forwardCompatibleAVL ---";
               MBB.dump(); dbgs() << "\n");

    IsFunctionModified |= forwardCompatibleGVL(MBB, MRI);
    LLVM_DEBUG(dbgs() << "--- BB dump after forwardCompatibleGVL ---";
               MBB.dump(); dbgs() << "\n");

    IsFunctionModified |= removeDuplicateVSETVLI(MBB);
    LLVM_DEBUG(dbgs() << "--- BB dump after removeDuplicateVSETVLI ---";
               MBB.dump(); dbgs() << "\n");
  }

  LiveVariables LV;
  LV.runOnMachineFunction(F);

  for (MachineBasicBlock &MBB : F) {
    IsFunctionModified |= removeDeadVSETVLInstructions(MBB, MRI);
    LLVM_DEBUG(dbgs() << "--- BB dump after removeDeadVSETVLInstructions ---";
               MBB.dump(); dbgs() << "\n");
  }

  LLVM_DEBUG(
      dbgs() << "*********** End remove redundant VSETVLI phase on function '"
             << F.getName() << "' ***********\n");

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
