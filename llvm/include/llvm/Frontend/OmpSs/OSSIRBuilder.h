//===- IR/OmpSsIRBuilder.h - OmpSs-2 encoding builder for LLVM IR - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OmpSsIRBuilder class and helpers used as a convenient
// way to create LLVM instructions for OmpSs directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OMPSS_IR_IRBUILDER_H
#define LLVM_OMPSS_IR_IRBUILDER_H

#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Allocator.h"
#include <forward_list>

namespace llvm {

/// An interface to create LLVM-IR for OmpSs-2 directives.
///
/// Each OmpSs-2 directive has a corresponding public generator method.
class OmpSsIRBuilder {
public:
  /// Create a new OmpSsIRBuilder operating on the given module \p M. This will
  /// not have an effect on \p M (see initialize).
  OmpSsIRBuilder(Module &M) : M(M), Builder(M.getContext()) {}

  /// Initialize the internal state, this will put structures types and
  /// potentially other helpers into the underlying module. Must be called
  /// before any other method and only once!
  void initialize();

  /// Finalize the underlying module, e.g., by outlining regions.
  void finalize();

  struct DirectiveClausesInfo {
    Value *LowerBound = nullptr;
    Value *UpperBound = nullptr;
    Value *Step = nullptr;
    Value *LoopType = nullptr;
    Value *IndVar = nullptr;
    Value *If = nullptr;
    Value *Final = nullptr;
    Value *Cost = nullptr;
    Value *Priority = nullptr;
    MapVector<Value *, Type *> Shareds;
    MapVector<Value *, Type *> Privates;
    MapVector<Value *, Type *> Firstprivates;
    SmallVector<std::pair<Value *, Value *>> Copies;
    SmallVector<std::pair<Value *, Value *>> Inits;
    SmallVector<std::pair<Value *, Value *>> Deinits;
    SmallVector< SmallVector<Value *, 4>, 4> Ins;
    SmallVector< SmallVector<Value *, 4>, 4> Outs;
    SmallVector< SmallVector<Value *, 4>, 4> Inouts;
    SmallVector< SmallVector<Value *, 4>, 4> Concurrents;
    SmallVector< SmallVector<Value *, 4>, 4> Commutatives;
    SmallVector< SmallVector<Value *, 4>, 4> WeakIns;
    SmallVector< SmallVector<Value *, 4>, 4> WeakOuts;
    SmallVector< SmallVector<Value *, 4>, 4> WeakInouts;
    SmallVector< SmallVector<Value *, 4>, 4> WeakConcurrents;
    SmallVector< SmallVector<Value *, 4>, 4> WeakCommutatives;
    SmallVector< SmallVector<Value *, 4>, 4> VlaDims;
    SmallVector< Value *, 4> Captures;
    Value *Chunksize = nullptr;
    Value *Grainsize = nullptr;
  };

  enum class OmpSsDirectiveKind {
    OSSD_task_for,
    OSSD_taskloop,
    OSSD_taskloop_for
  };

  /// Type used throughout for insertion points.
  using InsertPointTy = IRBuilder<>::InsertPoint;

  /// Callback type for body (=inner region) code generation
  ///
  /// The callback takes code locations as arguments, each describing a
  /// location at which code might need to be generated or a location that is
  /// the target of control transfer.
  ///
  /// \param BeforeRegionIP is the insertion point inside the directive before
  ///                       its region code.
  /// \param ContinuationBB is the basic block target to leave the body.
  ///
  /// Note that all blocks pointed to by the arguments have terminators.
  using BodyGenCallbackTy =
      function_ref<void(InsertPointTy BeforeRegionIP, BasicBlock &ContinuationBB)>;

  /// Description of a LLVM-IR insertion point (IP) and a debug/source location
  /// (filename, line, column, ...).
  struct LocationDescription {
    template <typename T, typename U>
    LocationDescription(const IRBuilder<T, U> &IRB)
        : IP(IRB.saveIP()), DL(IRB.getCurrentDebugLocation()) {}
    LocationDescription(const InsertPointTy &IP, const DebugLoc &DL)
        : IP(IP), DL(DL) {}
    InsertPointTy IP;
    DebugLoc DL;
  };

  /// Generator for loop directives
  ///   '#oss task for'
  ///   '#oss taskloop'
  ///   '#oss taskloop for'
  /// \param Loc The location where the taskwait directive was encountered.
  InsertPointTy createLoop(
    OmpSsDirectiveKind DKind,
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    const DirectiveClausesInfo &DirClauses);

  /// Generator for '#oss task'
  ///
  /// \param Loc The location where the taskwait directive was encountered.
  InsertPointTy createTask(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    const DirectiveClausesInfo &DirClauses);

  /// Generator for '#oss taskwait'
  ///
  /// \param Loc The location where the taskwait directive was encountered.
  void createTaskwait(const LocationDescription &Loc);

  /// Generator for '#oss release'
  ///
  /// \param Loc The location where the release directive was encountered.
  void createRelease(
    const LocationDescription &Loc,
    const DirectiveClausesInfo &DirClauses);

private:

  void emitDirectiveData(
    SmallVector<llvm::OperandBundleDef, 8> &TaskInfo,
    const DirectiveClausesInfo &DirClauses);

  /// Update the internal location to \p Loc.
  bool updateToLocation(const LocationDescription &Loc) {
    Builder.restoreIP(Loc.IP);
    Builder.SetCurrentDebugLocation(Loc.DL);
    return Loc.IP.getBlock() != nullptr;
  }

  /// Generate a taskwait runtime call.
  ///
  /// \param Loc The location at which the request originated and is fulfilled.
  void emitTaskwaitImpl(const LocationDescription &Loc);

  /// The underlying LLVM-IR module
  Module &M;

  /// The LLVM-IR Builder used to create IR.
  IRBuilder<> Builder;
};

} // end namespace llvm

#endif // LLVM_IR_IRBUILDER_H
