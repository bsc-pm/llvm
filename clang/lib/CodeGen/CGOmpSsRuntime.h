//===----- CGOmpSsRuntime.h - Interface to OmpSs Runtimes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OmpSs runtime code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOMPSSRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGOMPSSRUNTIME_H

#include "CGCall.h"
#include "CGValue.h"
#include "clang/AST/Type.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ValueHandle.h"

namespace clang {
class OSSExecutableDirective;

namespace CodeGen {
class Address;
class CodeGenFunction;
class CodeGenModule;

struct OSSDSAPrivateDataTy {
  const Expr *Ref;
  const Expr *Copy;
};

struct OSSDSAFirstprivateDataTy {
  const Expr *Ref;
  const Expr *Copy;
  const Expr *Init;
};

struct OSSTaskDSADataTy final {
  SmallVector<const Expr *, 4> Shareds;
  SmallVector<OSSDSAPrivateDataTy, 4> Privates;
  SmallVector<OSSDSAFirstprivateDataTy, 4> Firstprivates;
};

struct OSSDepDataTy {
  bool OSSSyntax;
  const Expr *E;
};

struct OSSTaskDepDataTy final {
  SmallVector<OSSDepDataTy, 4> WeakIns;
  SmallVector<OSSDepDataTy, 4> WeakOuts;
  SmallVector<OSSDepDataTy, 4> WeakInouts;
  SmallVector<OSSDepDataTy, 4> Ins;
  SmallVector<OSSDepDataTy, 4> Outs;
  SmallVector<OSSDepDataTy, 4> Inouts;
  SmallVector<OSSDepDataTy, 4> Concurrents;
  SmallVector<OSSDepDataTy, 4> Commutatives;
};

struct OSSTaskDataTy final {
  OSSTaskDSADataTy DSAs;
  OSSTaskDepDataTy Deps;
  const Expr *If = nullptr;
  const Expr *Final = nullptr;
  const Expr *Cost = nullptr;
  const Expr *Priority = nullptr;
};

class CGOmpSsRuntime {
protected:
  CodeGenModule &CGM;

private:
  struct TaskContext {
    llvm::AssertingVH<llvm::Instruction> InsertPt = nullptr;
    llvm::BasicBlock *TerminateLandingPad = nullptr;
    llvm::BasicBlock *TerminateHandler = nullptr;
    llvm::BasicBlock *UnreachableBlock = nullptr;
    Address ExceptionSlot = Address::invalid();
    Address EHSelectorSlot = Address::invalid();
    Address NormalCleanupDestSlot = Address::invalid();
    // Map to reuse Addresses emited for data sharings
    llvm::DenseMap<const VarDecl *, Address> RefMap;
  };

  SmallVector<TaskContext, 2> TaskStack;

public:
  explicit CGOmpSsRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGOmpSsRuntime() {};
  virtual void clear() {};

  bool InTaskEmission;

  // This is used to avoid creating the same generic funcion for constructors and
  // destructors, which will be stored in a bundle for each non-pod private/firstprivate
  // data-sharing
  // TODO: try to integrate this better in the class, not as a direct public member
  llvm::DenseMap<const CXXMethodDecl *, llvm::Function *> GenericCXXNonPodMethodDefs;

  // returns true if we're emitting code inside a task context (entry/exit)
  bool inTaskBody();
  // returns the innermost nested task InsertPt instruction
  llvm::AssertingVH<llvm::Instruction> getTaskInsertPt();
  // returns the innermost nested task TerminateHandler BB
  llvm::BasicBlock *getTaskTerminateHandler();
  // returns the innermost nested task TerminateLandingPad BB
  llvm::BasicBlock *getTaskTerminateLandingPad();
  // returns the innermost nested task UnreachableBlock BB
  llvm::BasicBlock *getTaskUnreachableBlock();
  // returns the innermost nested task ExceptionSlot address
  Address getTaskExceptionSlot();
  // returns the innermost nested task EHSelectorSlot address
  Address getTaskEHSelectorSlot();
  // returns the innermost nested task NormalCleanupDestSlot address
  Address getTaskNormalCleanupDestSlot();
  // returns the innermost nested task RefMap
  llvm::DenseMap<const VarDecl *, Address> &getTaskRefMap();

  // sets the innermost nested task InsertPt instruction
  void setTaskInsertPt(llvm::Instruction *I);
  // sets the innermost nested task TerminateHandler instruction
  void setTaskTerminateHandler(llvm::BasicBlock *BB);
  // sets the innermost nested task TerminateLandingPad instruction
  void setTaskTerminateLandingPad(llvm::BasicBlock *BB);
  // sets the innermost nested task UnreachableBlock instruction
  void setTaskUnreachableBlock(llvm::BasicBlock *BB);
  // sets the innermost nested task ExceptionSlot address
  void setTaskExceptionSlot(Address Addr);
  // sets the innermost nested task EHSelectorSlot address
  void setTaskEHSelectorSlot(Address Addr);
  // returns the innermost nested task NormalCleanupDestSlot address
  void setTaskNormalCleanupDestSlot(Address Addr);

  RValue emitTaskFunction(CodeGenFunction &CGF,
                          const FunctionDecl *FD,
                          const CallExpr *CE,
                          ReturnValueSlot ReturnValue);

  /// Emit code for 'taskwait' directive.
  virtual void emitTaskwaitCall(CodeGenFunction &CGF, SourceLocation Loc);
  /// Emit code for 'task' directive.
  virtual void emitTaskCall(CodeGenFunction &CGF,
                            const OSSExecutableDirective &D,
                            SourceLocation Loc,
                            const OSSTaskDataTy &Data);

};

} // namespace CodeGen
} // namespace clang

#endif
