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
};

class CGOmpSsRuntime {
protected:
  CodeGenModule &CGM;

private:
  SmallVector<llvm::AssertingVH<llvm::Instruction>, 2> TaskEntryStack;

public:
  explicit CGOmpSsRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGOmpSsRuntime() {};
  virtual void clear() {};

  // Map to reuse Addresses emited for data sharings
  // TODO: try to integrate this better in the class, not as a direct public member
  llvm::DenseMap<const VarDecl *, Address> RefMap;
  bool InTaskEmission;

  // This is used to avoid creating the same generic funcion for constructors and
  // destructors, which will be stored in a bundle for each non-pod private/firstprivate
  // data-sharing
  // TODO: try to integrate this better in the class, not as a direct public member
  llvm::DenseMap<const CXXMethodDecl *, llvm::Function *> GenericCXXNonPodMethodDefs;

  // returns true if we're emitting code inside a task context (entry/exit)
  bool inTaskBody();
  // returns the innermost nested task entry mark instruction
  llvm::AssertingVH<llvm::Instruction> getCurrentTask();

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
