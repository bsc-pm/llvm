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

struct OSSTaskDataTy final {
  SmallVector<const Expr *, 4> SharedVars;
  SmallVector<const Expr *, 4> PrivateVars;
  SmallVector<const Expr *, 4> FirstprivateVars;
  SmallVector<const Expr *, 4> DependWeakIn;
  SmallVector<const Expr *, 4> DependWeakOut;
  SmallVector<const Expr *, 4> DependWeakInout;
  SmallVector<const Expr *, 4> DependIn;
  SmallVector<const Expr *, 4> DependOut;
  SmallVector<const Expr *, 4> DependInout;
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

  // returns true if we're emitting code inside a task context (entry/exit)
  bool inTask();
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
