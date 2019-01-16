//===----- CGOmpSsRuntime.cpp - Interface to OmpSs Runtimes -------------===//
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

#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CGOmpSsRuntime.h"
#include "CGRecordLayout.h"
#include "CodeGenFunction.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/Basic/BitmaskEnum.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;

void CGOmpSsRuntime::emitTaskwaitCall(CodeGenFunction &CGF,
                                      SourceLocation Loc) {
  llvm::Value *Callee = CGM.getIntrinsic(llvm::Intrinsic::ompss_marker);
  CGF.Builder.CreateCall(Callee,
                         {},
                         {llvm::OperandBundleDef("kind",
                                                 llvm::ConstantDataArray::getString(CGM.getLLVMContext(),
                                                                                    "taskwait"))});
}

void CGOmpSsRuntime::emitTaskCall(CodeGenFunction &CGF,
                                  const OSSExecutableDirective &D,
                                  SourceLocation Loc,
                                  const OSSTaskDataTy &Data) {
  llvm::Value *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::ompss_region_entry);
  llvm::Value *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::ompss_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back("kind", llvm::ConstantDataArray::getString(CGM.getLLVMContext(), "task"));
  for (const Expr *E : Data.SharedVars) {
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    if (VD->hasLocalStorage()) {
        Address Addr = CGF.GetAddrOfLocalVar(VD);
        TaskInfo.emplace_back("shared", Addr.getPointer());
    }
    else {
        TaskInfo.emplace_back("shared", CGM.GetAddrOfGlobalVar(VD));
    }
  }
  for (const Expr *E : Data.PrivateVars) {
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    if (VD->hasLocalStorage()) {
        Address Addr = CGF.GetAddrOfLocalVar(VD);
        TaskInfo.emplace_back("private", Addr.getPointer());
    }
    else {
        TaskInfo.emplace_back("private", CGM.GetAddrOfGlobalVar(VD));
    }
  }
  for (const Expr *E : Data.FirstprivateVars) {
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    if (VD->hasLocalStorage()) {
        Address Addr = CGF.GetAddrOfLocalVar(VD);
        TaskInfo.emplace_back("firstprivate", Addr.getPointer());
    }
    else {
        TaskInfo.emplace_back("firstprivate", CGM.GetAddrOfGlobalVar(VD));
    }
  }

  llvm::Value *Result =
    CGF.Builder.CreateCall(EntryCallee,
                           {},
                           llvm::makeArrayRef(TaskInfo));

  CGF.EmitStmt(D.getAssociatedStmt());

  CGF.Builder.CreateCall(ExitCallee,
                         Result);
}

