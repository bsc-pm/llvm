//===--- CGStmtOmpSs.cpp - Emit LLVM Code from Statements ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OmpSs nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGCleanup.h"
#include "CGOmpSsRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOmpSs.h"
#include "llvm/IR/CallSite.h"
using namespace clang;
using namespace CodeGen;

void CodeGenFunction::EmitOSSTaskwaitDirective(const OSSTaskwaitDirective &S) {
  CGM.getOmpSsRuntime().emitTaskwaitCall(*this, S.getBeginLoc());
}

