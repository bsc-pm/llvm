//===--- StmtOmpSs.cpp - Classes for OmpSs directives -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in StmtOmpSs.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtOmpSs.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

OSSTaskwaitDirective *OSSTaskwaitDirective::Create(const ASTContext &C,
                                                   SourceLocation StartLoc,
                                                   SourceLocation EndLoc) {
  void *Mem = C.Allocate(sizeof(OSSTaskwaitDirective));
  OSSTaskwaitDirective *Dir = new (Mem) OSSTaskwaitDirective(StartLoc, EndLoc);
  return Dir;
}

OSSTaskwaitDirective *OSSTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                        EmptyShell) {
  void *Mem = C.Allocate(sizeof(OSSTaskwaitDirective));
  return new (Mem) OSSTaskwaitDirective();
}
