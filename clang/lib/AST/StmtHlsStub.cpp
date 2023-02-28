//===--- StmtHlsStub.cpp - Classes for HLS directives --------------------===//
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

#include "clang/AST/StmtHlsStub.h"
#include "llvm/Support/TrailingObjects.h"
using namespace clang;

HlsDirective::HlsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                           StringRef &&content,
                           llvm::ArrayRef<const Expr *> &&arrayExpr)
    : Stmt(StmtClass::HlsDirectiveClass), StartLoc(StartLoc), EndLoc(EndLoc),
      numChars(content.size()), numExprs(arrayExpr.size()) {
  std::copy(content.begin(), content.end(), getTrailingObjects<char>());
  std::copy(arrayExpr.begin(), arrayExpr.end(), getTrailingObjects<const Expr *>());
}