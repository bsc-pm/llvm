//===- OmpssFpgaTreeTransform.h - Transformations to the AST ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the AST transformations necessary for producing a valid
/// wrapper.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_OMPSSFPGATREETRANSFORM_H
#define LLVM_CLANG_FRONTEND_OMPSSFPGATREETRANSFORM_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/IdentifierTable.h"
namespace clang {

enum class WrapperPort {
  OMPIF_RANK = 0,
  OMPIF_SIZE = 1,
  MEMORY_PORT = 2,
  INPORT = 3,
  OUTPORT = 4,
  SPAWN_INPORT = 5,
  NUM_PORTS = 6
};

using WrapperPortMap =
    llvm::SmallMapVector<const Decl *,
                         std::array<bool, size_t(WrapperPort::NUM_PORTS)>, 16>;

class FPGAFunctionTreeVisitor
    : public RecursiveASTVisitor<FPGAFunctionTreeVisitor> {

  struct FunctionCallTree {
    const Decl *symbol;
    FunctionCallTree *parent;

    FunctionCallTree(const Decl *symbol, FunctionCallTree *parent)
        : symbol(symbol), parent(parent) {}
  };

  FunctionCallTree Top;
  FunctionCallTree *Current;
  WrapperPortMap &wrapperPortMap;

  void propagatePort(WrapperPort port);

public:
  bool CreatesTasks = false;
  bool UsesOmpif = false;
  bool MemcpyWideport = false;
  bool UsesLock = false;

  FPGAFunctionTreeVisitor(FunctionDecl *startSymbol,
                          WrapperPortMap &wrapperPortMap);

  bool VisitOSSTaskDirective(OSSTaskDirective *);
  bool VisitOSSTaskwaitDirective(OSSTaskwaitDirective *);
  bool VisitCXXConstructExpr(CXXConstructExpr *n);
  bool VisitOMPCriticalDirective(OMPCriticalDirective *n);
  bool VisitCallExpr(CallExpr *n);
};

void OmpssFpgaTreeTransform(clang::ASTContext &Ctx,
                            clang::IdentifierTable &identifierTable,
                            FunctionDecl *FD);
} // namespace clang

#endif