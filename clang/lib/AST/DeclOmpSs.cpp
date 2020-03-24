//===--- DeclOmpSs.cpp - Declaration OmpSs AST Node Implementation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements OSSDeclareReductionDecl
/// classes.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclOmpSs.h"
#include "clang/AST/Expr.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// OSSDeclareReductionDecl Implementation.
//===----------------------------------------------------------------------===//

OSSDeclareReductionDecl::OSSDeclareReductionDecl(
    Kind DK, DeclContext *DC, SourceLocation L, DeclarationName Name,
    QualType Ty, OSSDeclareReductionDecl *PrevDeclInScope)
    : ValueDecl(DK, DC, L, Name, Ty), DeclContext(DK), Combiner(nullptr),
      PrevDeclInScope(PrevDeclInScope) {
  setInitializer(nullptr, CallInit);
}

void OSSDeclareReductionDecl::anchor() {}

OSSDeclareReductionDecl *OSSDeclareReductionDecl::Create(
    ASTContext &C, DeclContext *DC, SourceLocation L, DeclarationName Name,
    QualType T, OSSDeclareReductionDecl *PrevDeclInScope) {
  return new (C, DC) OSSDeclareReductionDecl(OSSDeclareReduction, DC, L, Name,
                                             T, PrevDeclInScope);
}

OSSDeclareReductionDecl *
OSSDeclareReductionDecl::CreateDeserialized(ASTContext &C, unsigned ID) {
  return new (C, ID) OSSDeclareReductionDecl(
      OSSDeclareReduction, /*DC=*/nullptr, SourceLocation(), DeclarationName(),
      QualType(), /*PrevDeclInScope=*/nullptr);
}

OSSDeclareReductionDecl *OSSDeclareReductionDecl::getPrevDeclInScope() {
  return cast_or_null<OSSDeclareReductionDecl>(
      PrevDeclInScope.get(getASTContext().getExternalSource()));
}
const OSSDeclareReductionDecl *
OSSDeclareReductionDecl::getPrevDeclInScope() const {
  return cast_or_null<OSSDeclareReductionDecl>(
      PrevDeclInScope.get(getASTContext().getExternalSource()));
}

