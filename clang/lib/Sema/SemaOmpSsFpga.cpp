//===--- SemaOmpSsFpga.cpp - Semantic Analysis for OmpSs Fpga target ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OmpSs FPGA functions and
/// generates other sources for following steps in the generation
///
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclOmpSs.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprOmpSs.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace clang;

namespace {
QualType DerefOnceTypePointerTo(QualType type) {
  if (type->isPointerType()) {
    return type->getPointeeType().IgnoreParens();
  }
  if (type->isArrayType()) {
    return type->getAsArrayTypeUnsafe()->getElementType().IgnoreParens();
  }
  return type;
}
} // namespace

bool Sema::CheckFpgaLocalmems(FunctionDecl *FD) {
  enum Dir { IN = 0b01, OUT = 0b10, INOUT = 0b11 };
  auto *taskAttr = FD->getAttr<OSSTaskDeclAttr>();
  bool foundError = false;

  // All of the array shapings must have their expressions be computed to
  // constant integer expressions
  auto transformOSSArrayShapingExpr = [&](OSSArrayShapingExpr *shaping) {
    llvm::SmallVector<Expr *, 4> shapes;
    for (auto *shape : shaping->getShapes()) {

      llvm::APSInt valInteger;
      if (auto constantResult = VerifyIntegerConstantExpression(
              const_cast<Expr *>(shape), &valInteger, AllowFold);
          constantResult.isUsable() && valInteger >= 0) {
        shapes.push_back(constantResult.get());
      } else {
        Diag(shape->getExprLoc(), diag::err_expected_constant_unsigned_integer);
        foundError = true;
      }
    }
    if (!foundError) {
      shaping->setShapes(shapes);
    }
  };

  // First, compute the direction tags of the parameters. Do note that not
  // all parameters are guaranteed to be present
  llvm::SmallDenseMap<const ParmVarDecl *,
                      std::pair<OSSArrayShapingExpr *, Dir>>
      currentAssignationsOfArrays;
  auto EmitDepListIterDecls = [&](auto &&DepExprsIter, Dir dir) {
    for (Expr *DepExpr : DepExprsIter) {
      auto *arrShapingExpr = dyn_cast<OSSArrayShapingExpr>(DepExpr);
      if (!arrShapingExpr)
        return;
      if (taskAttr->getCopyDeps())
        transformOSSArrayShapingExpr(arrShapingExpr);
      auto *arrExprBase = dyn_cast<DeclRefExpr>(
          arrShapingExpr->getBase()->IgnoreParenImpCasts());
      assert(arrExprBase);
      auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
      assert(decl);
      auto res = currentAssignationsOfArrays.find(decl);
      if (res != currentAssignationsOfArrays.end()) {
        res->second.second = Dir(res->second.second | dir);
      } else {
        currentAssignationsOfArrays.insert({decl, {arrShapingExpr, dir}});
      };
    }
  };
  EmitDepListIterDecls(taskAttr->ins(), IN);
  EmitDepListIterDecls(taskAttr->outs(), OUT);
  EmitDepListIterDecls(taskAttr->inouts(), INOUT);
  EmitDepListIterDecls(taskAttr->concurrents(), INOUT);
  EmitDepListIterDecls(taskAttr->weakIns(), IN);
  EmitDepListIterDecls(taskAttr->weakOuts(), OUT);
  EmitDepListIterDecls(taskAttr->weakInouts(), INOUT);
  EmitDepListIterDecls(taskAttr->weakConcurrents(), INOUT);
  EmitDepListIterDecls(taskAttr->depIns(), IN);
  EmitDepListIterDecls(taskAttr->depOuts(), OUT);
  EmitDepListIterDecls(taskAttr->depInouts(), INOUT);
  EmitDepListIterDecls(taskAttr->depConcurrents(), INOUT);
  EmitDepListIterDecls(taskAttr->depWeakIns(), IN);
  EmitDepListIterDecls(taskAttr->depWeakOuts(), OUT);
  EmitDepListIterDecls(taskAttr->depWeakInouts(), INOUT);
  EmitDepListIterDecls(taskAttr->depWeakConcurrents(), INOUT);

  // Then compute the list of localmem parameters
  llvm::SmallDenseSet<const ParmVarDecl *> parametersToLocalmem;

  // If copy_deps
  if (taskAttr->getCopyDeps()) {
    for (auto *param : FD->parameters()) {
      if (currentAssignationsOfArrays.find(param) !=
          currentAssignationsOfArrays.end()) {
        parametersToLocalmem.insert(param);
      }
    }
  }
  // If we have an explicit list of localmem (copy_in, copy_out, copy_inout),
  // use that
  auto explicitCopy = [&](auto &&list, Dir dir) {
    for (auto *localmem : list) {
      auto *arrShapingExpr = dyn_cast<OSSArrayShapingExpr>(localmem);
      if (!arrShapingExpr) {
        continue;
      }
      transformOSSArrayShapingExpr(arrShapingExpr);

      auto *arrExprBase = dyn_cast<DeclRefExpr>(
          arrShapingExpr->getBase()->IgnoreParenImpCasts());
      assert(arrExprBase);
      auto *decl = dyn_cast<ParmVarDecl>(arrExprBase->getDecl());
      assert(decl);
      parametersToLocalmem.insert(decl);
      auto &def = currentAssignationsOfArrays[decl];
      def.first = arrShapingExpr;
      def.second = Dir(def.second | dir);
    }
  };
  explicitCopy(taskAttr->copyIn(), IN);
  explicitCopy(taskAttr->copyOut(), OUT);
  explicitCopy(taskAttr->copyInOut(), INOUT);

  // Now check that none of the decl are const qualified while out, and that
  // we know the sizes
  for (auto *param : parametersToLocalmem) {
    if (currentAssignationsOfArrays.find(param)->second.second & OUT &&
        param->getType()->getPointeeType().isConstQualified()) {
      Diag(
          param->getLocation(),
          diag::
              err_oss_fpga_param_used_in_localmem_marked_as_out_const_qualified);
      foundError = true;
    }
    auto paramType = param->getType();
    auto numShapes = currentAssignationsOfArrays.find(param)
                         ->second.first->getShapes()
                         .size();
    for (size_t i = 0; i < numShapes; ++i) {
      paramType = DerefOnceTypePointerTo(paramType);
    }
    while (paramType->isArrayType()) {
      if (!paramType->isConstantArrayType()) {
        Diag(param->getLocation(),
             diag::err_expected_constant_unsigned_integer);
        foundError = true;
      }
      paramType = DerefOnceTypePointerTo(paramType);
    }
    if (paramType->isPointerType()) {
      Diag(param->getLocation(), diag::err_expected_constant_unsigned_integer);
      foundError = true;
    }
  }
  return foundError;
}

bool Sema::ActOnOmpSsDeclareTaskDirectiveWithFpga(Decl *ADecl) {
  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_oss_function_expected);
    return false;
  }

  if (CheckFpgaLocalmems(FD)) {
    return false;
  }

  Context.ompssFpgaDecls.push_back(FD);
  return true;
}