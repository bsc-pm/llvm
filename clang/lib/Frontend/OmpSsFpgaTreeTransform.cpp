//===- OmpssFpgaTreeTransform.cpp - Transformations to the AST ------------===//
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

#include "clang/Frontend/OmpSsFpgaTreeTransform.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
namespace {
class OmpSsFpgaTreeTransformVisitor
    : public RecursiveASTVisitor<OmpSsFpgaTreeTransformVisitor> {

  using Inherited = RecursiveASTVisitor<OmpSsFpgaTreeTransformVisitor>;
  ASTContext &Ctx;
  IdentifierTable &IdentifierTable;
  WrapperPortMap &WrapperPortMap;

public:
  OmpSsFpgaTreeTransformVisitor(ASTContext &Ctx,
                                clang::IdentifierTable &IdentifierTable,
                                ::WrapperPortMap &WrapperPortMap)
      : Inherited(), Ctx(Ctx), IdentifierTable(IdentifierTable),
        WrapperPortMap(WrapperPortMap) {}
  bool VisitFunctionDecl(FunctionDecl *funcDecl) {
    llvm::SmallVector<ParmVarDecl *, 4> NewParamInfo(funcDecl->parameters());
    auto spawnInPortType = Ctx.getLValueReferenceType(
        Ctx.getPrintableASTType("hls::stream<ap_uint<8> >"));
    auto &II = IdentifierTable.get("mcxx_spawnInPort");

    /*NewParamInfo.push_back(ParmVarDecl::Create(
        Ctx, funcDecl->getDeclContext(), SourceLocation{}, SourceLocation{},
        &II, spawnInPortType, nullptr, SC_None, nullptr));*/

    const auto *origType = funcDecl->getType()->getAs<FunctionProtoType>();
    llvm::SmallVector<QualType, 4> typesParam;
    typesParam.reserve(NewParamInfo.size());
    for (auto *param : NewParamInfo) {
      typesParam.push_back(param->getType());
    }
    funcDecl->setType(Ctx.getFunctionType(origType->getReturnType(), typesParam,
                                          origType->getExtProtoInfo()));
    funcDecl->resetParams();
    funcDecl->setParams(NewParamInfo);
    return true;
  }
};
} // namespace
namespace clang {
void OmpssFpgaTreeTransform(clang::ASTContext &Ctx,
                            clang::IdentifierTable &identifierTable,
                            WrapperPortMap &WrapperPortMap) {
  OmpSsFpgaTreeTransformVisitor t(Ctx, identifierTable, WrapperPortMap);
  t.TraverseAST(Ctx);
}
} // namespace clang

void FPGAFunctionTreeVisitor::propagatePort(WrapperPort port) {
  const FunctionCallTree *node = Current;
  do {
    wrapperPortMap[node->symbol][(int)port] = true;
    node = node->parent;
  } while (node != nullptr);
}

FPGAFunctionTreeVisitor::FPGAFunctionTreeVisitor(FunctionDecl *startSymbol,
                                                 WrapperPortMap &wrapperPortMap)
    : Top(startSymbol, nullptr), Current(&Top), wrapperPortMap(wrapperPortMap) {
}

bool FPGAFunctionTreeVisitor::VisitOSSTaskDirective(OSSTaskDirective *) {
  CreatesTasks = true;
  propagatePort(WrapperPort::OUTPORT);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitOSSTaskwaitDirective(
    OSSTaskwaitDirective *) {
  CreatesTasks = true;
  propagatePort(WrapperPort::OUTPORT);
  propagatePort(WrapperPort::SPAWN_INPORT);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitCXXConstructExpr(CXXConstructExpr *n) {
  auto *body = n->getConstructor()->getBody();
  getDerived().VisitStmt(body);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitOMPCriticalDirective(
    OMPCriticalDirective *n) {
  for (auto *c : n->children())
    getDerived().VisitStmt(c);

  UsesLock = true;
  propagatePort(WrapperPort::INPORT);
  propagatePort(WrapperPort::OUTPORT);
  return true;
}

bool FPGAFunctionTreeVisitor::VisitCallExpr(CallExpr *n) {
  for (auto *arg : n->children()) {
    getDerived().VisitStmt(arg);
  }
  auto *sym = n->getCalleeDecl();
  if (const NamedDecl *symNamed = dyn_cast<NamedDecl>(sym); symNamed) {
    auto symName = symNamed->getName();

    if (symName == "OMPIF_Comm_rank") {
      UsesOmpif = true;
      propagatePort(WrapperPort::OMPIF_RANK);
      return true;
    }
    if (symName == "OMPIF_Comm_size") {
      UsesOmpif = true;
      propagatePort(WrapperPort::OMPIF_SIZE);
      return true;
    }
    if (symName.startswith("OMPIF_")) {
      UsesOmpif = CreatesTasks = true;
      propagatePort(WrapperPort::OUTPORT);
      if (symName == "OMPIF_Allgather") {
        propagatePort(WrapperPort::OMPIF_RANK);
        propagatePort(WrapperPort::SPAWN_INPORT);
      }
    } else if (symName == "nanos6_fpga_memcpy_wideport_in" ||
               symName == "nanos6_fpga_memcpy_wideport_out") {
      MemcpyWideport = true;
      propagatePort(WrapperPort::MEMORY_PORT);
      return true;
    }
  }
  auto *body = sym->getBody();
  if (!body) {
    return true;
  }

  FunctionCallTree newNode(sym, Current);
  FunctionCallTree *prev = Current;
  Current = &newNode;
  getDerived().VisitStmt(body);
  Current = prev;
  return true;
}
