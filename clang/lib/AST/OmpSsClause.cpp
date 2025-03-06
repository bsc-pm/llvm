//===- OmpSsClause.cpp - Classes for OmpSs clauses ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclesses of Stmt class declared in OmpSsClause.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OmpSsClause.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>

using namespace clang;
using namespace llvm;
using namespace oss;

OSSClause::child_range OSSClause::children() {
  switch (getClauseKind()) {
  default:
    break;
#define GEN_CLANG_CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class)                                         \
  case Enum:                                                                   \
    return static_cast<Class *>(this)->children();
#include "llvm/Frontend/OmpSs/OSS.inc"
  }
  llvm_unreachable("unknown OSSClause");
}

OSSLabelClause *OSSLabelClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSLabelClause *Clause =
      new (Mem) OSSLabelClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSLabelClause *OSSLabelClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSLabelClause(N);
}

void OSSPrivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

OSSPrivateClause *OSSPrivateClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL,
                                         ArrayRef<Expr *> PrivateVL) {
  // Info of item 'i' is in VL[i], PrivateVL[i]
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(2 * VL.size()));
  OSSPrivateClause *Clause =
      new (Mem) OSSPrivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  return Clause;
}

OSSPrivateClause *OSSPrivateClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSPrivateClause(N);
}

void OSSFirstprivateClause::setPrivateCopies(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of private copies is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), varlist_end());
}

void OSSFirstprivateClause::setInits(ArrayRef<Expr *> VL) {
  assert(VL.size() == varlist_size() &&
         "Number of inits is not the same as the preallocated buffer");
  std::copy(VL.begin(), VL.end(), getPrivateCopies().end());
}

OSSFirstprivateClause *OSSFirstprivateClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL,
                                         ArrayRef<Expr *> PrivateVL,
                                         ArrayRef<Expr *> InitVL) {
  // Info of item 'i' is in VL[i], PrivateVL[i] and InitVL[i]
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(3 * VL.size()));
  OSSFirstprivateClause *Clause =
      new (Mem) OSSFirstprivateClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  Clause->setPrivateCopies(PrivateVL);
  Clause->setInits(InitVL);
  return Clause;
}

OSSFirstprivateClause *OSSFirstprivateClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSFirstprivateClause(N);
}

OSSSharedClause *OSSSharedClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSSharedClause *Clause =
      new (Mem) OSSSharedClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSSharedClause *OSSSharedClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSSharedClause(N);
}

OSSDependClause *
OSSDependClause::Create(const ASTContext &C, SourceLocation StartLoc,
                        SourceLocation LParenLoc, SourceLocation EndLoc,
                        ArrayRef<OmpSsDependClauseKind> DepKinds,
                        ArrayRef<OmpSsDependClauseKind> DepKindsOrdered,
                        SourceLocation DepLoc, SourceLocation ColonLoc,
                        ArrayRef<Expr *> VL, bool OSSSyntax) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSDependClause *Clause = new (Mem)
      OSSDependClause(StartLoc, LParenLoc, EndLoc, VL.size(), OSSSyntax);
  Clause->setVarRefs(VL);
  Clause->setDependencyKinds(DepKinds);
  Clause->setDependencyKindsOrdered(DepKindsOrdered);
  Clause->setDependencyLoc(DepLoc);
  Clause->setColonLoc(ColonLoc);
  return Clause;
}

OSSDependClause *OSSDependClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSDependClause(N);
}

void OSSReductionClause::setLHSExprs(ArrayRef<Expr *> LHSExprs) {
  assert(
      LHSExprs.size() == varlist_size() &&
      "Number of LHS expressions is not the same as the preallocated buffer");
  std::copy(LHSExprs.begin(), LHSExprs.end(), varlist_end());
}

void OSSReductionClause::setRHSExprs(ArrayRef<Expr *> RHSExprs) {
  assert(
      RHSExprs.size() == varlist_size() &&
      "Number of RHS expressions is not the same as the preallocated buffer");
  std::copy(RHSExprs.begin(), RHSExprs.end(), getLHSExprs().end());
}

void OSSReductionClause::setReductionOps(ArrayRef<Expr *> ReductionOps) {
  assert(ReductionOps.size() == varlist_size() && "Number of reduction "
                                                  "expressions is not the same "
                                                  "as the preallocated buffer");
  std::copy(ReductionOps.begin(), ReductionOps.end(), getRHSExprs().end());
}

OSSReductionClause *OSSReductionClause::Create(
    const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation EndLoc, SourceLocation ColonLoc, ArrayRef<Expr *> VL,
    NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
    ArrayRef<Expr *> LHSExprs,
    ArrayRef<Expr *> RHSExprs, ArrayRef<Expr *> ReductionOps,
    ArrayRef<BinaryOperatorKind> ReductionKinds, bool IsWeak) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * VL.size()));
  OSSReductionClause *Clause = new (Mem) OSSReductionClause(
      StartLoc, LParenLoc, EndLoc, ColonLoc, VL.size(), QualifierLoc, NameInfo, IsWeak);
  Clause->setVarRefs(VL);
  Clause->setLHSExprs(LHSExprs);
  Clause->setRHSExprs(RHSExprs);
  Clause->setReductionOps(ReductionOps);
  Clause->ReductionKinds.append(ReductionKinds.begin(), ReductionKinds.end());
  return Clause;
}

OSSReductionClause *OSSReductionClause::CreateEmpty(const ASTContext &C,
                                                    unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(5 * N));
  return new (Mem) OSSReductionClause(N);
}

OSSNdrangeClause *OSSNdrangeClause::Create(const ASTContext &C,
                                         SourceLocation StartLoc,
                                         SourceLocation LParenLoc,
                                         SourceLocation EndLoc,
                                         ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSNdrangeClause *Clause =
      new (Mem) OSSNdrangeClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSNdrangeClause *OSSNdrangeClause::CreateEmpty(const ASTContext &C, unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSNdrangeClause(N);
}

OSSGridClause *OSSGridClause::Create(const ASTContext &C,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc,
                                     ArrayRef<Expr *> VL) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(VL.size()));
  OSSGridClause *Clause =
      new (Mem) OSSGridClause(StartLoc, LParenLoc, EndLoc, VL.size());
  Clause->setVarRefs(VL);
  return Clause;
}

OSSGridClause *OSSGridClause::CreateEmpty(const ASTContext &C,
                                                unsigned N) {
  void *Mem = C.Allocate(totalSizeToAlloc<Expr *>(N));
  return new (Mem) OSSGridClause(N);
}

//===----------------------------------------------------------------------===//
//  OmpSs clauses printing methods
//===----------------------------------------------------------------------===//

template<typename T>
void OSSClausePrinter::VisitOSSClauseList(T *Node, char StartSym) {
  for (typename T::varlist_iterator I = Node->varlist_begin(),
                                    E = Node->varlist_end();
       I != E; ++I) {
    assert(*I && "Expected non-null Stmt");
    OS << (I == Node->varlist_begin() ? StartSym : ',');
    if (auto *DRE = dyn_cast<DeclRefExpr>(*I)) {
        DRE->getDecl()->printQualifiedName(OS);
    } else
      (*I)->printPretty(OS, nullptr, Policy, 0);
  }
}

void OSSClausePrinter::VisitOSSImmediateClause(OSSImmediateClause *Node) {
  OS << "immediate(";
  Node->getCondition()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSMicrotaskClause(OSSMicrotaskClause *Node) {
  OS << "microtask(";
  Node->getCondition()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSIfClause(OSSIfClause *Node) {
  OS << "if(";
  Node->getCondition()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSFinalClause(OSSFinalClause *Node) {
  OS << "final(";
  Node->getCondition()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSCostClause(OSSCostClause *Node) {
  OS << "cost(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSPriorityClause(OSSPriorityClause *Node) {
  OS << "priority(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSLabelClause(OSSLabelClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "label";
    VisitOSSClauseList(Node, '(');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSChunksizeClause(OSSChunksizeClause *Node) {
  OS << "chunksize(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSGrainsizeClause(OSSGrainsizeClause *Node) {
  OS << "grainsize(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSUnrollClause(OSSUnrollClause *Node) {
  OS << "unroll(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSCollapseClause(OSSCollapseClause *Node) {
  OS << "collapse(";
  Node->getNumForLoops()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSWaitClause(OSSWaitClause *Node) {
  OS << "wait";
}

void OSSClausePrinter::VisitOSSUpdateClause(OSSUpdateClause *Node) {
  OS << "update";
}

void OSSClausePrinter::VisitOSSShmemClause(OSSShmemClause *Node) {
  OS << "shmem(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSOnreadyClause(OSSOnreadyClause *Node) {
  OS << "onready(";
  Node->getExpression()->printPretty(OS, nullptr, Policy, 0);
  OS << ")";
}

void OSSClausePrinter::VisitOSSDefaultClause(OSSDefaultClause *Node) {
  OS << "default("
     << getOmpSsSimpleClauseTypeName(OSSC_default, unsigned(Node->getDefaultKind()))
     << ")";
}

void OSSClausePrinter::VisitOSSPrivateClause(OSSPrivateClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "private";
    VisitOSSClauseList(Node, '(');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSFirstprivateClause(OSSFirstprivateClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "firstprivate";
    VisitOSSClauseList(Node, '(');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSSharedClause(OSSSharedClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "shared";
    VisitOSSClauseList(Node, '(');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSDependClause(OSSDependClause *Node) {
  if (Node->isOSSSyntax()) {
    OS << getOmpSsSimpleClauseTypeName(Node->getClauseKind(),
                                       Node->getDependencyKinds()[0]);
    if (!Node->varlist_empty()) {
      VisitOSSClauseList(Node, '(');
    }
  } else {
    OS << "depend(";
    OS << getOmpSsSimpleClauseTypeName(Node->getClauseKind(),
                                       Node->getDependencyKinds()[0]);
    if (Node->getDependencyKinds().size() == 2) {
      OS << " ,";
      OS << getOmpSsSimpleClauseTypeName(Node->getClauseKind(),
                                         Node->getDependencyKinds()[1]);
    }
    if (!Node->varlist_empty()) {
      OS << " :";
      VisitOSSClauseList(Node, ' ');
    }
  }

  OS << ")";
}

void OSSClausePrinter::VisitOSSReductionClause(OSSReductionClause *Node) {
  if (!Node->varlist_empty()) {
    OS << (Node->isWeak() ? "weakreduction" : "reduction") << "(";
    NestedNameSpecifier *QualifierLoc =
        Node->getQualifierLoc().getNestedNameSpecifier();
    OverloadedOperatorKind OOK =
        Node->getNameInfo().getName().getCXXOverloadedOperator();
    if (QualifierLoc == nullptr && OOK != OO_None) {
      // Print reduction identifier in C format
      OS << getOperatorSpelling(OOK);
    } else {
      // Use C++ format
      if (QualifierLoc != nullptr)
        QualifierLoc->print(OS, Policy);
      OS << Node->getNameInfo();
    }
    OS << ":";
    VisitOSSClauseList(Node, ' ');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSDeviceClause(OSSDeviceClause *Node) {
  OS << "device("
     << getOmpSsSimpleClauseTypeName(OSSC_device, Node->getDeviceKind())
     << ")";
}

void OSSClausePrinter::VisitOSSNdrangeClause(OSSNdrangeClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "ndrange";
    VisitOSSClauseList(Node, '(');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSGridClause(OSSGridClause *Node) {
  if (!Node->varlist_empty()) {
    OS << "grid";
    VisitOSSClauseList(Node, '(');
    OS << ")";
  }
}

void OSSClausePrinter::VisitOSSReadClause(OSSReadClause *Node) { OS << "read"; }

void OSSClausePrinter::VisitOSSWriteClause(OSSWriteClause *Node) { OS << "write"; }

void OSSClausePrinter::VisitOSSCaptureClause(OSSCaptureClause *Node) {
  OS << "capture";
}

void OSSClausePrinter::VisitOSSCompareClause(OSSCompareClause *Node) {
  OS << "compare";
}

void OSSClausePrinter::VisitOSSSeqCstClause(OSSSeqCstClause *Node) {
  OS << "seq_cst";
}

void OSSClausePrinter::VisitOSSAcqRelClause(OSSAcqRelClause *Node) {
  OS << "acq_rel";
}

void OSSClausePrinter::VisitOSSAcquireClause(OSSAcquireClause *Node) {
  OS << "acquire";
}

void OSSClausePrinter::VisitOSSReleaseClause(OSSReleaseClause *Node) {
  OS << "release";
}

void OSSClausePrinter::VisitOSSRelaxedClause(OSSRelaxedClause *Node) {
  OS << "relaxed";
}
