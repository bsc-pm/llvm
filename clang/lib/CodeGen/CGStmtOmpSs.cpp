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

template<typename DSAKind>
static void AddDSAData(const OSSTaskDirective &S, SmallVectorImpl<const Expr *> &Data) {
  // All DSA are DeclRefExpr
  llvm::SmallSet<const ValueDecl *, 8> DeclExpr;
  for (const auto *C : S.getClausesOfKind<DSAKind>()) {
      for (const Expr *Ref : C->varlists()) {
        if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Ref)) {
          const ValueDecl *VD = DRE->getDecl();
          if (DeclExpr.insert(VD).second)
            Data.push_back(Ref);
        }
      }
  }
}

void CodeGenFunction::EmitOSSTaskDirective(const OSSTaskDirective &S) {
  OSSTaskDataTy Data;

  AddDSAData<OSSSharedClause>(S, Data.SharedVars);
  AddDSAData<OSSPrivateClause>(S, Data.PrivateVars);
  AddDSAData<OSSFirstprivateClause>(S, Data.FirstprivateVars);

  for (const auto *C : S.getClausesOfKind<OSSDependClause>()) {
    ArrayRef<OmpSsDependClauseKind> DepKinds = C->getDependencyKind();
    if (DepKinds.size() == 2) {
      for (const Expr *Ref : C->varlists()) {
        if (DepKinds[0] == OSSC_DEPEND_in
            || DepKinds[1] == OSSC_DEPEND_in)
          Data.DependWeakIn.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_out
            || DepKinds[1] == OSSC_DEPEND_out)
          Data.DependWeakOut.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_inout
            || DepKinds[1] == OSSC_DEPEND_inout)
          Data.DependWeakInout.push_back(Ref);
      }
    }
    else {
      for (const Expr *Ref : C->varlists()) {
        if (DepKinds[0] == OSSC_DEPEND_in)
          Data.DependIn.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_out)
          Data.DependOut.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_inout)
          Data.DependInout.push_back(Ref);
      }
    }
  }

  CGM.getOmpSsRuntime().emitTaskCall(*this, S, S.getBeginLoc(), Data);
}

