//===----- CGOmpSsRuntime.cpp - Interface to OmpSs Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "ConstantEmitter.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/AST/Decl.h"
#include "clang/AST/OmpSsClause.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/BitmaskEnum.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsOmpSs.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::oss;

namespace {

enum OmpSsBundleKind {
  OSSB_directive,
  OSSB_task,
  OSSB_critical_start,
  OSSB_critical_end,
  OSSB_task_for,
  OSSB_taskiter_for,
  OSSB_taskiter_while,
  OSSB_taskloop,
  OSSB_taskloop_for,
  OSSB_taskwait,
  OSSB_release,
  OSSB_shared,
  OSSB_private,
  OSSB_firstprivate,
  OSSB_if,
  OSSB_final,
  OSSB_cost,
  OSSB_priority,
  OSSB_label,
  OSSB_chunksize,
  OSSB_grainsize,
  OSSB_unroll,
  OSSB_wait,
  OSSB_update,
  OSSB_device,
  OSSB_device_ndrange,
  OSSB_device_dev_func,
  OSSB_device_call_order,
  OSSB_device_shmem,
  OSSB_onready,
  OSSB_while_cond,
  OSSB_in,
  OSSB_out,
  OSSB_inout,
  OSSB_concurrent,
  OSSB_commutative,
  OSSB_reduction,
  OSSB_multi_in,
  OSSB_multi_out,
  OSSB_multi_inout,
  OSSB_multi_concurrent,
  OSSB_multi_commutative,
  OSSB_redinit,
  OSSB_redcomb,
  OSSB_weakin,
  OSSB_weakout,
  OSSB_weakinout,
  OSSB_weakconcurrent,
  OSSB_weakcommutative,
  OSSB_weakreduction,
  OSSB_multi_weakin,
  OSSB_multi_weakout,
  OSSB_multi_weakinout,
  OSSB_multi_weakconcurrent,
  OSSB_multi_weakcommutative,
  OSSB_init,
  OSSB_copy,
  OSSB_deinit,
  OSSB_vladims,
  OSSB_captured,
  OSSB_loop_type,
  OSSB_loop_indvar,
  OSSB_loop_lowerbound,
  OSSB_loop_upperbound,
  OSSB_loop_step,
  OSSB_decl_source,
  OSSB_unknown
};

const char *getBundleStr(OmpSsBundleKind Kind) {
  switch (Kind) {
  case OSSB_directive:
    return "DIR.OSS";
  case OSSB_task:
    return "TASK";
  case OSSB_critical_start:
    return "CRITICAL.START";
  case OSSB_critical_end:
    return "CRITICAL.END";
  case OSSB_task_for:
    return "TASK.FOR";
  case OSSB_taskiter_for:
    return "TASKITER.FOR";
  case OSSB_taskiter_while:
    return "TASKITER.WHILE";
  case OSSB_taskloop:
    return "TASKLOOP";
  case OSSB_taskloop_for:
    return "TASKLOOP.FOR";
  case OSSB_taskwait:
    return "TASKWAIT";
  case OSSB_release:
    return "RELEASE";
  case OSSB_shared:
    return "QUAL.OSS.SHARED";
  case OSSB_private:
    return "QUAL.OSS.PRIVATE";
  case OSSB_firstprivate:
    return "QUAL.OSS.FIRSTPRIVATE";
  case OSSB_if:
    return "QUAL.OSS.IF";
  case OSSB_final:
    return "QUAL.OSS.FINAL";
  case OSSB_cost:
    return "QUAL.OSS.COST";
  case OSSB_priority:
    return "QUAL.OSS.PRIORITY";
  case OSSB_label:
    return "QUAL.OSS.LABEL";
  case OSSB_chunksize:
    return "QUAL.OSS.LOOP.CHUNKSIZE";
  case OSSB_grainsize:
    return "QUAL.OSS.LOOP.GRAINSIZE";
  case OSSB_unroll:
    return "QUAL.OSS.LOOP.UNROLL";
  case OSSB_wait:
    return "QUAL.OSS.WAIT";
  case OSSB_update:
    return "QUAL.OSS.LOOP.UPDATE";
  case OSSB_device:
    return "QUAL.OSS.DEVICE";
  case OSSB_device_ndrange:
    return "QUAL.OSS.DEVICE.NDRANGE";
  case OSSB_device_dev_func:
    return "QUAL.OSS.DEVICE.DEVFUNC";
  case OSSB_device_call_order:
    return "QUAL.OSS.DEVICE.CALL.ORDER";
  case OSSB_device_shmem:
    return "QUAL.OSS.DEVICE.SHMEM";
  case OSSB_onready:
    return "QUAL.OSS.ONREADY";
  case OSSB_while_cond:
    return "QUAL.OSS.WHILE.COND";
  case OSSB_in:
    return "QUAL.OSS.DEP.IN";
  case OSSB_out:
    return "QUAL.OSS.DEP.OUT";
  case OSSB_inout:
    return "QUAL.OSS.DEP.INOUT";
  case OSSB_concurrent:
    return "QUAL.OSS.DEP.CONCURRENT";
  case OSSB_commutative:
    return "QUAL.OSS.DEP.COMMUTATIVE";
  case OSSB_reduction:
    return "QUAL.OSS.DEP.REDUCTION";
  case OSSB_multi_in:
    return "QUAL.OSS.MULTIDEP.RANGE.IN";
  case OSSB_multi_out:
    return "QUAL.OSS.MULTIDEP.RANGE.OUT";
  case OSSB_multi_inout:
    return "QUAL.OSS.MULTIDEP.RANGE.INOUT";
  case OSSB_multi_concurrent:
    return "QUAL.OSS.MULTIDEP.RANGE.CONCURRENT";
  case OSSB_multi_commutative:
    return "QUAL.OSS.MULTIDEP.RANGE.COMMUTATIVE";
  case OSSB_redinit:
    return "QUAL.OSS.DEP.REDUCTION.INIT";
  case OSSB_redcomb:
    return "QUAL.OSS.DEP.REDUCTION.COMBINE";
  case OSSB_weakin:
    return "QUAL.OSS.DEP.WEAKIN";
  case OSSB_weakout:
    return "QUAL.OSS.DEP.WEAKOUT";
  case OSSB_weakinout:
    return "QUAL.OSS.DEP.WEAKINOUT";
  case OSSB_weakconcurrent:
    return "QUAL.OSS.DEP.WEAKCONCURRENT";
  case OSSB_weakcommutative:
    return "QUAL.OSS.DEP.WEAKCOMMUTATIVE";
  case OSSB_weakreduction:
    return "QUAL.OSS.DEP.WEAKREDUCTION";
  case OSSB_multi_weakin:
    return "QUAL.OSS.MULTIDEP.RANGE.WEAKIN";
  case OSSB_multi_weakout:
    return "QUAL.OSS.MULTIDEP.RANGE.WEAKOUT";
  case OSSB_multi_weakinout:
    return "QUAL.OSS.MULTIDEP.RANGE.WEAKINOUT";
  case OSSB_multi_weakconcurrent:
    return "QUAL.OSS.MULTIDEP.RANGE.WEAKCONCURRENT";
  case OSSB_multi_weakcommutative:
    return "QUAL.OSS.MULTIDEP.RANGE.WEAKCOMMUTATIVE";
  case OSSB_init:
    return "QUAL.OSS.INIT";
  case OSSB_copy:
    return "QUAL.OSS.COPY";
  case OSSB_deinit:
    return "QUAL.OSS.DEINIT";
  case OSSB_vladims:
    return "QUAL.OSS.VLA.DIMS";
  case OSSB_captured:
    return "QUAL.OSS.CAPTURED";
  case OSSB_loop_type:
    return "QUAL.OSS.LOOP.TYPE";
  case OSSB_loop_indvar:
    return "QUAL.OSS.LOOP.IND.VAR";
  case OSSB_loop_lowerbound:
    return "QUAL.OSS.LOOP.LOWER.BOUND";
  case OSSB_loop_upperbound:
    return "QUAL.OSS.LOOP.UPPER.BOUND";
  case OSSB_loop_step:
    return "QUAL.OSS.LOOP.STEP";
  case OSSB_decl_source:
    return "QUAL.OSS.DECL.SOURCE";
  default:
    llvm_unreachable("Invalid OmpSs bundle kind");
  }
}

} // namespace

namespace {
// This class gathers all the info needed for OSSDependVisitor to emit a dependency in
// compute_dep.
// Also its used to visit init/ub/step expressions in multideps
// The visitor walks and annotates the number of dimensions in the same way that OSSDependVisitor
// NOTE: keep synchronized
//
// Additionally it gathers:
// 1. all the shape expressions involved variables. Since a shaping expression may be a VLA, it's dimensions
// have not been built yet. We need the involved variable to be able to emit them in compute_dep
// 2. all the vla size expressions. We may have 'vla[sizeof(vla)]' which needs all the dimensions for example
// 3. 'this'
class OSSExprInfoGathering
  : public ConstStmtVisitor<OSSExprInfoGathering, void> {

  CodeGenFunction &CGF;
  QualType OSSArgTy;

  // List of vars (not references or globals)
  // involved in the dependency
  llvm::MapVector<const VarDecl *, LValue> ExprInvolvedVarList;
  // Map of VLASizeExpr involved in the dependency.
  llvm::MapVector<const Expr *, llvm::Value *> VLASizeInvolvedMap;
  // Map of captures involved in the dependency.
  // i.e. references and global variables
  llvm::DenseMap<const VarDecl *, Address> CaptureInvolvedMap;

  SmallVector<QualType, 4> RetTypes;

  llvm::Value *Base = nullptr;
  bool HasThis = false;

  void AddDimStartEnd() {
    RetTypes.push_back(OSSArgTy);
    RetTypes.push_back(OSSArgTy);
    RetTypes.push_back(OSSArgTy);
  }

public:

  OSSExprInfoGathering(CodeGenFunction &CGF)
    : CGF(CGF), OSSArgTy(CGF.getContext().LongTy)
      {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  // This is used for DeclRefExpr, MemberExpr and Unary Deref
  void FillBaseExprDims(const Expr *E) {
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType() || !TmpTy->isArrayType()) {
      // T * || T
      AddDimStartEnd();
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  // This is used in the innermost Expr * in ArraySubscripts and OSSArraySection. Also in OSSArrayShaping
  void FillDimsFromInnermostExpr(const Expr *E) {
    // Go through the expression which may be a DeclRefExpr or MemberExpr or OSSArrayShapingExpr
    E = E->IgnoreParenImpCasts();
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType()) {
      // T *
      // We have added section dimension before
      if (RetTypes.empty())
        AddDimStartEnd();
      TmpTy = TmpTy->getPointeeType();
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        AddDimStartEnd();
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  // Walk down into the type and look for VLA expressions.
  void FillTypeVLASizes(const Expr *E) {
    QualType type = E->getType().getCanonicalType();
    while (type->isVariablyModifiedType()) {
      if (type->isPointerType()) {
        type = type->getPointeeType();
      } else if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(type)) {
        type = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(type)) {
        const Expr *VLASizeExpr = BaseArrayTy->getSizeExpr();
        auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
        VLASizeInvolvedMap[VLASizeExpr] = VlaSize.NumElts;

        type = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  void VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
    if (RetTypes.empty()) {
      // We want the original type, not the shaping expr. type
      RetTypes.push_back(E->getBase()->getType());

      // Annotate dims
      FillDimsFromInnermostExpr(E);
    }

    Visit(E->getBase());

    // Annotate all the variables involved in shape expr. We will
    // emit its VLASizeExpr in compute_dep context
    for (const Expr *S : E->getShapes())
      Visit(S);
  }

  void VisitDeclRefExpr(const DeclRefExpr *E) {
    if (E->isNonOdrUse() == NOUR_Unevaluated)
      return;

    CodeGenFunction::ConstantEmission CE = CGF.tryEmitAsConstant(const_cast<DeclRefExpr*>(E));
    if (CE && !CE.isReference())
      // Constant value, no need to annotate it.
      return;

    if (RetTypes.empty()) {
      RetTypes.push_back(CGF.getContext().getPointerType(E->getType()));
      FillBaseExprDims(E);
    }
    FillTypeVLASizes(E);

    llvm::Value *PossibleBase = nullptr;
    if (const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      if (VD->getType()->isReferenceType()
          || E->refersToEnclosingVariableOrCapture()) {
        // Reuse the ref addr. got in DSA
        LValue LV = CGF.EmitDeclRefLValue(E);
        PossibleBase = LV.getPointer(CGF);
        CaptureInvolvedMap.try_emplace(VD, LV.getAddress(CGF));
      } else if (VD->hasLinkage() || VD->isStaticDataMember()) {
        PossibleBase = CGF.CGM.GetAddrOfGlobalVar(VD);
        CharUnits Alignment = CGF.getContext().getDeclAlign(VD);
        Address Addr(PossibleBase, CGF.ConvertType(VD->getType()), Alignment);
        CaptureInvolvedMap.try_emplace(VD, Addr);
      } else {
        LValue LV = CGF.EmitDeclRefLValue(E);
        PossibleBase = LV.getPointer(CGF);
        ExprInvolvedVarList[VD] = LV;
      }
      // Since we prioritize visiting the base of the expression
      // the first DeclRefExpr is always the Base
      if (!Base)
        Base = PossibleBase;
    }
  }

  void VisitLambdaExpr(const LambdaExpr *LE) {
    for (const auto &Capture : LE->capture_inits()) {
      Visit(Capture);
    }
  }

  void VisitCXXThisExpr(const CXXThisExpr *ThisE) {
    HasThis = true;
    Address CXXThisAddress = CGF.LoadCXXThisAddress();
    if (!Base)
      Base = CXXThisAddress.getPointer();
  }

  void VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
    if (RetTypes.empty()) {
      // Get the inner expr
      const Expr *TmpE = E;
      // First come OSSArraySection
      bool FoundPointer = false;
      while (const OSSArraySectionExpr *ASE = dyn_cast<OSSArraySectionExpr>(TmpE->IgnoreParenImpCasts())) {
        // Stop in the innermost ArrayToPointerDecay
        TmpE = ASE->getBase();
        // If we see a Pointer we must add one dimension and done
        if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
          AddDimStartEnd();
          FoundPointer = true;
          break;
        }
      }
      if (!FoundPointer) {
        while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
          // Stop in the innermost ArrayToPointerDecay
          TmpE = ASE->getBase();
          // If we see a Pointer we must add one dimension and done
          if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
            AddDimStartEnd();
            break;
          }
        }
      }
      RetTypes.insert(RetTypes.begin(), TmpE->getType());
      FillDimsFromInnermostExpr(TmpE);
    }

    Visit(E->getBase());

    if (E->getLowerBound())
      Visit(E->getLowerBound());
    if (E->getLengthUpper())
      Visit(E->getLengthUpper());
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    if (RetTypes.empty()) {
      // Get the inner expr
      const Expr *TmpE = E;
      while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
        // Stop in the innermost ArrayToPointerDecay
        TmpE = ASE->getBase();
        // If we see a Pointer we must to add one dimension and done
        if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
          AddDimStartEnd();
          break;
        }
      }

      RetTypes.insert(RetTypes.begin(), TmpE->getType());
      FillDimsFromInnermostExpr(TmpE);
    }

    Visit(E->getBase());
    Visit(E->getIdx());
  }

  void VisitMemberExpr(const MemberExpr *E) {
    if (RetTypes.empty()) {
      RetTypes.push_back(CGF.getContext().getPointerType(E->getType()));
      FillBaseExprDims(E);
    }

    Visit(E->getBase());
  }

  void VisitUnaryDeref(const UnaryOperator *E) {
    if (RetTypes.empty()) {
      RetTypes.push_back(CGF.getContext().getPointerType(E->getType()));
      FillBaseExprDims(E);
    }

    Visit(E->getSubExpr());
  }

  void VisitStmt(const Stmt *S) {
    for (const Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  const llvm::MapVector<const VarDecl *, LValue> &getInvolvedVarList() const { return ExprInvolvedVarList; }
  const llvm::MapVector<const Expr *, llvm::Value *> &getVLASizeInvolvedMap() const { return VLASizeInvolvedMap; }
  const llvm::DenseMap<const VarDecl *, Address> &getCaptureInvolvedMap() const { return CaptureInvolvedMap; }
  ArrayRef<QualType> getRetTypes() const { return RetTypes; }
  llvm::Value *getBaseValue() const { assert(Base); return Base; }
  bool hasThis() const { return HasThis; }

};

} // namespace

namespace {
class OSSDependVisitor
  : public ConstStmtVisitor<OSSDependVisitor, void> {
  CodeGenFunction &CGF;
  bool OSSSyntax;

  llvm::Type *OSSArgTy;

  llvm::Value *Ptr;
  SmallVector<llvm::Value *, 4> Starts;
  SmallVector<llvm::Value *, 4> Ends;
  SmallVector<llvm::Value *, 4> Dims;
  QualType BaseElementTy;

public:

  OSSDependVisitor(CodeGenFunction &CGF, bool OSSSyntax)
    : CGF(CGF), OSSSyntax(OSSSyntax),
      OSSArgTy(CGF.ConvertType(CGF.getContext().LongTy))
      {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  QualType GetInnermostElementType(const QualType &Q) {
    if (Q->isArrayType()) {
      if (CGF.getContext().getAsConstantArrayType(Q)) {
        return CGF.getContext().getBaseElementType(Q);
      } else if (CGF.getContext().getAsVariableArrayType(Q)) {
        return CGF.getContext().getBaseElementType(Q);
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
    return Q;
  }

  void FillBaseExprDimsAndType(const Expr *E) {
    BaseElementTy = GetInnermostElementType(E->getType());
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType() || !TmpTy->isArrayType()) {
      // T * || T
      Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, DimSize));
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
        llvm::Value *DimExpr = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
        Dims.push_back(DimExpr);
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  // This is used in the innermost Expr * in ArraySubscripts and OSSArraySection
  void FillDimsFromInnermostExpr(const Expr *E) {
    // Go through the expression which may be a DeclRefExpr or MemberExpr or OSSArrayShapingExpr
    E = E->IgnoreParenImpCasts();
    QualType TmpTy = E->getType();
    // Add Dimensions
    if (TmpTy->isPointerType()) {
      // T *
      // We have added section dimension before
      if (Dims.empty())
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
      TmpTy = TmpTy->getPointeeType();
    }
    while (TmpTy->isArrayType()) {
      // T []
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(TmpTy)) {
        uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, DimSize));
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(TmpTy)) {
        auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
        llvm::Value *DimExpr = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
        Dims.push_back(DimExpr);
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  void Visit(const Expr *E) {
    ConstStmtVisitor<OSSDependVisitor, void>::Visit(E);
  }

  void VisitStmt(const Stmt *S) {
    llvm_unreachable("Unhandled stmt");
  }

  void VisitExpr(const Expr *E) {
    llvm_unreachable("Unhandled expr");
  }

  void VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
    BaseElementTy = GetInnermostElementType(E->getType());
    Ptr = CGF.EmitLValue(E).getPointer(CGF);
    if (E->getType()->isVariablyModifiedType()) {
      CGF.EmitVariablyModifiedType(E->getType());
    }
    FillDimsFromInnermostExpr(E);
  }

  void VisitOSSMultiDepExpr(const OSSMultiDepExpr *E) {
    Visit(E->getDepExpr());
  }

  // l-values.
  void VisitDeclRefExpr(const DeclRefExpr *E) {
    Ptr = CGF.EmitDeclRefLValue(E).getPointer(CGF);
    FillBaseExprDimsAndType(E);
  }

  void VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
    // Get Base Type
    // An array section is considered a built-in type
    BaseElementTy =
        OSSArraySectionExpr::getBaseOriginalType(
                          E->getBase());
    if (BaseElementTy->isAnyPointerType()) {
      BaseElementTy = BaseElementTy->getPointeeType();
    } else if (BaseElementTy->isArrayType()) {
      BaseElementTy = BaseElementTy->getAsArrayTypeUnsafe()->getElementType();
    } else {
      llvm_unreachable("Unhandled Type");
    }
    BaseElementTy = GetInnermostElementType(BaseElementTy);

    // Get the inner expr
    const Expr *TmpE = E;
    // First come OSSArraySection
    bool FoundPointer = false;
    while (const OSSArraySectionExpr *ASE = dyn_cast<OSSArraySectionExpr>(TmpE->IgnoreParenImpCasts())) {
      // Stop in the innermost ArrayToPointerDecay
      TmpE = ASE->getBase();
      // Add indexes
      llvm::Value *Idx, *IdxEnd;

      const Expr *LowerB = ASE->getLowerBound();
      if (LowerB)
        Idx = CGF.EmitScalarExpr(LowerB);
      else
        // OpenMP 5.0 2.1.5 When the lower-bound is absent it defaults to 0.
        Idx = llvm::ConstantInt::getSigned(OSSArgTy, 0);
      Idx = CGF.Builder.CreateSExt(Idx, OSSArgTy);

      const Expr *LengthUpper = ASE->getLengthUpper();
      bool ColonForm = ASE->isColonForm();
      if (LengthUpper &&
          (!OSSSyntax || (OSSSyntax && !ColonForm))) {
        // depend(in: array[ : length])
        // in(array[ ; length])
        IdxEnd = CGF.EmitScalarExpr(LengthUpper);
        IdxEnd = CGF.Builder.CreateSExt(IdxEnd, OSSArgTy);
        IdxEnd = CGF.Builder.CreateAdd(Idx, IdxEnd);
      } else if (LengthUpper
                 && (OSSSyntax && ColonForm)) {
        // in(array[ : upper])
        IdxEnd = CGF.EmitScalarExpr(LengthUpper);
        IdxEnd = CGF.Builder.CreateSExt(IdxEnd, OSSArgTy);
        IdxEnd = CGF.Builder.CreateAdd(llvm::ConstantInt::getSigned(OSSArgTy, 1), IdxEnd);
      } else if (ASE->getColonLoc().isInvalid()) {
        assert(!LengthUpper);
        // OSSArraySection without ':' are regular array subscripts
        IdxEnd = CGF.Builder.CreateAdd(Idx, llvm::ConstantInt::getSigned(OSSArgTy, 1));
      } else {

        // OpenMP 5.0 2.1.5
        // depend(in: array[lower : ]) -> [lower, dimsize)
        // When the length is absent it defaults to ⌈(size - lowerbound)∕stride⌉,
        // where size is the size of the array dimension.
        //
        // OmpSs-2
        // in(array[lower ; ]) -> [lower, dimsize)
        // in(array[lower : ]) -> [lower, dimsize)
        QualType BaseOriginalTy =
          OSSArraySectionExpr::getBaseOriginalType(ASE->getBase());

        if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(BaseOriginalTy)) {
          uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
          IdxEnd = llvm::ConstantInt::getSigned(OSSArgTy, DimSize);
        } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(BaseOriginalTy)) {
          auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
          IdxEnd = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
        } else {
          llvm_unreachable("Unhandled array type");
        }
      }
      IdxEnd = CGF.Builder.CreateSExt(IdxEnd, OSSArgTy);

      Starts.push_back(Idx);
      Ends.push_back(IdxEnd);
      // If we see a Pointer we must to add one dimension and done
      if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
        assert(LengthUpper && "Sema should have forbidden unspecified sizes in pointers");
        Dims.push_back(CGF.Builder.CreateSExt(CGF.EmitScalarExpr(LengthUpper), OSSArgTy));
        FoundPointer = true;
        break;
      }
    }
    if (!FoundPointer) {
      while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
        // Stop in the innermost ArrayToPointerDecay
        TmpE = ASE->getBase();
        // Add indexes
        llvm::Value *Idx = CGF.EmitScalarExpr(ASE->getIdx());
        Idx = CGF.Builder.CreateSExt(Idx, OSSArgTy);
        llvm::Value *IdxEnd = CGF.Builder.CreateAdd(Idx, llvm::ConstantInt::getSigned(OSSArgTy, 1));
        Starts.push_back(Idx);
        Ends.push_back(IdxEnd);
        // If we see a Pointer we must to add one dimension and done
        if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
          Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
          break;
        }
      }
    }

    Ptr = CGF.EmitScalarExpr(TmpE);
    if (const OSSArrayShapingExpr *OSA = dyn_cast<OSSArrayShapingExpr>(TmpE->IgnoreParenImpCasts())) {
      // We must to emit VLA args
      if (OSA->getType()->isVariablyModifiedType()) {
        CGF.EmitVariablyModifiedType(OSA->getType());
      }
    }
    FillDimsFromInnermostExpr(TmpE);
  }

  void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    // Get Base Type
    BaseElementTy = GetInnermostElementType(E->getType());
    // Get the inner expr
    const Expr *TmpE = E;
    while (const ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TmpE->IgnoreParenImpCasts())) {
      // Stop in the innermost ArrayToPointerDecay
      TmpE = ASE->getBase();
      // Add indexes
      llvm::Value *Idx = CGF.EmitScalarExpr(ASE->getIdx());
      Idx = CGF.Builder.CreateSExt(Idx, OSSArgTy);
      llvm::Value *IdxEnd = CGF.Builder.CreateAdd(Idx, llvm::ConstantInt::getSigned(OSSArgTy, 1));
      Starts.push_back(Idx);
      Ends.push_back(IdxEnd);
      // If we see a Pointer we must to add one dimension and done
      if (TmpE->IgnoreParenImpCasts()->getType()->isPointerType()) {
        Dims.push_back(llvm::ConstantInt::getSigned(OSSArgTy, 1));
        break;
      }
    }

    Ptr = CGF.EmitScalarExpr(TmpE);
    if (const OSSArrayShapingExpr *OSA = dyn_cast<OSSArrayShapingExpr>(TmpE->IgnoreParenImpCasts())) {
      // We must to emit VLA args
      if (OSA->getType()->isVariablyModifiedType()) {
        CGF.EmitVariablyModifiedType(OSA->getType());
      }
    }
    FillDimsFromInnermostExpr(TmpE);
  }

  void VisitMemberExpr(const MemberExpr *E) {
    Ptr = CGF.EmitMemberExpr(E).getPointer(CGF);
    FillBaseExprDimsAndType(E);
  }

  void VisitUnaryDeref(const UnaryOperator *E) {
    Ptr = CGF.EmitUnaryOpLValue(E).getPointer(CGF);
    FillBaseExprDimsAndType(E);
  }

  ArrayRef<llvm::Value *> getStarts() const {
    return Starts;
  }

  ArrayRef<llvm::Value *> getEnds() const {
    return Ends;
  }

  ArrayRef<llvm::Value *> getDims() const {
    return Dims;
  }

  QualType getBaseElementTy() {
    return BaseElementTy;
  }

  llvm::Value *getPtr() {
    return Ptr;
  }

};
} // namespace

// Gathers VLA dims info for VLA.DIMS (if IsPtr) and CAPTURED bundles
static void GatherVLADims(CodeGenFunction &CGF, llvm::Value *V, QualType Q,
                          SmallVectorImpl<llvm::Value *> &DimsWithValue,
                          SmallVectorImpl<llvm::Value *> &CapturedList,
                          bool IsPtr) {
  assert(DimsWithValue.empty() && "DimsWithValue must be empty");
  // C long -> LLVM long
  llvm::Type *OSSArgTy = CGF.ConvertType(CGF.getContext().LongTy);

  if (!IsPtr)
    DimsWithValue.push_back(V);
  while (Q->isArrayType()) {
    if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(Q)) {
      auto VlaSize = CGF.getVLAElements1D(BaseArrayTy);
      llvm::Value *DimExpr = CGF.Builder.CreateSExt(VlaSize.NumElts, OSSArgTy);
      if (!IsPtr)
        DimsWithValue.push_back(DimExpr);
      CapturedList.push_back(DimExpr);
      Q = BaseArrayTy->getElementType();
    } else if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(Q)) {
      uint64_t DimSize = BaseArrayTy->getSize().getSExtValue();
      llvm::Value *DimConstant = llvm::ConstantInt::getSigned(OSSArgTy, DimSize);
      if (!IsPtr)
        DimsWithValue.push_back(DimConstant);
      CapturedList.push_back(DimConstant);
      Q = BaseArrayTy->getElementType();
    } else {
      llvm_unreachable("Unhandled array type");
    }
  }
}

void CGOmpSsRuntime::EmitCopyCtorFunc(
    llvm::Value *DSAValue, const CXXConstructExpr *CtorE,
    const VarDecl *CopyD, const VarDecl *InitD,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  QualType Q = CopyD->getType();

  const CXXConstructorDecl *CtorD = cast<CXXConstructorDecl>(CtorE->getConstructor());
  // If we have already created the function we're done
  auto It = GenericCXXNonPodMethodDefs.find(CtorD);
  if (It != GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_copy), ArrayRef<llvm::Value*>{DSAValue, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(CopyD->getType());
  ImplicitParamDecl SrcArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamKind::Other);

  Args.push_back(&SrcArg);
  Args.push_back(&DstArg);
  Args.push_back(&NelemsArg);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);

  GlobalDecl CtorGD(CtorD, Ctor_Complete);
  llvm::Value *CtorValue = CGM.getAddrOfCXXStructor(CtorGD);

  std::string Name = "oss_copy_ctor";
  Name += CtorValue->getName();

  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());
  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(CGF);

  LValue SrcLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&SrcArg),
                                             PQ->castAs<PointerType>());
  Address SrcAddr = SrcLV.getAddress(CGF);
  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *SrcBegin = SrcLV.getPointer(CGF);
  llvm::Value *DstBegin = DstLV.getPointer(CGF);
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), DstBegin, NelemsValue,
      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  llvm::PHINode *SrcCur = CGF.Builder.CreatePHI(SrcBegin->getType(), 2,
                                             "arrayctor.src.cur");
  DstCur->addIncoming(DstBegin, EntryBB);
  SrcCur->addIncoming(SrcBegin, EntryBB);

  {
    CodeGenFunction::OSSPrivateScope InitScope(CGF);
    InitScope.addPrivate(InitD, [SrcCur, SrcAddr]() -> Address {

      return Address(
          SrcCur, SrcAddr.getElementType(), SrcAddr.getAlignment());
    });
    (void)InitScope.Privatize();
    CGF.EmitExprAsInit(CtorE, CopyD,
                       CGF.MakeAddrLValue(DstCur, DstLV.getType(), DstLV.getAlignment()),
                       /*capturedByInit=*/false);
  }

  // Go to the next element. Move SrcBegin too
  llvm::Value *DstNext = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), DstCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  llvm::Value *SrcDest = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), SrcCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.src.next");
  SrcCur->addIncoming(SrcDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  GenericCXXNonPodMethodDefs[CtorD] = Fn;

  TaskInfo.emplace_back(getBundleStr(OSSB_copy), ArrayRef<llvm::Value*>{DSAValue, Fn});

}

void CGOmpSsRuntime::EmitCtorFunc(
    llvm::Value *DSAValue, const VarDecl *CopyD,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {
  const CXXConstructExpr *CtorE = cast<CXXConstructExpr>(CopyD->getInit());
  const CXXConstructorDecl *CtorD = cast<CXXConstructorDecl>(CtorE->getConstructor());

  GlobalDecl CtorGD(CtorD, Ctor_Complete);
  llvm::Value *CtorValue = CGM.getAddrOfCXXStructor(CtorGD);

  auto It = GenericCXXNonPodMethodDefs.find(CtorD);
  if (It != GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_init), ArrayRef<llvm::Value*>{DSAValue, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType Q = CopyD->getType();
  QualType PQ = C.getPointerType(Q);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamKind::Other);

  Args.push_back(&DstArg);
  Args.push_back(&NelemsArg);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);

  std::string Name = "oss_ctor";
  Name += CtorValue->getName();

  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());
  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(CGF);

  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *DstBegin = DstLV.getPointer(CGF);
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), DstBegin, NelemsValue,
      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  DstCur->addIncoming(DstBegin, EntryBB);

  CGF.EmitExprAsInit(CtorE, CopyD,
                     CGF.MakeAddrLValue(DstCur, DstLV.getType(), DstLV.getAlignment()),
                     /*capturedByInit=*/false);

  // Go to the next element
  llvm::Value *DstNext = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), DstCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  GenericCXXNonPodMethodDefs[CtorD] = Fn;

  TaskInfo.emplace_back(getBundleStr(OSSB_init), ArrayRef<llvm::Value*>{DSAValue, Fn});
}

void CGOmpSsRuntime::EmitDtorFunc(
    llvm::Value *DSAValue, const VarDecl *CopyD,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  QualType Q = CopyD->getType();

  const RecordType *RT = Q->getAs<RecordType>();
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());

  if (RD->hasTrivialDestructor())
    return;

  const CXXDestructorDecl *DtorD = RD->getDestructor();

  GlobalDecl DtorGD(DtorD, Dtor_Complete);
  llvm::Value *DtorValue = CGM.getAddrOfCXXStructor(DtorGD);

  auto It = GenericCXXNonPodMethodDefs.find(DtorD);
  if (It != GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(getBundleStr(OSSB_deinit), ArrayRef<llvm::Value*>{DSAValue, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(Q);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamKind::Other);

  Args.push_back(&DstArg);
  Args.push_back(&NelemsArg);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);

  std::string Name = "oss_dtor";
  Name += DtorValue->getName();

  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());
  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(CGF);

  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *DstBegin = DstLV.getPointer(CGF);
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), DstBegin, NelemsValue,
      "arraydtor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arraydtor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arraydtor.dst.cur");
  DstCur->addIncoming(DstBegin, EntryBB);

  CGF.EmitCXXDestructorCall(
    DtorD, Dtor_Complete, /*ForVirtualBase=*/false, /*Delegating=*/false,
    Address(DstCur, DstLV.getAddress(CGF).getElementType(), DstLV.getAddress(CGF).getAlignment()),
    Q);

  // Go to the next element
  llvm::Value *DstNext = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), DstCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arraydtor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arraydtor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arraydtor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  GenericCXXNonPodMethodDefs[DtorD] = Fn;

  TaskInfo.emplace_back(getBundleStr(OSSB_deinit), ArrayRef<llvm::Value*>{DSAValue, Fn});
}

void CGOmpSsRuntime::EmitDSAShared(
  CodeGenFunction &CGF, const Expr *E,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList) {

  SmallVector<llvm::Value *> DSABundleList;
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
    llvm::Value *DSAValue;
    if ((VD->getType()->isReferenceType()
         || DRE->refersToEnclosingVariableOrCapture())
        && !getTaskCaptureAddr(VD).isValid()) {
      // 1. Record Ref Address to be reused in task body and other clasues
      // 2. Record Addr of lambda captured variables to be reused in task body and
      //    other clauses.
      LValue LV = CGF.EmitDeclRefLValue(DRE);
      CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
      DSAValue = LV.getPointer(CGF);
      DSABundleList.push_back(DSAValue);
    } else {
      DSAValue = CGF.EmitDeclRefLValue(DRE).getPointer(CGF);
      DSABundleList.push_back(DSAValue);
    }
    QualType Q = VD->getType();
    // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
    bool IsPtr = Q->isPointerType();
    SmallVector<llvm::Value *, 4> DimsWithValue;
    while (Q->isPointerType()) {
      Q = Q->getPointeeType();
    }
    if (Q->isVariableArrayType()) {
      GatherVLADims(CGF, DSAValue, Q, DimsWithValue, CapturedList, IsPtr);
      QualType BaseElementTy = CGF.getContext().getBaseElementType(DRE->getType());
      DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(BaseElementTy)));
    } else {
      DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(DRE->getType())));
    }

    TaskInfo.emplace_back(getBundleStr(OSSB_shared), DSABundleList);

    if (!DimsWithValue.empty())
      TaskInfo.emplace_back(getBundleStr(OSSB_vladims), DimsWithValue);

  } else if (const CXXThisExpr *ThisE = dyn_cast<CXXThisExpr>(E)) {
    DSABundleList.push_back(CGF.EmitScalarExpr(ThisE));
    DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(ThisE->getType()->getPointeeType())));
    TaskInfo.emplace_back(getBundleStr(OSSB_shared), DSABundleList);
  } else {
    llvm_unreachable("Unhandled expression");
  }
}

void CGOmpSsRuntime::EmitDSAPrivate(
  CodeGenFunction &CGF, const OSSDSAPrivateDataTy &PDataTy,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList) {

  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(PDataTy.Ref);
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  llvm::Value *DSAValue;
  SmallVector<llvm::Value *> DSABundleList;
  if ((VD->getType()->isReferenceType()
       || DRE->refersToEnclosingVariableOrCapture())
      && !getTaskCaptureAddr(VD).isValid()) {
    // 1. Record Ref Address to be reused in task body and other clasues
    // 2. Record Addr of lambda captured variables to be reused in task body and
    //    other clauses.
    LValue LV = CGF.EmitDeclRefLValue(DRE);
    CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
    DSAValue = LV.getPointer(CGF);
    DSABundleList.push_back(DSAValue);
  } else {
    DSAValue = CGF.EmitDeclRefLValue(DRE).getPointer(CGF);
    DSABundleList.push_back(DSAValue);
  }
  QualType Q = VD->getType();
  // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
  bool IsPtr = Q->isPointerType();
  SmallVector<llvm::Value *, 4> DimsWithValue;
  while (Q->isPointerType()) {
    Q = Q->getPointeeType();
  }
  if (Q->isVariableArrayType()) {
    GatherVLADims(CGF, DSAValue, Q, DimsWithValue, CapturedList, IsPtr);
    QualType BaseElementTy = CGF.getContext().getBaseElementType(DRE->getType());
    DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(BaseElementTy)));
  } else {
    DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(DRE->getType())));
  }

  TaskInfo.emplace_back(getBundleStr(OSSB_private), DSABundleList);

  if (!DimsWithValue.empty())
    TaskInfo.emplace_back(getBundleStr(OSSB_vladims), DimsWithValue);

  // TODO: is this sufficient to skip COPY/DEINIT? (task functions)
  if (PDataTy.Copy) {
    const DeclRefExpr *CopyE = cast<DeclRefExpr>(PDataTy.Copy);
    const VarDecl *CopyD = cast<VarDecl>(CopyE->getDecl());

    if (!CopyD->getType().isPODType(CGF.getContext())) {
      InDirectiveEmission = false;
      EmitCtorFunc(DSAValue, CopyD, TaskInfo);
      EmitDtorFunc(DSAValue, CopyD, TaskInfo);
      InDirectiveEmission = true;
    }
  }
}

void CGOmpSsRuntime::EmitDSAFirstprivate(
  CodeGenFunction &CGF, const OSSDSAFirstprivateDataTy &FpDataTy,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList) {

  const DeclRefExpr *DRE = cast<DeclRefExpr>(FpDataTy.Ref);
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  llvm::Value *DSAValue;
  SmallVector<llvm::Value *> DSABundleList;
  if ((VD->getType()->isReferenceType()
       || DRE->refersToEnclosingVariableOrCapture())
      && !getTaskCaptureAddr(VD).isValid()) {
    // 1. Record Ref Address to be reused in task body and other clasues
    // 2. Record Addr of lambda captured variables to be reused in task body and
    //    other clauses.
    LValue LV = CGF.EmitDeclRefLValue(DRE);
    CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
    DSAValue = LV.getPointer(CGF);
    DSABundleList.push_back(DSAValue);
  } else {
    DSAValue = CGF.EmitDeclRefLValue(DRE).getPointer(CGF);
    DSABundleList.push_back(DSAValue);
  }
  QualType Q = VD->getType();
  // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
  bool IsPtr = Q->isPointerType();
  SmallVector<llvm::Value *, 4> DimsWithValue;
  while (Q->isPointerType()) {
    Q = Q->getPointeeType();
  }
  if (Q->isVariableArrayType()) {
    GatherVLADims(CGF, DSAValue, Q, DimsWithValue, CapturedList, IsPtr);
    QualType BaseElementTy = CGF.getContext().getBaseElementType(DRE->getType());
    DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(BaseElementTy)));
  } else {
    DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(DRE->getType())));
  }

  TaskInfo.emplace_back(getBundleStr(OSSB_firstprivate), DSABundleList);

  if (!DimsWithValue.empty())
    TaskInfo.emplace_back(getBundleStr(OSSB_vladims), DimsWithValue);

  // TODO: is this sufficient to skip COPY/DEINIT? (task functions)
  if (FpDataTy.Copy) {
    const DeclRefExpr *CopyE = cast<DeclRefExpr>(FpDataTy.Copy);
    const VarDecl *CopyD = cast<VarDecl>(CopyE->getDecl());
    const DeclRefExpr *InitE = cast<DeclRefExpr>(FpDataTy.Init);
    const VarDecl *InitD = cast<VarDecl>(InitE->getDecl());

    if (!CopyD->getType().isPODType(CGF.getContext())) {
      const CXXConstructExpr *CtorE = cast<CXXConstructExpr>(CopyD->getAnyInitializer());

      InDirectiveEmission = false;
      EmitCopyCtorFunc(DSAValue, CtorE, CopyD, InitD, TaskInfo);
      EmitDtorFunc(DSAValue, CopyD, TaskInfo);
      InDirectiveEmission = true;
    }
  }
}

llvm::Function *CGOmpSsRuntime::createCallWrapperFunc(
    CodeGenFunction &CGF,
    const Expr *E,
    const Decl *FunContext,
    const llvm::MapVector<const VarDecl *, LValue> &ExprInvolvedVarList,
    const llvm::MapVector<const Expr *, llvm::Value *> &VLASizeInvolvedMap,
    const llvm::DenseMap<const VarDecl *, Address> &CaptureInvolvedMap,
    ArrayRef<QualType> RetTypes,
    bool HasThis, bool HasSwitch, std::string FuncName, std::string RetName,
    llvm::function_ref<void(CodeGenFunction &, const Expr *E, std::optional<llvm::Value *>)> Body) {

  InDirectiveEmission = false;

  ASTContext &C = CGF.CGM.getContext();

  CodeGenFunction NewCGF(CGF.CGM);

  FunctionArgList Args;
  for (const auto &p : ExprInvolvedVarList) {
    QualType Q = C.getPointerType(p.first->getType());
    auto *Arg =
      ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, SourceLocation(), p.first->getIdentifier(), Q, ImplicitParamKind::Other);
    Args.push_back(Arg);
  }
  for (const auto &p : VLASizeInvolvedMap) {
    (void)p;
    // VLASizes are SizeTy
    QualType Q = C.getSizeType();
    auto *Arg =
      ImplicitParamDecl::Create(
        C, Q, ImplicitParamKind::Other);
    Args.push_back(Arg);
  }
  for (const auto &p : CaptureInvolvedMap) {
    QualType Q = C.getPointerType(p.first->getType().getNonReferenceType());
    auto *Arg =
      ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, SourceLocation(), p.first->getIdentifier(), Q, ImplicitParamKind::Other);
    Args.push_back(Arg);
  }
  if (HasThis) {
    // We don't care about lambdas.
    // NOTE: We have seen 'this' so it's fine to assume we are in a method function
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FunContext)) {
      if (const CXXConstructorDecl *CtorD = dyn_cast<CXXConstructorDecl>(MD)) {
        NewCGF.CurGD = GlobalDecl(CtorD, Ctor_Complete);
      } else if (const CXXDestructorDecl *DtorD = dyn_cast<CXXDestructorDecl>(MD)) {
        NewCGF.CurGD = GlobalDecl(DtorD, Dtor_Complete);
      } else {
        NewCGF.CurGD = GlobalDecl(MD);
      }
      NewCGF.CGM.getCXXABI().buildThisParam(NewCGF, Args);
    }
  }
  if (HasSwitch) {
    QualType Q = C.getSizeType();
    auto *Arg =
      ImplicitParamDecl::Create(
        C, Q, ImplicitParamKind::Other);
    Args.push_back(Arg);
  }

  QualType RetQ;
  assert(!RetTypes.empty());
  if (RetTypes.size() != 1) {
    RecordDecl *RD = RecordDecl::Create(C, TagTypeKind::Struct,
                                        C.getTranslationUnitDecl(),
                                        SourceLocation(), SourceLocation(),
                                        &C.Idents.get(RetName));
    for (QualType Q : RetTypes) {
      RD->addDecl(FieldDecl::Create(C, RD, SourceLocation(), SourceLocation(),
                                    nullptr, Q, nullptr, nullptr, false,
                                    ICIS_NoInit));
    }
    RD->completeDefinition();
    RetQ = C.getTagDeclType(RD);
  } else {
    RetQ = RetTypes[0];
  }

  auto GetReturnType = [](QualType Q) -> CanQualType {
    // Borrowed from CGCall.cpp
    return Q->getCanonicalTypeUnqualified().getUnqualifiedType();
  };

  SmallVector<CanQualType, 16> ArgTypes;
  for (auto &Arg : Args)
    ArgTypes.push_back(C.getCanonicalParamType(Arg->getType()));
  const CGFunctionInfo &FuncInfo = CGF.CGM.getTypes().arrangeLLVMFunctionInfo(
    GetReturnType(RetQ), FnInfoOpts::None,
    ArgTypes, FunctionType::ExtInfo(CC_Trivial), {}, RequiredArgs::All);

  llvm::FunctionType *FuncType = CGF.CGM.getTypes().GetFunctionType(FuncInfo);

  auto *FuncVar = llvm::Function::Create(
      FuncType, llvm::GlobalValue::InternalLinkage, CGF.CurFn->getAddressSpace(),
      FuncName, &CGF.CGM.getModule());

  {
    CodeGenFunction::OSSPrivateScope Scope(NewCGF);
    auto *ArgI = FuncVar->arg_begin() + ExprInvolvedVarList.size();
    for (const auto &p : VLASizeInvolvedMap) {
      const Expr *VLASizeExpr = p.first;
      Scope.addPrivateVLA(VLASizeExpr, [ArgI]() -> llvm::Value * {
        return ArgI;
      });

      ++ArgI;
    }

    (void)Scope.Privatize();

    NewCGF.StartFunction(NewCGF.CurGD, RetQ, FuncVar, FuncInfo, Args, SourceLocation(), SourceLocation(), /*DoNotEH=*/true);
  }

  InDirectiveEmission = true;
  NewCGF.EHStack.pushTerminate();

  {
    CodeGenFunction::OSSPrivateScope InitScope(NewCGF);
    auto *ArgI = FuncVar->arg_begin();
    for (const auto &p : ExprInvolvedVarList) {
      const VarDecl *VD = p.first;
      LValue LV = p.second;
      Address Addr = LV.getAddress(NewCGF);
      InitScope.addPrivate(VD, [&ArgI, &Addr]() -> Address {
        return Address(
            ArgI, Addr.getElementType(), Addr.getAlignment());
      });

      ++ArgI;
    }
    for (const auto &p : VLASizeInvolvedMap) {
      const Expr *VLASizeExpr = p.first;
      InitScope.addPrivateVLA(VLASizeExpr, [ArgI]() -> llvm::Value * {
        return ArgI;
      });

      ++ArgI;
    }
    CaptureMapTy CaptureReplacedMap;
    for (const auto &p : CaptureInvolvedMap) {
      const VarDecl *VD = p.first;
      Address Addr = p.second;
      Address NewAddr = Address(ArgI, Addr.getElementType(), Addr.getAlignment());
      CaptureReplacedMap.try_emplace(VD, NewAddr);
      ++ArgI;
    }
    CaptureMapStack.push_back(CaptureReplacedMap);

    (void)InitScope.Privatize();

    // Function body generation
    std::optional<llvm::Value *> SwitchValue;
    if (HasSwitch)
      SwitchValue = &*(FuncVar->arg_end() - 1);
    Body(NewCGF, E, SwitchValue);
  }

  NewCGF.EHStack.popTerminate();
  NewCGF.FinishFunction(SourceLocation(), /*DoNotEH=*/true);

  // Pop temporal empty refmap, we are not in wrapper function anymore
  CaptureMapStack.pop_back();
  return FuncVar;
}

static llvm::Value *emitDiscreteArray(
    CodeGenFunction &CGF, const Expr *const DiscreteArrExpr, const Expr *const IterExpr) {
  if (auto *Ref = dyn_cast<DeclRefExpr>(DiscreteArrExpr)) {
    auto *VD = cast<VarDecl>(Ref->getDecl());
    CGF.EmitDecl(*VD);
    LValue DiscreteArrLV = CGF.EmitLValue(Ref);
    llvm::Value *Idx[2];
    Idx[0] = llvm::Constant::getNullValue(CGF.ConvertType(CGF.getContext().IntTy));
    Idx[1] = CGF.EmitScalarExpr(IterExpr);
    llvm::Value *Ptr = DiscreteArrLV.getPointer(CGF);
    llvm::Value *GEP = CGF.Builder.CreateGEP(
        CGF.ConvertType(VD->getType()), Ptr, Idx, "discreteidx");
    llvm::Value *LoadGEP = CGF.Builder.CreateLoad(
        Address(GEP, CGF.ConvertType(CGF.getContext().getBaseElementType(VD->getType())),
                DiscreteArrLV.getAddress(CGF).getAlignment()));
    return LoadGEP;
  }
  return nullptr;
}

void CGOmpSsRuntime::EmitMultiDependencyList(
    CodeGenFunction &CGF, const Decl *FunContext,
    const OSSDepDataTy &Dep, SmallVectorImpl<llvm::Value *> &List) {
  auto *MDExpr = cast<OSSMultiDepExpr>(Dep.E);

  OSSExprInfoGathering MultiDepInfoGathering(CGF);
  for (size_t i = 0; i < MDExpr->getDepInits().size(); ++i) {
    MultiDepInfoGathering.Visit(MDExpr->getDepIterators()[i]);
    MultiDepInfoGathering.Visit(MDExpr->getDepInits()[i]);
    if (MDExpr->getDepSizes()[i])
      MultiDepInfoGathering.Visit(MDExpr->getDepSizes()[i]);
    if (MDExpr->getDepSteps()[i])
      MultiDepInfoGathering.Visit(MDExpr->getDepSteps()[i]);
  }

  // NOTE:
  // OSSExprInfoGathering is too specific of dependency emission
  // It's better to just build the ret types here
  // For each iterator we have {init, remap, ub, step}
  SmallVector<QualType, 3 + 1> MultiDepRetTypes(
    MDExpr->getDepInits().size()*(3 + 1), CGF.getContext().IntTy);
  auto Body = [](CodeGenFunction &NewCGF, const Expr *E, std::optional<llvm::Value *> Switch) {
    auto *MDExpr = cast<OSSMultiDepExpr>(E);
    uint64_t SwitchTyBits =
      NewCGF.CGM.getDataLayout().getTypeSizeInBits(Switch.value()->getType());

    llvm::BasicBlock *ContBB = NewCGF.createBasicBlock("", NewCGF.CurFn);
    llvm::BasicBlock *FallBack = NewCGF.createBasicBlock("", NewCGF.CurFn);
    llvm::SwitchInst *SI = NewCGF.Builder.CreateSwitch(Switch.value(), FallBack);
    NewCGF.Builder.SetInsertPoint(FallBack);
    NewCGF.Builder.CreateBr(ContBB);

    Address RetAddr = NewCGF.ReturnValue;

    // Emit init/ub/step expressions
    for (size_t i = 0; i < MDExpr->getDepInits().size(); ++i) {
      llvm::BasicBlock *BB = NewCGF.createBasicBlock("", NewCGF.CurFn);
      SI->addCase(NewCGF.Builder.getIntN(SwitchTyBits, i), BB);
      NewCGF.Builder.SetInsertPoint(BB);

      const Expr *const IterExpr = MDExpr->getDepIterators()[i];
      const Expr *const InitExpr = MDExpr->getDepInits()[i];
      const Expr *const SizeExpr = MDExpr->getDepSizes()[i];
      const Expr *const StepExpr = MDExpr->getDepSteps()[i];
      const Expr *const DiscreteArrExpr = MDExpr->getDiscreteArrays()[i];
      bool IsSizeOrSection = MDExpr->getDepSizeOrSection()[i];
      llvm::Value *InitValue = nullptr;
      llvm::Value *RemapValue = nullptr;
      llvm::Value *SizeValue = nullptr;
      llvm::Value *StepValue = nullptr;
      if (DiscreteArrExpr) {
        // Discrete initialize to 0 the iterator and its remapped to what discrete array has
        InitValue = llvm::ConstantInt::get(
          NewCGF.ConvertType(NewCGF.getContext().IntTy), 0);
        RemapValue = emitDiscreteArray(NewCGF, DiscreteArrExpr, IterExpr);

        const ConstantArrayType *CAT =
          NewCGF.getContext().getAsConstantArrayType(DiscreteArrExpr->getType());

        uint64_t ArraySize = CAT->getSize().getZExtValue() - 1;
        SizeValue = llvm::ConstantInt::get(
          NewCGF.ConvertType(NewCGF.getContext().IntTy), ArraySize);
        StepValue = llvm::ConstantInt::get(
          NewCGF.ConvertType(NewCGF.getContext().IntTy), 1);
      } else {
        // Ranges initialize to 'init' and there's no remap
        InitValue = NewCGF.EmitScalarExpr(InitExpr);
        RemapValue = NewCGF.EmitScalarExpr(IterExpr);

        SizeValue = NewCGF.EmitScalarExpr(SizeExpr);
        // Convert Size to UB, that is -> Init + Size - 1
        if (IsSizeOrSection) {
          SizeValue = NewCGF.Builder.CreateAdd(InitValue, SizeValue);
          SizeValue = NewCGF.Builder.CreateAdd(
            SizeValue, llvm::ConstantInt::getSigned(
              NewCGF.ConvertType(NewCGF.getContext().IntTy), -1));
        }

        if (StepExpr) {
          StepValue = NewCGF.EmitScalarExpr(StepExpr);
        } else {
          // There's no step, so default to 1
          StepValue = llvm::ConstantInt::getSigned(
            NewCGF.ConvertType(NewCGF.getContext().IntTy), 1);
        }
      }
      assert(InitValue && RemapValue && SizeValue && StepValue);
      NewCGF.Builder.CreateStore(InitValue, NewCGF.Builder.CreateStructGEP(RetAddr, i*4 + 0));
      NewCGF.Builder.CreateStore(RemapValue, NewCGF.Builder.CreateStructGEP(RetAddr, i*4 + 1));
      NewCGF.Builder.CreateStore(SizeValue, NewCGF.Builder.CreateStructGEP(RetAddr, i*4 + 2));
      NewCGF.Builder.CreateStore(StepValue, NewCGF.Builder.CreateStructGEP(RetAddr, i*4 + 3));
      NewCGF.Builder.CreateBr(ContBB);
    }
    NewCGF.Builder.SetInsertPoint(ContBB);
  };

  llvm::Function *ComputeMultiDepFun = createCallWrapperFunc(
    CGF,
    MDExpr,
    FunContext,
    MultiDepInfoGathering.getInvolvedVarList(),
    MultiDepInfoGathering.getVLASizeInvolvedMap(),
    MultiDepInfoGathering.getCaptureInvolvedMap(),
    MultiDepRetTypes,
    MultiDepInfoGathering.hasThis(),
    /*HasSwitch=*/true,
    "compute_dep", "_depend_unpack_t",
    Body);

  // Fill the bundle
  for (auto *E : MDExpr->getDepIterators()) {
    List.push_back(CGF.EmitLValue(E).getPointer(CGF));
  }
  List.push_back(ComputeMultiDepFun);

  for (const auto &p : MultiDepInfoGathering.getInvolvedVarList()) {
    LValue LV = p.second;
    List.push_back(LV.getPointer(CGF));
  }
  for (const auto &p : MultiDepInfoGathering.getVLASizeInvolvedMap()) {
    llvm::Value *VLASizeValue = p.second;
    List.push_back(VLASizeValue);
  }
  for (const auto &p : MultiDepInfoGathering.getCaptureInvolvedMap()) {
    Address Addr = p.second;
    llvm::Value *V = Addr.getPointer();
    List.push_back(V);
  }
  if (MultiDepInfoGathering.hasThis()) {
    List.push_back(CGF.LoadCXXThisAddress().getPointer());
  }
}

void CGOmpSsRuntime::BuildWrapperCallBundleList(
    std::string FuncName,
    CodeGenFunction &CGF, const Expr *E, QualType Q,
    llvm::function_ref<void(CodeGenFunction &, const Expr *E, std::optional<llvm::Value *>)> Body,
    SmallVectorImpl<llvm::Value *> &List) {

  OSSExprInfoGathering CallVisitor(CGF);
  CallVisitor.Visit(E);

  CodeGenFunction NewCGF(CGF.CGM);

  SmallVector<QualType, 1> RetTypes{Q};

  auto *Func = createCallWrapperFunc(
    CGF,
    E,
    CGF.CurGD.getDecl()->getNonClosureContext(),
    CallVisitor.getInvolvedVarList(),
    CallVisitor.getVLASizeInvolvedMap(),
    CallVisitor.getCaptureInvolvedMap(),
    RetTypes,
    CallVisitor.hasThis(),
    /*HasSwitch=*/false,
    FuncName, "",
    Body);

  List.push_back(Func);
  for (const auto &p : CallVisitor.getInvolvedVarList()) {
    LValue LV = p.second;
    List.push_back(LV.getPointer(CGF));
  }
  for (const auto &p : CallVisitor.getVLASizeInvolvedMap()) {
    llvm::Value *VLASizeValue = p.second;
    List.push_back(VLASizeValue);
  }
  for (const auto &p : CallVisitor.getCaptureInvolvedMap()) {
    Address Addr = p.second;
    llvm::Value *V = Addr.getPointer();
    List.push_back(V);
  }
  if (CallVisitor.hasThis()) {
    List.push_back(CGF.LoadCXXThisAddress().getPointer());
  }
}

void CGOmpSsRuntime::EmitWrapperCallBundle(
    std::string Name, std::string FuncName,
    CodeGenFunction &CGF, const Expr *E, QualType Q,
    llvm::function_ref<void(CodeGenFunction &, const Expr *E, std::optional<llvm::Value *>)> Body,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  SmallVector<llvm::Value *, 4> List;
  BuildWrapperCallBundleList(FuncName, CGF, E, Q, Body, List);
  TaskInfo.emplace_back(Name, List);
}

void CGOmpSsRuntime::EmitScalarWrapperCallBundle(
    std::string Name, std::string FuncName,
    CodeGenFunction &CGF, const Expr *E,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  auto Body = [](CodeGenFunction &NewCGF, const Expr *E, std::optional<llvm::Value *>) {
    llvm::Value *V = NewCGF.EmitScalarExpr(E);

    Address RetAddr = NewCGF.ReturnValue;
    NewCGF.Builder.CreateStore(V, RetAddr);
  };
  EmitWrapperCallBundle(
    Name, FuncName, CGF, E, E->getType(), Body, TaskInfo);
}

void CGOmpSsRuntime::EmitBoolWrapperCallBundle(
    std::string Name, std::string FuncName,
    CodeGenFunction &CGF, const Expr *E,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  auto Body = [](CodeGenFunction &NewCGF, const Expr *E, std::optional<llvm::Value *>) {
    llvm::Value *V = NewCGF.EvaluateExprAsBool(E);

    Address RetAddr = NewCGF.ReturnValue;
    NewCGF.Builder.CreateStore(V, RetAddr);
  };
  EmitWrapperCallBundle(
    Name, FuncName, CGF, E, CGF.getContext().BoolTy, Body, TaskInfo);
}

void CGOmpSsRuntime::EmitIgnoredWrapperCallBundle(
    std::string Name, std::string FuncName,
    CodeGenFunction &CGF, const Expr *E,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  auto Body = [](CodeGenFunction &NewCGF, const Expr *E, std::optional<llvm::Value *>) {
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
      if (DRE->isNonOdrUse() == NOUR_Unevaluated)
        return;

      CodeGenFunction::ConstantEmission CE = NewCGF.tryEmitAsConstant(const_cast<DeclRefExpr*>(DRE));
      if (CE && !CE.isReference())
        // Constant value, no need to annotate it.
        return;
    }

    NewCGF.EmitIgnoredExpr(E);
  };
  EmitWrapperCallBundle(
    Name, FuncName, CGF, E, CGF.getContext().VoidTy, Body, TaskInfo);

}

void CGOmpSsRuntime::EmitDependencyList(
    CodeGenFunction &CGF, const Decl *FunContext,
    const OSSDepDataTy &Dep, SmallVectorImpl<llvm::Value *> &List) {

  OSSExprInfoGathering DependInfoGathering(CGF);
  DependInfoGathering.Visit(Dep.E);

  CodeGenFunction NewCGF(CGF.CGM);

  auto Body = [&Dep](CodeGenFunction &NewCGF, const Expr *E, std::optional<llvm::Value *>) {
    SmallVector<llvm::Value *, 4> List;
    // C long -> LLVM long
    llvm::Type *OSSArgTy = NewCGF.ConvertType(NewCGF.getContext().LongTy);

    OSSDependVisitor DepVisitor(NewCGF, Dep.OSSSyntax);
    DepVisitor.Visit(E);

    SmallVector<llvm::Value *, 4> Starts(
        DepVisitor.getStarts().begin(),
        DepVisitor.getStarts().end());

    SmallVector<llvm::Value *, 4> Ends(
        DepVisitor.getEnds().begin(),
        DepVisitor.getEnds().end());

    SmallVector<llvm::Value *, 4> Dims(
        DepVisitor.getDims().begin(),
        DepVisitor.getDims().end());
    QualType BaseElementTy = DepVisitor.getBaseElementTy();
    llvm::Value *Ptr = DepVisitor.getPtr();

    uint64_t BaseElementSize =
               NewCGF.getContext()
                 .getTypeSizeInChars(BaseElementTy).getQuantity();

    List.push_back(Ptr);
    bool First = true;
    for (size_t i = Dims.size() - 1; i > 0; --i) {
      llvm::Value *Dim = Dims[i];
      llvm::Value *IdxStart;
      llvm::Value *IdxEnd;
      // In arrays we have to output all dimensions, but
      // the number of indices may be less than the number
      // of dimensions (array[1] -> int array[10][20])
      if (i < Starts.size()) {
        IdxStart = Starts[Starts.size() - i - 1];
        IdxEnd = Ends[Starts.size() - i - 1];
      } else {
        IdxStart = llvm::ConstantInt::getSigned(OSSArgTy, 0);
        IdxEnd = Dim;
      }
      if (First) {
        First = false;
        Dim = NewCGF.Builder.CreateMul(Dim,
                                    llvm::ConstantInt::getSigned(OSSArgTy,
                                                                 BaseElementSize));
        IdxStart = NewCGF.Builder.CreateMul(IdxStart,
                                    llvm::ConstantInt::getSigned(OSSArgTy,
                                                                 BaseElementSize));
        IdxEnd = NewCGF.Builder.CreateMul(IdxEnd,
                                    llvm::ConstantInt::getSigned(OSSArgTy,
                                                                 BaseElementSize));
      }
      List.push_back(Dim);
      List.push_back(IdxStart);
      List.push_back(IdxEnd);
    }
    llvm::Value *Dim = Dims[0];
    llvm::Value *IdxStart;
    llvm::Value *IdxEnd;
    if (Starts.size() > 0) {
      IdxStart = Starts[Starts.size() - 1];
      IdxEnd = Ends[Starts.size() - 1];
    } else {
      IdxStart = llvm::ConstantInt::getSigned(OSSArgTy, 0);
      IdxEnd = Dim;
    }

    if (First) {
      First = false;
      Dim = NewCGF.Builder.CreateMul(Dim,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
      IdxStart = NewCGF.Builder.CreateMul(IdxStart,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
      IdxEnd = NewCGF.Builder.CreateMul(IdxEnd,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
    }
    List.push_back(Dim);
    List.push_back(IdxStart);
    List.push_back(IdxEnd);

    Address RetAddr = NewCGF.ReturnValue;
    for (size_t i = 0; i < List.size(); ++i) {
      NewCGF.Builder.CreateStore(List[i], NewCGF.Builder.CreateStructGEP(RetAddr, i));
    }
  };

  llvm::Function *ComputeDepFun = createCallWrapperFunc(
    CGF,
    Dep.E,
    FunContext,
    DependInfoGathering.getInvolvedVarList(),
    DependInfoGathering.getVLASizeInvolvedMap(),
    DependInfoGathering.getCaptureInvolvedMap(),
    DependInfoGathering.getRetTypes(),
    DependInfoGathering.hasThis(),
    /*HasSwitch=*/false,
    "compute_dep", "_depend_unpack_t",
    Body);

  // assert(List.size() == DependInfoGathering.getRetTypes().size());

  // Fill the bundle
  List.push_back(DependInfoGathering.getBaseValue());

  auto &SM = CGM.getContext().getSourceManager();
  CharSourceRange ExpansionRange =
    SM.getExpansionRange(SourceRange(Dep.E->getBeginLoc(), Dep.E->getEndLoc()));
  StringRef ExprStringified = Lexer::getSourceText(
    ExpansionRange, SM, CGM.getLangOpts());
  List.push_back(
    llvm::ConstantDataArray::getString(
      CGM.getLLVMContext(), ExprStringified));

  List.push_back(ComputeDepFun);

  for (const auto &p : DependInfoGathering.getInvolvedVarList()) {
    LValue LV = p.second;
    List.push_back(LV.getPointer(CGF));
  }
  for (const auto &p : DependInfoGathering.getVLASizeInvolvedMap()) {
    llvm::Value *VLASizeValue = p.second;
    List.push_back(VLASizeValue);
  }
  for (const auto &p : DependInfoGathering.getCaptureInvolvedMap()) {
    Address Addr = p.second;
    llvm::Value *V = Addr.getPointer();
    List.push_back(V);
  }
  if (DependInfoGathering.hasThis()) {
    List.push_back(CGF.LoadCXXThisAddress().getPointer());
  }
}

static std::string convertDepToMultiDepStr(StringRef Str) {
  if (Str == getBundleStr(OSSB_in)) return getBundleStr(OSSB_multi_in);
  if (Str == getBundleStr(OSSB_out)) return getBundleStr(OSSB_multi_out);
  if (Str == getBundleStr(OSSB_inout)) return getBundleStr(OSSB_multi_inout);
  if (Str == getBundleStr(OSSB_concurrent)) return getBundleStr(OSSB_multi_concurrent);
  if (Str == getBundleStr(OSSB_commutative)) return getBundleStr(OSSB_multi_commutative);
  if (Str == getBundleStr(OSSB_weakin)) return getBundleStr(OSSB_multi_weakin);
  if (Str == getBundleStr(OSSB_weakout)) return getBundleStr(OSSB_multi_weakout);
  if (Str == getBundleStr(OSSB_weakinout)) return getBundleStr(OSSB_multi_weakinout);
  if (Str == getBundleStr(OSSB_weakconcurrent)) return getBundleStr(OSSB_multi_weakconcurrent);
  if (Str == getBundleStr(OSSB_weakcommutative)) return getBundleStr(OSSB_multi_weakcommutative);
  llvm_unreachable("unexpected dependency bundle string");
}

void CGOmpSsRuntime::EmitDependency(
    std::string Name, CodeGenFunction &CGF, const Decl *FunContext,
    const OSSDepDataTy &Dep, SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  // Save CurFuncDecl to get the 'this' address
  const Decl *CurFuncDecl = CGF.CurFuncDecl;
  CGF.CurFuncDecl = FunContext;

  SmallVector<llvm::Value*, 4> MultiDepData;
  SmallVector<llvm::Value*, 4> DepData;
  SmallVector<llvm::Value*, 4> MultiAndDepData;

  if (isa<OSSMultiDepExpr>(Dep.E)) {
    // Convert Name in MultiName
    Name = convertDepToMultiDepStr(Name);
    EmitMultiDependencyList(CGF, FunContext, Dep, MultiDepData);
  }

  EmitDependencyList(CGF, FunContext, Dep, DepData);

  // Restore CurFuncDecl
  CGF.CurFuncDecl = CurFuncDecl;

  // Merge the two value lists
  MultiAndDepData.append(MultiDepData.begin(), MultiDepData.end());
  MultiAndDepData.append(DepData.begin(), DepData.end());

  TaskInfo.emplace_back(Name, llvm::ArrayRef(MultiAndDepData));
}

/// Check if the combiner is a call to UDR and if it is so return the
/// UDR decl used for reduction.
static const OSSDeclareReductionDecl *
getReductionDecl(const Expr *ReductionOp) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(ReductionOp))
    if (const auto *DRD = dyn_cast<OSSDeclareReductionDecl>(DRE->getDecl()))
      return DRD;
  return nullptr;
}

static llvm::Value *emitReduceInitFunction(CodeGenModule &CGM,
                                           const OSSReductionDataTy &Red) {
  ASTContext &C = CGM.getContext();

  const DeclRefExpr *RHSRef = cast<DeclRefExpr>(Red.RHS);
  const VarDecl *RHSVD = cast<VarDecl>(RHSRef->getDecl());

  const OSSDeclareReductionDecl *DRD =
      getReductionDecl(Red.ReductionOp);

  QualType Q = RHSVD->getType();
  QualType PQ = C.getPointerType(Q);

  FunctionArgList Args;
  ImplicitParamDecl PrivArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl OrigArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl NBytesArg(C, C.getSizeType(), ImplicitParamKind::Other);

  Args.emplace_back(&PrivArg);
  Args.emplace_back(&OrigArg);
  Args.emplace_back(&NBytesArg);

  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = "red_init";
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());

  LValue PrivLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&PrivArg),
                                              PQ->castAs<PointerType>());
  Address PrivAddr = PrivLV.getAddress(CGF);
  LValue OrigLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&OrigArg),
                                              PQ->castAs<PointerType>());
  Address OrigAddr = OrigLV.getAddress(CGF);
  llvm::Value *NBytesValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NBytesArg), /*Volatile=*/false,
                       NBytesArg.getType(), NBytesArg.getLocation());

  uint64_t BaseElementSize =
             CGF.CGM
               .getDataLayout()
               .getTypeSizeInBits(CGF
                                  .ConvertType(Q))/8;

  llvm::Value *NelemsValue = CGF.Builder.CreateExactUDiv(NBytesValue,
                                  llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()),
                                                         BaseElementSize));

  // Find the end of the array.
  llvm::Value *OrigBegin = OrigLV.getPointer(CGF);
  llvm::Value *PrivBegin = PrivLV.getPointer(CGF);
  llvm::Value *PrivEnd = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), PrivBegin, NelemsValue,
      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *PrivCur = CGF.Builder.CreatePHI(PrivBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  llvm::PHINode *OrigCur = CGF.Builder.CreatePHI(OrigBegin->getType(), 2,
                                             "arrayctor.src.cur");
  PrivCur->addIncoming(PrivBegin, EntryBB);
  OrigCur->addIncoming(OrigBegin, EntryBB);

  if (DRD && DRD->getInitializer()) {

    const auto *PrivVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getInitPriv())->getDecl());
    const auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getInitOrig())->getDecl());

    CodeGenFunction::OSSPrivateScope InitScope(CGF);
    InitScope.addPrivate(PrivVD, [PrivCur, PrivAddr]() -> Address {
      return Address(PrivCur, PrivAddr.getElementType(), PrivAddr.getAlignment());
    });
    InitScope.addPrivate(OrigVD, [OrigCur, OrigAddr]() -> Address {
      return Address(OrigCur, OrigAddr.getElementType(), OrigAddr.getAlignment());
    });
    (void)InitScope.Privatize();

    if (DRD->getInitializerKind() == OSSDeclareReductionDecl::CallInit) {
      // initializer(foo(&omp_priv, &omp_orig))
      CGF.EmitIgnoredExpr(DRD->getInitializer());
    } else {
      // initializer(omp_priv = ...)
      // initializer(omp_priv(...))
      CGF.EmitExprAsInit(PrivVD->getInit(), PrivVD,
                         CGF.MakeAddrLValue(PrivCur, PrivLV.getType(), PrivLV.getAlignment()),
                         /*capturedByInit=*/false);
    }
  } else {
    assert(RHSVD->hasInit() && "RHSVD has no initializer");
    CGF.EmitExprAsInit(RHSVD->getInit(), RHSVD,
                       CGF.MakeAddrLValue(PrivCur, PrivLV.getType(), PrivLV.getAlignment()),
                       /*capturedByInit=*/false);
  }

  // Go to the next element. Move OrigBegin too
  llvm::Value *PrivNext = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), PrivCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.dst.next");
  PrivCur->addIncoming(PrivNext, CGF.Builder.GetInsertBlock());

  llvm::Value *OrigDest = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), OrigCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.src.next");
  OrigCur->addIncoming(OrigDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(PrivNext, PrivEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();
  return Fn;
}

static llvm::Value *emitReduceCombFunction(CodeGenModule &CGM,
                                           const OSSReductionDataTy &Red) {
  ASTContext &C = CGM.getContext();
  const auto *LHSVD = cast<VarDecl>(cast<DeclRefExpr>(Red.LHS)->getDecl());
  const auto *RHSVD = cast<VarDecl>(cast<DeclRefExpr>(Red.RHS)->getDecl());

  const OSSDeclareReductionDecl *DRD =
      getReductionDecl(Red.ReductionOp);

  QualType Q = LHSVD->getType();
  QualType PQ = C.getPointerType(Q);

  FunctionArgList Args;
  ImplicitParamDecl OutArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl InArg(C, PQ, ImplicitParamKind::Other);
  ImplicitParamDecl NBytesArg(C, C.getSizeType(), ImplicitParamKind::Other);

  Args.emplace_back(&OutArg);
  Args.emplace_back(&InArg);
  Args.emplace_back(&NBytesArg);

  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = "red_comb";
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, SourceLocation(), SourceLocation());

  LValue OutLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&OutArg),
                                              PQ->castAs<PointerType>());
  Address OutAddr = OutLV.getAddress(CGF);
  LValue InLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&InArg),
                                              PQ->castAs<PointerType>());
  Address InAddr = InLV.getAddress(CGF);
  llvm::Value *NBytesValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NBytesArg), /*Volatile=*/false,
                       NBytesArg.getType(), NBytesArg.getLocation());

  uint64_t BaseElementSize =
             CGF.CGM
               .getDataLayout()
               .getTypeSizeInBits(CGF
                                  .ConvertType(Q))/8;

  llvm::Value *NelemsValue = CGF.Builder.CreateExactUDiv(NBytesValue,
                                  llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()),
                                                         BaseElementSize));

  // Find the end of the array.
  llvm::Value *InBegin = InLV.getPointer(CGF);
  llvm::Value *OutBegin = OutLV.getPointer(CGF);
  llvm::Value *OutEnd = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), OutBegin, NelemsValue,
      "arrayctor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arrayctor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *OutCur = CGF.Builder.CreatePHI(OutBegin->getType(), 2,
                                             "arrayctor.dst.cur");
  llvm::PHINode *InCur = CGF.Builder.CreatePHI(InBegin->getType(), 2,
                                             "arrayctor.src.cur");
  OutCur->addIncoming(OutBegin, EntryBB);
  InCur->addIncoming(InBegin, EntryBB);

  // Remap lhs and rhs variables to the addresses of the function arguments.
  if (DRD) {
    const auto *OutVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getCombinerOut())->getDecl());
    const auto *InVD = cast<VarDecl>(cast<DeclRefExpr>(DRD->getCombinerIn())->getDecl());
    CodeGenFunction::OSSPrivateScope CombScope(CGF);
    CombScope.addPrivate(OutVD, [OutCur, OutAddr]() -> Address {
      return Address(OutCur, OutAddr.getElementType(), OutAddr.getAlignment());
    });
    CombScope.addPrivate(InVD, [InCur, InAddr]() -> Address {
      return Address(InCur, InAddr.getElementType(), InAddr.getAlignment());
    });
    (void)CombScope.Privatize();

    CGF.EmitIgnoredExpr(DRD->getCombiner());
  } else {
    // Emit the combiner body:
    // %2 = <ReductionOp>(<type> *%lhs, <type> *%rhs)
    // store <type> %2, <type>* %lhs
    CodeGenFunction::OSSPrivateScope CombScope(CGF);
    CombScope.addPrivate(LHSVD, [OutCur, OutAddr]() -> Address {
      return Address(OutCur, OutAddr.getElementType(), OutAddr.getAlignment());
    });
    CombScope.addPrivate(RHSVD, [InCur, InAddr]() -> Address {
      return Address(InCur, InAddr.getElementType(), InAddr.getAlignment());
    });
    (void)CombScope.Privatize();

    CGF.EmitIgnoredExpr(Red.ReductionOp);
  }

  // Go to the next element. Move InBegin too
  llvm::Value *OutNext = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), OutCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.dst.next");
  OutCur->addIncoming(OutNext, CGF.Builder.GetInsertBlock());

  llvm::Value *InDest = CGF.Builder.CreateInBoundsGEP(
      CGF.ConvertType(Q), InCur,
      llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
      "arrayctor.src.next");
  InCur->addIncoming(InDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(OutNext, OutEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();
  return Fn;
}

static llvm::ConstantInt *reductionKindToNanos6Enum(CodeGenFunction &CGF, QualType Q, BinaryOperatorKind BOK) {
// enum ReductionType
// {
//   RED_TYPE_CHAR = 1000,
//   RED_TYPE_SIGNED_CHAR = 2000,
//   RED_TYPE_UNSIGNED_CHAR = 3000,
//   RED_TYPE_SHORT = 4000,
//   RED_TYPE_UNSIGNED_SHORT = 5000,
//   RED_TYPE_INT = 6000,
//   RED_TYPE_UNSIGNED_INT = 7000,
//   RED_TYPE_LONG = 8000,
//   RED_TYPE_UNSIGNED_LONG = 9000,
//   RED_TYPE_LONG_LONG = 10000,
//   RED_TYPE_UNSIGNED_LONG_LONG = 11000,
//   RED_TYPE_FLOAT = 12000,
//   RED_TYPE_DOUBLE = 13000,
//   RED_TYPE_LONG_DOUBLE = 14000,
//   RED_TYPE_COMPLEX_FLOAT = 15000,
//   RED_TYPE_COMPLEX_DOUBLE = 16000,
//   RED_TYPE_COMPLEX_LONG_DOUBLE = 17000,
//   RED_TYPE_BOOLEAN = 18000,
//   NUM_RED_TYPES = 19000
// };
// enum ReductionOperation
// {
//   RED_OP_ADDITION = 0,
//   RED_OP_PRODUCT = 1,
//   RED_OP_BITWISE_AND = 2,
//   RED_OP_BITWISE_OR = 3,
//   RED_OP_BITWISE_XOR = 4,
//   RED_OP_LOGICAL_AND = 5,
//   RED_OP_LOGICAL_OR = 6,
//   RED_OP_LOGICAL_XOR = 7,
//   RED_OP_LOGICAL_NXOR = 8,
//   RED_OP_MAXIMUM = 9,
//   RED_OP_MINIMUM = 10,
//   NUM_RED_OPS = 11
// };
//
// int Result = ReductionType + ReductionOperation
//
// In case of UDR return -1

  llvm::Type *RedOpTy = CGF.ConvertType(CGF.getContext().IntTy);

  if (BOK == BO_Comma)
    return cast<llvm::ConstantInt>(llvm::ConstantInt::getSigned(RedOpTy, -1));

  int ReductionType = -1;
  int ReductionOperation = -1;

  // Template instantiated types are not canonical
  // See SubstTemplateTypeParmType
  Q = Q.getCanonicalType();

  if (Q == CGF.getContext().CharTy)                  ReductionType = 1000;
  else if (Q == CGF.getContext().SignedCharTy)       ReductionType = 2000;
  else if (Q == CGF.getContext().UnsignedCharTy)     ReductionType = 3000;
  else if (Q == CGF.getContext().ShortTy)            ReductionType = 4000;
  else if (Q == CGF.getContext().UnsignedShortTy)    ReductionType = 5000;
  else if (Q == CGF.getContext().IntTy)              ReductionType = 6000;
  else if (Q == CGF.getContext().UnsignedIntTy)      ReductionType = 7000;
  else if (Q == CGF.getContext().LongTy)             ReductionType = 8000;
  else if (Q == CGF.getContext().UnsignedLongTy)     ReductionType = 9000;
  else if (Q == CGF.getContext().LongLongTy)         ReductionType = 10000;
  else if (Q == CGF.getContext().UnsignedLongLongTy) ReductionType = 11000;
  else if (Q == CGF.getContext().FloatTy)            ReductionType = 12000;
  else if (Q == CGF.getContext().DoubleTy)           ReductionType = 13000;
  else if (Q == CGF.getContext().LongDoubleTy)       ReductionType = 14000;
  else if (
    Q == CGF.getContext().getComplexType(
      CGF.getContext().FloatTy))                     ReductionType = 15000;
  else if (
    Q == CGF.getContext().getComplexType(
      CGF.getContext().DoubleTy))                    ReductionType = 16000;
  else if (
    Q == CGF.getContext().getComplexType(
      CGF.getContext().LongDoubleTy))                ReductionType = 17000;
  else if (Q == CGF.getContext().BoolTy)             ReductionType = 18000;
  else llvm_unreachable("unhandled reduction type");

  if (BOK == BO_Add)           ReductionOperation = 0;
  else if (BOK == BO_Mul)      ReductionOperation = 1;
  else if (BOK == BO_And)      ReductionOperation = 2;
  else if (BOK == BO_Or)       ReductionOperation = 3;
  else if (BOK == BO_Xor)      ReductionOperation = 4;
  else if (BOK == BO_LAnd)     ReductionOperation = 5;
  else if (BOK == BO_LOr)      ReductionOperation = 6;
  // else if (BOK == BO_LXor)  ReductionOperation = 7;
  // else if (BOK == BO_LNXor) ReductionOperation = 8;
  else if (BOK == BO_GT)       ReductionOperation = 9;
  else if (BOK == BO_LT)       ReductionOperation = 10;
  else llvm_unreachable("unhandled reduction operation");

  return cast<llvm::ConstantInt>(llvm::ConstantInt::getSigned(RedOpTy, ReductionType + ReductionOperation));

}

void CGOmpSsRuntime::EmitReduction(
    std::string RedName, std::string RedInitName, std::string RedCombName,
    CodeGenFunction &CGF, const Decl *FunContext,
    const OSSReductionDataTy &Red, SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  // Save CurFuncDecl to get the 'this' address
  const Decl *CurFuncDecl = CGF.CurFuncDecl;
  CGF.CurFuncDecl = FunContext;

  SmallVector<llvm::Value *, 4> List;

  llvm::ConstantInt *RedKind = reductionKindToNanos6Enum(CGF, Red.ReductionOp->getType(), Red.ReductionKind);
  llvm::Value *RedInit = nullptr;
  llvm::Value *RedComb = nullptr;
  InDirectiveEmission = false;
  if (!RedKind->isMinusOne()) {
    auto It = BuiltinRedMap.find(RedKind);
    if (It == BuiltinRedMap.end()) {
      RedInit = emitReduceInitFunction(CGF.CGM, Red);
      RedComb = emitReduceCombFunction(CGF.CGM, Red);
      BuiltinRedMap[RedKind] = {RedInit, RedComb};
    } else {
      RedInit = It->second.first;
      RedComb = It->second.second;
    }
  } else {
    const OSSDeclareReductionDecl *D = getReductionDecl(Red.ReductionOp);
    auto It = UDRMap.find(getReductionDecl(Red.ReductionOp));
    if (It == UDRMap.end()) {
      RedInit = emitReduceInitFunction(CGF.CGM, Red);
      RedComb = emitReduceCombFunction(CGF.CGM, Red);
      UDRMap[D] = {RedInit, RedComb};
    } else {
      RedInit = It->second.first;
      RedComb = It->second.second;
    }
  }
  InDirectiveEmission = true;

  EmitDependencyList(CGF, FunContext, {/*OSSSyntax=*/true, Red.Ref}, List);

  // Restore CurFuncDecl
  CGF.CurFuncDecl = CurFuncDecl;

  // First operand has to be the DSA over the dependency is made
  llvm::Value *DepBaseDSA = List[0];

  List.insert(List.begin(), RedKind);

  TaskInfo.emplace_back(RedName, ArrayRef(List));
  TaskInfo.emplace_back(RedInitName, ArrayRef<llvm::Value *>{DepBaseDSA, RedInit});
  TaskInfo.emplace_back(RedCombName, ArrayRef<llvm::Value *>{DepBaseDSA, RedComb});
}

void CGOmpSsRuntime::emitTaskwaitCall(CodeGenFunction &CGF,
                                      SourceLocation Loc,
                                      const OSSTaskDataTy &Data) {
  if (Data.empty()) {
    // Regular taskwait
    llvm::Function *Callee = CGM.getIntrinsic(llvm::Intrinsic::directive_marker);
    CGF.Builder.CreateCall(
        Callee, {},
        {
          llvm::OperandBundleDef(
            std::string(getBundleStr(OSSB_directive)),
            llvm::ConstantDataArray::getString(
              CGM.getLLVMContext(), getBundleStr(OSSB_taskwait)))
        });
  } else {
    // taskwait with deps -> task with deps if(0)
    llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
    llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
    SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
    TaskInfo.emplace_back(
        getBundleStr(OSSB_directive),
        llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_task)));

    // Add if(0) flag
    llvm::Type *Int1Ty = CGF.ConvertType(CGF.getContext().BoolTy);
    TaskInfo.emplace_back(getBundleStr(OSSB_if), llvm::ConstantInt::getSigned(Int1Ty, 0));

    // Push Task Stack
    TaskStack.push_back(TaskContext());
    CaptureMapStack.push_back(CaptureMapTy());

    InDirectiveEmission = true;
    EmitDirectiveData(CGF, Data, TaskInfo);
    InDirectiveEmission = false;

    llvm::Instruction *Result =
      CGF.Builder.CreateCall(EntryCallee, {}, llvm::ArrayRef(TaskInfo));
    CGF.Builder.CreateCall(ExitCallee, Result);

    // Pop Task Stack
    TaskStack.pop_back();
    CaptureMapStack.pop_back();
  }
}

void CGOmpSsRuntime::emitReleaseCall(
    CodeGenFunction &CGF, SourceLocation Loc, const OSSTaskDataTy &Data) {

  SmallVector<llvm::OperandBundleDef, 8> ReleaseInfo;

  ReleaseInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_release)));

  // Push Task Stack
  TaskStack.push_back(TaskContext());
  CaptureMapStack.push_back(CaptureMapTy());

  InDirectiveEmission = true;
  EmitDirectiveData(CGF, Data, ReleaseInfo);
  InDirectiveEmission = false;

  llvm::Function *Callee = CGM.getIntrinsic(llvm::Intrinsic::directive_marker);
  CGF.Builder.CreateCall(Callee, {}, llvm::ArrayRef(ReleaseInfo));

  // Pop Task Stack
  TaskStack.pop_back();
  CaptureMapStack.pop_back();
}

// We're in task body context once we set InsertPt
bool CGOmpSsRuntime::inTaskBody() {
  return !TaskStack.empty() && TaskStack.back().InsertPt;
}

llvm::AssertingVH<llvm::Instruction> CGOmpSsRuntime::getTaskInsertPt() {
  return TaskStack.back().InsertPt;
}

llvm::BasicBlock *CGOmpSsRuntime::getTaskTerminateHandler() {
  return TaskStack.back().TerminateHandler;
}

llvm::BasicBlock *CGOmpSsRuntime::getTaskTerminateLandingPad() {
  return TaskStack.back().TerminateLandingPad;
}

llvm::BasicBlock *CGOmpSsRuntime::getTaskUnreachableBlock() {
  return TaskStack.back().UnreachableBlock;
}

Address CGOmpSsRuntime::getTaskExceptionSlot() {
  return TaskStack.back().ExceptionSlot;
}

Address CGOmpSsRuntime::getTaskEHSelectorSlot() {
  return TaskStack.back().EHSelectorSlot;
}

Address CGOmpSsRuntime::getTaskNormalCleanupDestSlot() {
  return TaskStack.back().NormalCleanupDestSlot;
}

Address CGOmpSsRuntime::getTaskCaptureAddr(const VarDecl *VD) {
  for (auto ItMap = CaptureMapStack.rbegin();
       ItMap != CaptureMapStack.rend(); ++ItMap) {
    auto it = ItMap->find(VD);
    if (it != ItMap->end()) {
      return it->second;
    }
  }
  return Address::invalid();
}

void CGOmpSsRuntime::setTaskInsertPt(llvm::Instruction *I) {
  TaskStack.back().InsertPt = I;
}

void CGOmpSsRuntime::setTaskTerminateHandler(llvm::BasicBlock *BB) {
  TaskStack.back().TerminateHandler = BB;
}

void CGOmpSsRuntime::setTaskTerminateLandingPad(llvm::BasicBlock *BB) {
  TaskStack.back().TerminateLandingPad = BB;
}

void CGOmpSsRuntime::setTaskUnreachableBlock(llvm::BasicBlock *BB) {
  TaskStack.back().UnreachableBlock = BB;
}

void CGOmpSsRuntime::setTaskExceptionSlot(Address Addr) {
  TaskStack.back().ExceptionSlot = Addr;
}

void CGOmpSsRuntime::setTaskEHSelectorSlot(Address Addr) {
  TaskStack.back().EHSelectorSlot = Addr;
}

void CGOmpSsRuntime::setTaskNormalCleanupDestSlot(Address Addr) {
  TaskStack.back().NormalCleanupDestSlot = Addr;
}

// Borrowed brom CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty()) {
    CGF.CurFn->insert(CGF.CurFn->end(), BB);
    return;
  }
  delete BB;
}

static void EmitLoopType(const OSSLoopDataTy &LoopData, CodeGenFunction &CGF,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  SmallVector<llvm::Value*, 4> List;
  // LT → <    → 0
  // LE → <=   → 1
  // GT → >    → 2
  // GE → >=   → 3
  for (unsigned i = 0; i < LoopData.NumCollapses; ++i) {
    int IsLessOp = !*LoopData.TestIsLessOp [i]* 2;
    int IsStrictOp = !LoopData.TestIsStrictOp [i]* 1;
    int LoopCmp = IsLessOp + IsStrictOp;
    List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopCmp));
    List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.IndVar[i]->getType()->isSignedIntegerOrEnumerationType()));
    List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.LB[i]->getType()->isSignedIntegerOrEnumerationType()));
    List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.UB[i]->getType()->isSignedIntegerOrEnumerationType()));
    List.push_back(llvm::ConstantInt::getSigned(CGF.Int64Ty, LoopData.Step[i]->getType()->isSignedIntegerOrEnumerationType()));
    // TODO: missing step increment/decrement
  }
  TaskInfo.emplace_back(getBundleStr(OSSB_loop_type), List);
}

static VarDecl *createImplicitVarDecl(
    CodeGenFunction &CGF, StringRef Name, QualType Q, SourceLocation Loc,
    const DeclContext *DC) {
  ASTContext &Ctx = CGF.getContext();
  auto *VD =
    VarDecl::Create(
      Ctx, const_cast<DeclContext *>(DC),
      Loc, Loc, &Ctx.Idents.get(Name),
      Q, Ctx.getTrivialTypeSourceInfo(Q, Loc), SC_Auto);
  VD->setImplicit();
  return VD;
}

static DeclRefExpr *emitTaskCallArg(
    CodeGenFunction &CGF, StringRef Name, QualType Q, SourceLocation Loc,
    std::optional<const Expr *> InitE = std::nullopt) {

  ASTContext &Ctx = CGF.getContext();

  Q = Q.getUnqualifiedType();
  if (Q->isArrayType())
    Q = Ctx.getBaseElementType(Q).getCanonicalType();

  auto *VD = createImplicitVarDecl(CGF, Name, Q, Loc, cast<DeclContext>(CGF.CurCodeDecl));

  VD->setReferenced();
  VD->markUsed(Ctx);
  VD->setInitStyle(VarDecl::CInit);
  if (InitE.has_value())
    VD->setInit(const_cast<Expr *>(*InitE));

  CGF.EmitVarDecl(*VD);

  DeclRefExpr *Ref = DeclRefExpr::Create(
      Ctx, NestedNameSpecifierLoc(), SourceLocation(), VD,
      /*RefersToEnclosingVariableOrCapture=*/false, Loc, Q.getNonReferenceType(), VK_LValue);
  return Ref;
}

static void emitOutlineMultiDepIterDecls(
    CodeGenFunction &CGF, const OSSTaskDeclAttr *TaskDecl, SmallVectorImpl<Expr *> &PrivateCopies,
    CodeGenFunction::OSSPrivateScope &Scope) {
  auto EmitDepListIterDecls = [&CGF, &PrivateCopies, &Scope](const Expr *DepExpr) {
    if (auto *MDExpr = dyn_cast<OSSMultiDepExpr>(DepExpr)) {
      for (auto *IterE : MDExpr->getDepIterators()) {
        auto *IterVD = cast<VarDecl>(cast<DeclRefExpr>(IterE)->getDecl());

        // Do not emit the iterator as is because we want it to be unique
        // of each task call

        QualType Q = IterVD->getType();
        // Do not put any location since it is not an argument
        SourceLocation Loc;
        std::string Name(IterVD->getName());
        Name += ".call_arg";

        Expr *NewIterE = emitTaskCallArg(CGF, Name, Q, Loc);

        PrivateCopies.push_back(NewIterE);

        LValue NewIterLV = CGF.EmitLValue(NewIterE);

        Scope.addPrivate(IterVD, [&CGF, &NewIterLV]() -> Address { return NewIterLV.getAddress(CGF); });
        (void)Scope.Privatize();
      }
    }
  };
  for (const Expr *E : TaskDecl->ins()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->outs()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->inouts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->concurrents()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->commutatives()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->weakIns()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->weakOuts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->weakInouts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->weakConcurrents()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->weakCommutatives()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depIns()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depOuts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depInouts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depConcurrents()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depCommutatives()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depWeakIns()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depWeakOuts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depWeakInouts()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depWeakConcurrents()) { EmitDepListIterDecls(E); }
  for (const Expr *E : TaskDecl->depWeakCommutatives()) { EmitDepListIterDecls(E); }
  // TODO: reductions
}

static void emitMultiDepIterDecls(CodeGenFunction &CGF, const OSSTaskDataTy &Data) {
  auto EmitDepListIterDecls = [&CGF](ArrayRef<OSSDepDataTy> DepList) {
    for (const OSSDepDataTy &Dep : DepList) {
      if (auto *MDExpr = dyn_cast<OSSMultiDepExpr>(Dep.E)) {
        for (auto *E : MDExpr->getDepIterators()) {
          CGF.EmitDecl(*cast<DeclRefExpr>(E)->getDecl());
        }
      }
    }
  };
  EmitDepListIterDecls(Data.Deps.Ins);
  EmitDepListIterDecls(Data.Deps.Outs);
  EmitDepListIterDecls(Data.Deps.Inouts);
  EmitDepListIterDecls(Data.Deps.Concurrents);
  EmitDepListIterDecls(Data.Deps.Commutatives);
  EmitDepListIterDecls(Data.Deps.WeakIns);
  EmitDepListIterDecls(Data.Deps.WeakOuts);
  EmitDepListIterDecls(Data.Deps.WeakInouts);
  EmitDepListIterDecls(Data.Deps.WeakConcurrents);
  EmitDepListIterDecls(Data.Deps.WeakCommutatives);
  // TODO: reductions
}

// NOTE: keep this synchronized with
// OSSTaskDeclAttr::DeviceType
// and
// OmpSsDeviceClauseKind
static int convertDeviceTypeToInt(int Device) {
  const int NumDevAttrTypes = 4;
  const int DevAttrToNanos6Map[NumDevAttrTypes] = {
    0, // nanos6_host_device
    1, // nanos6_cuda_device
    4, // nanos6_opencl_device
    5, // nanos6_fpga_device
  };
  return DevAttrToNanos6Map[Device];
}

void CGOmpSsRuntime::EmitDirectiveData(
    CodeGenFunction &CGF, const OSSTaskDataTy &Data,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
    const OSSLoopDataTy &LoopData, const Expr *WhileCond) {

  emitMultiDepIterDecls(CGF, Data);

  SmallVector<llvm::Value*, 4> CapturedList;
  for (const Expr *E : Data.DSAs.Shareds) {
    EmitDSAShared(CGF, E, TaskInfo, CapturedList);
  }
  for (const OSSDSAPrivateDataTy &PDataTy : Data.DSAs.Privates) {
    EmitDSAPrivate(CGF, PDataTy, TaskInfo, CapturedList);
  }
  for (const OSSDSAFirstprivateDataTy &FpDataTy : Data.DSAs.Firstprivates) {
    EmitDSAFirstprivate(CGF, FpDataTy, TaskInfo, CapturedList);
  }

  if (Data.Cost) {
    EmitScalarWrapperCallBundle(
      getBundleStr(OSSB_cost), "compute_cost", CGF, Data.Cost, TaskInfo);
  }
  if (Data.Priority) {
    EmitScalarWrapperCallBundle(
      getBundleStr(OSSB_priority), "compute_priority", CGF, Data.Priority, TaskInfo);
  }
  if (!Data.Labels.empty()) {
    SmallVector<llvm::Value *, 2> LabelValues;
    for (const Expr *E : Data.Labels)
      LabelValues.push_back(CGF.EmitScalarExpr(E));
    TaskInfo.emplace_back(getBundleStr(OSSB_label), LabelValues);
  }
  if (Data.Wait) {
    TaskInfo.emplace_back(
        getBundleStr(OSSB_wait),
        llvm::ConstantInt::getTrue(CGM.getLLVMContext()));
  }
  if (Data.Onready) {
    EmitIgnoredWrapperCallBundle(
      getBundleStr(OSSB_onready), "compute_onready", CGF, Data.Onready, TaskInfo);
  }

  if (WhileCond) {
    EmitBoolWrapperCallBundle(
      getBundleStr(OSSB_while_cond), "compute_while_cond", CGF, WhileCond, TaskInfo);
    if (LoopData.Unroll) {
      llvm::Value *V = CGF.EmitScalarExpr(LoopData.Unroll);
      CapturedList.push_back(V);
      TaskInfo.emplace_back(getBundleStr(OSSB_unroll), V);
    }
    if (LoopData.Update) {
      TaskInfo.emplace_back(
          getBundleStr(OSSB_update),
          llvm::ConstantInt::getTrue(CGM.getLLVMContext()));
    }
  } else if (!LoopData.empty()) {
    SmallVector<llvm::Value *> IndVarList;
    SmallVector<llvm::Value *> LBList;
    SmallVector<llvm::Value *> UBList;
    SmallVector<llvm::Value *> StepList;
    SmallVector<llvm::Value *> LTypeList;
    for (unsigned i = 0; i < LoopData.NumCollapses; ++i) {
      IndVarList.push_back(CGF.EmitLValue(LoopData.IndVar[i]).getPointer(CGF));
      auto Body = [](CodeGenFunction &NewCGF, const Expr *E, std::optional<llvm::Value *>) {
        llvm::Value *V = NewCGF.EmitScalarExpr(E);

        Address RetAddr = NewCGF.ReturnValue;
        NewCGF.Builder.CreateStore(V, RetAddr);
      };
      BuildWrapperCallBundleList(
        "compute_lb", CGF, LoopData.LB[i], LoopData.LB[i]->getType(), Body, LBList);
      BuildWrapperCallBundleList(
        "compute_ub", CGF, LoopData.UB[i], LoopData.UB[i]->getType(), Body, UBList);
      BuildWrapperCallBundleList(
        "compute_step", CGF, LoopData.Step[i], LoopData.Step[i]->getType(), Body, StepList);
    }
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_indvar), IndVarList);
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_lowerbound), LBList);
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_upperbound), UBList);
    TaskInfo.emplace_back(getBundleStr(OSSB_loop_step), StepList);

    if (LoopData.Chunksize) {
      llvm::Value *V = CGF.EmitScalarExpr(LoopData.Chunksize);
      TaskInfo.emplace_back(getBundleStr(OSSB_chunksize), V);
    }
    if (LoopData.Grainsize) {
      llvm::Value *V = CGF.EmitScalarExpr(LoopData.Grainsize);
      TaskInfo.emplace_back(getBundleStr(OSSB_grainsize), V);
    }
    if (LoopData.Unroll) {
      llvm::Value *V = CGF.EmitScalarExpr(LoopData.Unroll);
      CapturedList.push_back(V);
      TaskInfo.emplace_back(getBundleStr(OSSB_unroll), V);
    }
    if (LoopData.Update) {
      TaskInfo.emplace_back(
          getBundleStr(OSSB_update),
          llvm::ConstantInt::getTrue(CGM.getLLVMContext()));
    }
    EmitLoopType(LoopData, CGF, TaskInfo);
  }

  if (!CapturedList.empty())
    TaskInfo.emplace_back(getBundleStr(OSSB_captured), CapturedList);

  const Decl *FunContext = CGF.CurGD.getDecl()->getNonClosureContext();
  for (const OSSDepDataTy &Dep : Data.Deps.Ins) {
    EmitDependency(getBundleStr(OSSB_in), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Outs) {
    EmitDependency(getBundleStr(OSSB_out), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Inouts) {
    EmitDependency(getBundleStr(OSSB_inout), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Concurrents) {
    EmitDependency(getBundleStr(OSSB_concurrent), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Commutatives) {
    EmitDependency(getBundleStr(OSSB_commutative), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakIns) {
    EmitDependency(getBundleStr(OSSB_weakin), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakOuts) {
    EmitDependency(getBundleStr(OSSB_weakout), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakInouts) {
    EmitDependency(getBundleStr(OSSB_weakinout), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakConcurrents) {
    EmitDependency(getBundleStr(OSSB_weakconcurrent), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakCommutatives) {
    EmitDependency(getBundleStr(OSSB_weakcommutative), CGF, FunContext, Dep, TaskInfo);
  }
  for (const OSSReductionDataTy &Red : Data.Reductions.RedList) {
    EmitReduction(getBundleStr(OSSB_reduction),
                  getBundleStr(OSSB_redinit),
                  getBundleStr(OSSB_redcomb),
                  CGF, FunContext, Red, TaskInfo);
  }
  for (const OSSReductionDataTy &Red : Data.Reductions.WeakRedList) {
    EmitReduction(getBundleStr(OSSB_weakreduction),
                  getBundleStr(OSSB_redinit),
                  getBundleStr(OSSB_redcomb),
                  CGF, FunContext, Red, TaskInfo);
  }

  if (Data.If)
    TaskInfo.emplace_back(getBundleStr(OSSB_if), CGF.EvaluateExprAsBool(Data.If));
  if (Data.Final)
    TaskInfo.emplace_back(getBundleStr(OSSB_final), CGF.EvaluateExprAsBool(Data.Final));
  if (!Data.Devices.empty())
    TaskInfo.emplace_back(
        getBundleStr(OSSB_device),
        llvm::ConstantInt::getSigned(
          CGF.ConvertType(CGF.getContext().IntTy),
          convertDeviceTypeToInt(Data.Devices.DvKind)));
}

static void EmitDbgInfo(CodeGenFunction &CGF, CGDebugInfo *DI, const Expr *E) {
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
    if (VD->getType()->isReferenceType()) {
      const VarDecl *ImplVD = createImplicitVarDecl(
        CGF, VD->getName(), E->getType(), VD->getLocation(), VD->getDeclContext());
      (void)DI->EmitDeclareOfAutoVariable(
        ImplVD, CGF.EmitLValue(DRE).getPointer(CGF), CGF.Builder, /*UsePointerValue=*/false);
    } else {
      (void)DI->EmitDeclareOfAutoVariable(
        VD, CGF.EmitLValue(DRE).getPointer(CGF), CGF.Builder, /*UsePointerValue=*/false);
    }
  } else if (isa<CXXThisExpr>(E)) {
    const VarDecl *ImplVD = createImplicitVarDecl(
      CGF, "this", E->getType()->getPointeeType(), E->getExprLoc(), cast<DeclContext>(CGF.CurCodeDecl));

    (void)DI->EmitDeclareOfAutoVariable(
      ImplVD, CGF.EmitScalarExpr(E), CGF.Builder, /*UsePointerValue=*/false);
  } else {
    llvm_unreachable("Unsupported debug info in OmpSs-2 directive");
  }
}

void CGOmpSsRuntime::EmitDirectiveDbgInfo(
    CodeGenFunction &CGF, const OSSTaskDataTy &Data) {
  auto *DI = CGF.getDebugInfo();
  for (const Expr *E : Data.DSAs.Shareds) {
    EmitDbgInfo(CGF, DI, E);
  }
  for (const OSSDSAPrivateDataTy &PDataTy : Data.DSAs.Privates) {
    EmitDbgInfo(CGF, DI, PDataTy.Ref);
  }
  for (const OSSDSAFirstprivateDataTy &FpDataTy : Data.DSAs.Firstprivates) {
    EmitDbgInfo(CGF, DI, FpDataTy.Ref);
  }
}

llvm::AllocaInst *CGOmpSsRuntime::createTaskAwareAlloca(
    CodeGenFunction &CGF, llvm::Type *Ty, const Twine &Name, llvm::Value *ArraySize) {
  if (InDirectiveEmission && TaskStack.size() > 1)
    return new llvm::AllocaInst(Ty, CGM.getDataLayout().getAllocaAddrSpace(),
                                ArraySize, Name, TaskStack[TaskStack.size() - 2].InsertPt);
  if (inTaskBody())
    return new llvm::AllocaInst(Ty, CGM.getDataLayout().getAllocaAddrSpace(),
                                ArraySize, Name, TaskStack[TaskStack.size() - 1].InsertPt);
  return new llvm::AllocaInst(Ty, CGM.getDataLayout().getAllocaAddrSpace(),
                              ArraySize, Name, CGF.AllocaInsertPt);
}

RValue CGOmpSsRuntime::emitTaskFunction(CodeGenFunction &CGF,
                                        const FunctionDecl *FD,
                                        const CallExpr *CE,
                                        ReturnValueSlot ReturnValue) {
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Ctx = CGM.getContext();

  SmallVector<Expr *, 4> ParmCopies;
  // Annotate which are the call arguments. Used to place them
  // first in the task args.
  SmallVector<llvm::Value *, 4> CallValues;
  SmallVector<Expr *, 4> FirstprivateCopies;
  SmallVector<Expr *, 4> PrivateCopies;
  SmallVector<Expr *, 4> SharedCopies;

  MemberExpr *NewME = nullptr;

  CodeGenFunction::OSSPrivateScope InitScope(CGF, /*Restore=*/true);

  InDirectiveEmission = true;
  CaptureMapStack.push_back(CaptureMapTy());

  auto ArgI = CE->arg_begin();
  auto ParI = FD->param_begin();
  while (ArgI != CE->arg_end()) {

    QualType ParQ = (*ParI)->getType();
    SourceLocation Loc = (*ArgI)->getExprLoc();

    // The a new VarDecl like ParamArgDecl, but in context of function call
    Expr *ParmRef = emitTaskCallArg(CGF, "call_arg", ParQ, Loc, *ArgI);

    if (!ParQ->isReferenceType()) {
      switch (CGF.getEvaluationKind(ParQ)) {
      case TEK_Complex:
      case TEK_Scalar: {
        ParmRef =
          ImplicitCastExpr::Create(Ctx, ParmRef->getType(), CK_LValueToRValue,
                                   ParmRef, /*BasePath=*/nullptr,
                                   VK_PRValue, FPOptionsOverride());
        break;
      }
      case TEK_Aggregate: {
        ParmRef =
          ImplicitCastExpr::Create(Ctx, ParmRef->getType(), CK_NoOp,
                                   ParmRef, /*BasePath=*/nullptr,
                                   VK_PRValue, FPOptionsOverride());
        break;
      }
      }
      FirstprivateCopies.push_back(ParmRef);

      LValue ParmLV = CGF.EmitLValue(ParmRef);

      // Annotate the call argument
      CallValues.push_back(ParmLV.getPointer(CGF));

      InitScope.addPrivate(*ParI, [&CGF, &ParmLV]() -> Address { return ParmLV.getAddress(CGF); });
      // We do need to do this every time because the next param may use the previous one
      (void)InitScope.Privatize();
    } else {
      // We want to pass references as shared so task can modify the original value
      SharedCopies.push_back(ParmRef);

      // Since it is a reference we have to manually set both call_arg and
      // function parameter address. The first because we rewrite the call expr and
      // the second because the dependency is made over that Decl
      LValue ParmLV = CGF.EmitLValue(ParmRef);

      // Annotate the call argument
      CallValues.push_back(ParmLV.getPointer(CGF));

      CaptureMapStack.back().try_emplace(*ParI, ParmLV.getAddress(CGF));
      CaptureMapStack.back().try_emplace(cast<VarDecl>(cast<DeclRefExpr>(ParmRef)->getDecl()), ParmLV.getAddress(CGF));

    }

    ParmCopies.push_back(ParmRef);

    ++ArgI;
    ++ParI;
  }

  // Emit multidep iterators before building task bundles
  // NOTE: this should do only one iteration
  for (const auto *Attr : FD->specific_attrs<OSSTaskDeclAttr>()) {
    emitOutlineMultiDepIterDecls(CGF, Attr, PrivateCopies, InitScope);
  }

  llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_task)));

  TaskStack.push_back(TaskContext());

  bool IsMethodCall = false;
  bool HasNdrange = false;
  if (const auto *CXXE = dyn_cast<CXXMemberCallExpr>(CE)) {
    IsMethodCall = true;
    const Expr *callee = CXXE->getCallee()->IgnoreParens();
    const MemberExpr *ME = cast<MemberExpr>(callee);
    bool IsArrow = ME->isArrow();
    const Expr *Base = ME->getBase();

    // Emulate call_arg as a reference
    // Built the call_arg that stores 'this' or any complex expression like M[1][2]->foo();
    Expr *MEBase;
    if (IsArrow) {
      Expr *TmpBase =
        UnaryOperator::Create(
          Ctx, const_cast<Expr *>(Base), UO_Deref, Base->getType()->getPointeeType(),
          VK_PRValue, OK_Ordinary, SourceLocation(), false, FPOptionsOverride());
      MEBase = emitTaskCallArg(
        CGF, "call_arg", Ctx.getLValueReferenceType(Base->getType()->getPointeeType()),
        Base->getExprLoc(), TmpBase);
    } else {
      MEBase = emitTaskCallArg(
        CGF, "call_arg", Ctx.getLValueReferenceType(Base->getType()),
        Base->getExprLoc(), Base);
    }

    NewME = MemberExpr::Create(
          Ctx, MEBase, /*IsArrow=*/false, ME->getOperatorLoc(),
          ME->getQualifierLoc(), ME->getTemplateKeywordLoc(),
          ME->getMemberDecl(), ME->getFoundDecl(), ME->getMemberNameInfo(),
          nullptr, ME->getType(), ME->getValueKind(),
          ME->getObjectKind(), ME->isNonOdrUse());

    LValue This = CGF.EmitLValue(MEBase);

    InitScope.setThis(This.getPointer(CGF), This.getAlignment());

    CaptureMapStack.back().try_emplace(cast<VarDecl>(cast<DeclRefExpr>(MEBase)->getDecl()), This.getAddress(CGF));
    SmallVector<llvm::Value *> DSABundleList;
    DSABundleList.push_back(This.getPointer(CGF));
    DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(MEBase->getType())));
    TaskInfo.emplace_back(getBundleStr(OSSB_shared), DSABundleList);
  }

  // This class is used to inspect task outline clauses looking
  // for global variables, which have to be captured as shared.
  class OSSOutlineGlobalVarVisitor
    : public ConstStmtVisitor<OSSOutlineGlobalVarVisitor, void> {
    CodeGenFunction &CGF;
    llvm::SetVector<const DeclRefExpr *> SharedGlobals;
  public:
    OSSOutlineGlobalVarVisitor(CodeGenFunction &CGF)
      : CGF(CGF)
        {}

    //===--------------------------------------------------------------------===//
    //                            Visitor Methods
    //===--------------------------------------------------------------------===//

    void VisitStmt(const Stmt *S) {
      for (const Stmt *C : S->children()) {
        if (C) {
          Visit(C);
        }
      }
    }

    void VisitOSSMultiDepExpr(const OSSMultiDepExpr *E) {
      Visit(E->getDepExpr());

      for (size_t i = 0; i < E->getDepInits().size(); ++i) {
        Visit(E->getDepInits()[i]);
        if (E->getDepSizes()[i])
          Visit(E->getDepSizes()[i]);
        if (E->getDepSteps()[i])
          Visit(E->getDepSteps()[i]);
      }
    }

    void VisitOSSArrayShapingExpr(const OSSArrayShapingExpr *E) {
      Visit(E->getBase());

      for (const Expr *S : E->getShapes())
        Visit(S);
    }

    void VisitOSSArraySectionExpr(const OSSArraySectionExpr *E) {
      Visit(E->getBase());

      if (E->getLowerBound())
        Visit(E->getLowerBound());
      if (E->getLengthUpper())
        Visit(E->getLengthUpper());
    }

    void VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
      Visit(E->getBase());
      Visit(E->getIdx());
    }

    void VisitUnaryOperator(const UnaryOperator *E) {
      Visit(E->getSubExpr());
    }

    void VisitMemberExpr(const MemberExpr *E) {
      Visit(E->getBase());
    }

    void VisitDeclRefExpr(const DeclRefExpr *E) {
      if (E->isNonOdrUse() == NOUR_Unevaluated)
        return;

      CodeGenFunction::ConstantEmission CE = CGF.tryEmitAsConstant(const_cast<DeclRefExpr*>(E));
      if (CE && !CE.isReference())
        // Constant value, no need to annotate it.
        return;

      if (const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
        if (VD->hasGlobalStorage())
          SharedGlobals.insert(E);
      }
    }

    ArrayRef<const DeclRefExpr *> getSharedGlobals() const {
      return SharedGlobals.getArrayRef();
    }

  };

  // NOTE: this should do only one iteration
  for (const auto *Attr : FD->specific_attrs<OSSTaskDeclAttr>()) {
    SmallVector<llvm::Value*, 4> CapturedList;
    for (const Expr *E : SharedCopies) {
      EmitDSAShared(CGF, E, TaskInfo, CapturedList);
    }
    for (const Expr *E : PrivateCopies) {
      OSSDSAPrivateDataTy PDataTy;
      PDataTy.Ref = E;
      PDataTy.Copy = nullptr;
      EmitDSAPrivate(CGF, PDataTy, TaskInfo, CapturedList);
    }
    for (const Expr *E : FirstprivateCopies) {
      OSSDSAFirstprivateDataTy FpDataTy;
      // Ignore ImplicitCast built for the new function call
      FpDataTy.Ref = E->IgnoreImpCasts();
      FpDataTy.Copy = FpDataTy.Init = nullptr;
      EmitDSAFirstprivate(CGF, FpDataTy, TaskInfo, CapturedList);
    }

    if (const Expr *E = Attr->getIfExpr()) {
      TaskInfo.emplace_back(getBundleStr(OSSB_if), CGF.EvaluateExprAsBool(E));
    }
    if (const Expr *E = Attr->getFinalExpr()) {
      TaskInfo.emplace_back(getBundleStr(OSSB_final), CGF.EvaluateExprAsBool(E));
    }
    if (const Expr *E = Attr->getCostExpr()) {
      EmitScalarWrapperCallBundle(
        getBundleStr(OSSB_cost), "compute_cost", CGF, E, TaskInfo);
    }
    if (const Expr *E = Attr->getPriorityExpr()) {
      EmitScalarWrapperCallBundle(
        getBundleStr(OSSB_priority), "compute_priority", CGF, E, TaskInfo);
    }
    for (const Expr *E : Attr->labelExprs()) {
      llvm::Value *V = CGF.EmitScalarExpr(E);
      TaskInfo.emplace_back(getBundleStr(OSSB_label), V);
    }
    if (Attr->getWait()) {
      TaskInfo.emplace_back(
          getBundleStr(OSSB_wait),
          llvm::ConstantInt::getTrue(CGM.getLLVMContext()));
    }

    if (Attr->getDevice() != OSSTaskDeclAttr::DeviceType::Unknown) {
      TaskInfo.emplace_back(
          getBundleStr(OSSB_device),
          llvm::ConstantInt::getSigned(
            CGF.ConvertType(CGF.getContext().IntTy),
            convertDeviceTypeToInt(Attr->getDevice())));
    }

    bool IsCudaPure =
      Attr->getDevice() == OSSTaskDeclAttr::DeviceType::Cuda &&
       Attr->ndranges_size() > 0;
    if (Attr->getDevice() == OSSTaskDeclAttr::DeviceType::Fpga || IsCudaPure) {
      TaskInfo.emplace_back(
        getBundleStr(OSSB_device_dev_func),
        llvm::ConstantDataArray::getString(
          CGM.getLLVMContext(), CGF.CGM.getMangledName(GlobalDecl(FD))));
    }
    if (Attr->ndranges_size() > 0) {
      HasNdrange = true;
      SmallVector<llvm::Value *, 4> Result;
      for (const Expr *E : Attr->ndranges()) {
        llvm::Value *V = CGF.EmitScalarExpr(E);
        Result.push_back(V);
      }
      TaskInfo.emplace_back(
          getBundleStr(OSSB_device_ndrange), Result);
    }
    if (Attr->getShmemExpr()) {
      assert(HasNdrange && "shmem clause requires ndrange too");
      llvm::Value *V = CGF.EmitScalarExpr(Attr->getShmemExpr());
      TaskInfo.emplace_back(getBundleStr(OSSB_device_shmem), V);
    }
    if (const Expr *E = Attr->getOnreadyExpr()) {
      EmitIgnoredWrapperCallBundle(
        getBundleStr(OSSB_onready), "compute_onready", CGF, E, TaskInfo);
    }

    // Annotate device call arguments order.
    if (HasNdrange)
      TaskInfo.emplace_back(
          getBundleStr(OSSB_device_call_order), CallValues);

    assert(Attr->reductions_size() == Attr->reductionLHSs_size() &&
           Attr->reductions_size() == Attr->reductionRHSs_size() &&
           Attr->reductions_size() == Attr->reductionOps_size() &&
           Attr->reductions_size() == Attr->reductionKinds_size());
    assert(Attr->reductionListSizes_size() == Attr->reductionClauseType_size() &&
           Attr->reductionListSizes_size() == Attr->nameSpecifierLocs_size() &&
           Attr->reductionListSizes_size() == Attr->declNameInfos_size());

    // Add global variables as shared
    {
      OSSOutlineGlobalVarVisitor OutlineGlobalVarVisitor(CGF);
      for (const Expr *E : Attr->ins()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->outs()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->inouts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->concurrents()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->commutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->weakIns()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->weakOuts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->weakInouts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->weakConcurrents()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->weakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depIns()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depOuts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depInouts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depConcurrents()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakIns()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakOuts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakInouts()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakConcurrents()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      for (const Expr *E : Attr->depWeakCommutatives()) { OutlineGlobalVarVisitor.Visit(E); }
      // Note: reductions over globals are not allowed in task outline
      for (const Expr *E : OutlineGlobalVarVisitor.getSharedGlobals()) {
        const DeclRefExpr *DRE = cast<DeclRefExpr>(E);
        const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
        SmallVector<llvm::Value *> DSABundleList;
        LValue LV = CGF.EmitLValue(DRE);
        llvm::Value *DSAValue = LV.getPointer(CGF);
        if (VD->getType()->isReferenceType() && !getTaskCaptureAddr(VD).isValid())
          CaptureMapStack.back().try_emplace(VD, LV.getAddress(CGF));
        DSABundleList.push_back(DSAValue);
        DSABundleList.push_back(llvm::UndefValue::get(CGF.ConvertType(DRE->getType())));
        TaskInfo.emplace_back(getBundleStr(OSSB_shared), DSABundleList);
      }
    }

    // in()
    for (const Expr *E : Attr->ins()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_in), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->outs()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_out), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->inouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_inout), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->concurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_concurrent), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->commutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_commutative), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakin), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakout), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakinout), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakconcurrent), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakcommutative), CGF, FD, Dep, TaskInfo);
    }
    // depend(in :)
    for (const Expr *E : Attr->depIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_in), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_out), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_inout), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_concurrent), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_commutative), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakin), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakout), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakinout), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakconcurrent), CGF, FD, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency(getBundleStr(OSSB_weakcommutative), CGF, FD, Dep, TaskInfo);
    }

    auto reductions_it = Attr->reductions_begin();
    auto reductionLHSs_it = Attr->reductionLHSs_begin();
    auto reductionRHSs_it = Attr->reductionRHSs_begin();
    auto reductionOps_it = Attr->reductionOps_begin();
    auto reductionKinds_it = Attr->reductionKinds_begin();
    for (unsigned i = 0; i < Attr->reductionListSizes_size(); ++i) {
      OmpSsClauseKind CKind = (OmpSsClauseKind)*(Attr->reductionClauseType_begin() + i);
      for (unsigned j = 0; j < *(Attr->reductionListSizes_begin() + i); ++j) {
        const Expr *Ref = *(reductions_it++);
        const Expr *LHS = *(reductionLHSs_it++);
        const Expr *RHS = *(reductionRHSs_it++);
        const Expr *ReductionOp = *(reductionOps_it++);

        BinaryOperatorKind ReductionKind = (BinaryOperatorKind)*(reductionKinds_it++);
        OSSReductionDataTy Red{Ref, LHS, RHS, ReductionOp, ReductionKind};
        EmitReduction(getBundleStr(CKind == OSSC_weakreduction ? OSSB_weakreduction : OSSB_reduction),
                      getBundleStr(OSSB_redinit),
                      getBundleStr(OSSB_redcomb),
                      CGF, FD, Red, TaskInfo);
      }
    }

    if (!CapturedList.empty())
      TaskInfo.emplace_back(getBundleStr(OSSB_captured), CapturedList);

    CGDebugInfo *DI = CGF.getDebugInfo();
    // Get location information.
    llvm::DebugLoc DbgLoc = DI->SourceLocToDebugLoc(Attr->getLocation());

    StringRef Name = CGM.getModule().getSourceFileName();
    Name = llvm::sys::path::filename(Name);
    TaskInfo.emplace_back(
      getBundleStr(OSSB_decl_source),
      llvm::ConstantDataArray::getString(
        CGM.getLLVMContext(),
        (Name + ":" + Twine(DbgLoc.getLine()) + ":" + Twine(DbgLoc.getCol())).str()));
  }
  InDirectiveEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::ArrayRef(TaskInfo));

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

  // TODO:
  // EmitDirectiveDbgInfo

  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();

  // Pure task outlines (with ndrange) do not have a defined function
  // so do not emit a call to it to avoid linking issues.
  RValue RV = RValue::get(nullptr);
  if (!HasNdrange) {
    // From EmitCallExpr
    if (IsMethodCall) {
      assert(NewME && "Expected a call_arg MemberExpr");

      CXXMemberCallExpr *NewCXXE = CXXMemberCallExpr::Create(
          Ctx, NewME, ParmCopies, Ctx.VoidTy,
          VK_PRValue, SourceLocation(), FPOptionsOverride());

      RV = CGF.EmitCXXMemberCallExpr(NewCXXE, ReturnValue);
    } else {
      // Regular function call
      CGCallee callee = CGF.EmitCallee(CE->getCallee());

      CallExpr *NewCE = CallExpr::Create(
          Ctx, const_cast<Expr *>(CE->getCallee()), ParmCopies,
          Ctx.VoidTy, VK_PRValue, SourceLocation(), FPOptionsOverride());

      RV = CGF.EmitCall(CE->getCallee()->getType(), callee, NewCE, ReturnValue);
    }
  }

  CGF.EHStack.popTerminate();

  // TODO: do we need this? we're pushing a terminate...
  // EmitIfUsed(*this, EHResumeBlock);
  EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
  EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
  EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

  CGF.Builder.CreateCall(ExitCallee, Result);

  // Pop Task Stack
  TaskStack.pop_back();
  CaptureMapStack.pop_back();
  TaskAllocaInsertPt->eraseFromParent();

  return RV;
}

void CGOmpSsRuntime::emitTaskCall(CodeGenFunction &CGF,
                                  const OSSExecutableDirective &D,
                                  SourceLocation Loc,
                                  const OSSTaskDataTy &Data) {

  llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(OSSB_task)));

  // Push Task Stack
  TaskStack.push_back(TaskContext());
  CaptureMapStack.push_back(CaptureMapTy());

  InDirectiveEmission = true;
  EmitDirectiveData(CGF, Data, TaskInfo);
  InDirectiveEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::ArrayRef(TaskInfo));

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

  if (CGM.getCodeGenOpts().hasReducedDebugInfo())
    EmitDirectiveDbgInfo(CGF, Data);

  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  CGF.EmitStmt(D.getAssociatedStmt());
  CGF.EHStack.popTerminate();

  // TODO: do we need this? we're pushing a terminate...
  // EmitIfUsed(*this, EHResumeBlock);
  EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
  EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
  EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

  CGF.Builder.CreateCall(ExitCallee, Result);

  // Pop Task Stack
  TaskStack.pop_back();
  CaptureMapStack.pop_back();
  TaskAllocaInsertPt->eraseFromParent();

}

void CGOmpSsRuntime::emitCriticalCall(CodeGenFunction &CGF,
                                      const OSSExecutableDirective &D,
                                      SourceLocation Loc,
                                      const DeclarationNameInfo &DirName) {
  llvm::Function *Callee = CGM.getIntrinsic(llvm::Intrinsic::directive_marker);

  std::string CriticalName = "default";
  if (!DirName.getAsString().empty())
    CriticalName = DirName.getAsString();

  llvm::Value *CriticalNameValue =
    llvm::ConstantDataArray::getString(
      CGM.getLLVMContext(), (Twine("nanos6_critical_") + CriticalName).str());

  CGF.Builder.CreateCall(
      Callee, {},
      {
        llvm::OperandBundleDef(
          std::string(getBundleStr(OSSB_directive)),
          ArrayRef<llvm::Value *>{
            llvm::ConstantDataArray::getString(
              CGM.getLLVMContext(), getBundleStr(OSSB_critical_start)),
            CriticalNameValue
          })
      });
  CGF.EmitStmt(D.getAssociatedStmt());
  CGF.Builder.CreateCall(
      Callee, {},
      {
        llvm::OperandBundleDef(
          std::string(getBundleStr(OSSB_directive)),
          ArrayRef<llvm::Value *>{
            llvm::ConstantDataArray::getString(
              CGM.getLLVMContext(), getBundleStr(OSSB_critical_end)),
            CriticalNameValue
          })
      });
}

void CGOmpSsRuntime::emitLoopCall(CodeGenFunction &CGF,
                                  const OSSLoopDirective &D,
                                  SourceLocation Loc,
                                  const OSSTaskDataTy &Data,
                                  const OSSLoopDataTy &LoopData) {

  llvm::Function *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Function *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;

  OmpSsBundleKind LoopDirectiveBundleKind = OSSB_unknown;
  if (isa<OSSTaskForDirective>(D)) LoopDirectiveBundleKind = OSSB_task_for;
  else if (isa<OSSTaskIterDirective>(D)) {
    if (isa<WhileStmt>(D.getAssociatedStmt()))
      LoopDirectiveBundleKind = OSSB_taskiter_while;
    else
      LoopDirectiveBundleKind = OSSB_taskiter_for;
  }
  else if (isa<OSSTaskLoopDirective>(D)) LoopDirectiveBundleKind = OSSB_taskloop;
  else if (isa<OSSTaskLoopForDirective>(D)) LoopDirectiveBundleKind = OSSB_taskloop_for;
  else llvm_unreachable("Unexpected loop directive in codegen");

  TaskInfo.emplace_back(
      getBundleStr(OSSB_directive),
      llvm::ConstantDataArray::getString(CGM.getLLVMContext(), getBundleStr(LoopDirectiveBundleKind)));

  // Emit for-init before task entry
  const Stmt *Body = D.getAssociatedStmt();
  const Expr *WhileCond = nullptr;
  if (const WhileStmt *While = dyn_cast<WhileStmt>(Body)) {
    // At this stage only taskiter (while) should end up
    // here.
    Body = While->getBody();
    WhileCond = While->getCond();

    // Push Task Stack
    TaskStack.push_back(TaskContext());
    CaptureMapStack.push_back(CaptureMapTy());

    InDirectiveEmission = true;
    EmitDirectiveData(CGF, Data, TaskInfo, LoopData, WhileCond);
    InDirectiveEmission = false;

    llvm::Instruction *Result =
      CGF.Builder.CreateCall(EntryCallee, {}, llvm::ArrayRef(TaskInfo));

    llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
    llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
    setTaskInsertPt(TaskAllocaInsertPt);

    if (CGM.getCodeGenOpts().hasReducedDebugInfo())
      EmitDirectiveDbgInfo(CGF, Data);

    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CGF.EHStack.pushTerminate();
    CGF.EmitStmt(Body);
    CGF.EHStack.popTerminate();

    // TODO: do we need this? we're pushing a terminate...
    // EmitIfUsed(*this, EHResumeBlock);
    EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
    EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
    EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

    CGF.Builder.CreateCall(ExitCallee, Result);

    // Pop Task Stack
    TaskStack.pop_back();
    CaptureMapStack.pop_back();
    TaskAllocaInsertPt->eraseFromParent();
  } else {
    CodeGenFunction::LexicalScope ForScope(CGF, cast<ForStmt>(D.getAssociatedStmt())->getSourceRange());

    for (size_t i = 0; i < D.getNumCollapses(); ++i) {
      const ForStmt *For = cast<ForStmt>(Body);
      CGF.EmitStmt(For->getInit());
      Body = For->getBody();
      if (i + 1 < D.getNumCollapses()) {
        for (const Stmt *Child : Body->children()) {
          if ((Body = dyn_cast<ForStmt>(Child)))
            break;
        }
      }
    }
    // Push Task Stack
    TaskStack.push_back(TaskContext());
    CaptureMapStack.push_back(CaptureMapTy());

    InDirectiveEmission = true;
    EmitDirectiveData(CGF, Data, TaskInfo, LoopData);
    InDirectiveEmission = false;

    llvm::Instruction *Result =
      CGF.Builder.CreateCall(EntryCallee, {}, llvm::ArrayRef(TaskInfo));

    llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
    llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
    setTaskInsertPt(TaskAllocaInsertPt);

    if (CGM.getCodeGenOpts().hasReducedDebugInfo())
      EmitDirectiveDbgInfo(CGF, Data);

    // The point of exit cannot be a branch out of the structured block.
    // longjmp() and throw() must not violate the entry/exit criteria.
    CGF.EHStack.pushTerminate();
    CGF.EmitStmt(Body);
    CGF.EHStack.popTerminate();

    // TODO: do we need this? we're pushing a terminate...
    // EmitIfUsed(*this, EHResumeBlock);
    EmitIfUsed(CGF, TaskStack.back().TerminateLandingPad);
    EmitIfUsed(CGF, TaskStack.back().TerminateHandler);
    EmitIfUsed(CGF, TaskStack.back().UnreachableBlock);

    CGF.Builder.CreateCall(ExitCallee, Result);

    // Pop Task Stack
    TaskStack.pop_back();
    CaptureMapStack.pop_back();
    TaskAllocaInsertPt->eraseFromParent();
  }
}

/// Emit code for 'flush' "directive".
void CGOmpSsRuntime::emitFlush(CodeGenFunction &CGF, llvm::AtomicOrdering AO) {
  llvm::Function *Callee = CGM.getIntrinsic(llvm::Intrinsic::directive_marker);
  llvm::Instruction *Call = CGF.Builder.CreateCall(Callee);
  new llvm::FenceInst(CGF.CGM.getLLVMContext(), AO, llvm::SyncScope::System, Call);
  Call->eraseFromParent();
}

void CGOmpSsRuntime::addMetadata(ArrayRef<llvm::Metadata *> List) {
  MetadataList.append(List.begin(), List.end());
}

llvm::MDNode *CGOmpSsRuntime::getMetadataNode() {
  if (MetadataList.empty())
    return nullptr;
  return llvm::MDTuple::get(CGM.getLLVMContext(), MetadataList);
}

llvm::AtomicOrdering CGOmpSsRuntime::getDefaultMemoryOrdering() const {
  return RequiresAtomicOrdering;
}
