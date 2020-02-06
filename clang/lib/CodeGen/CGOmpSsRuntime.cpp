//===----- CGOmpSsRuntime.cpp - Interface to OmpSs Runtimes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/BitmaskEnum.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;

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
      if (const ConstantArrayType *BaseArrayTy = CGF.getContext().getAsConstantArrayType(Q)) {
        return CGF.getContext().getBaseElementType(Q);
      } else if (const VariableArrayType *BaseArrayTy = CGF.getContext().getAsVariableArrayType(Q)) {
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
    Ptr = CGF.EmitLValue(E).getPointer();
    if (E->getType()->isVariablyModifiedType()) {
      CGF.EmitVariablyModifiedType(E->getType());
    }
    FillDimsFromInnermostExpr(E);
  }

  // l-values.
  void VisitDeclRefExpr(const DeclRefExpr *E) {
    Ptr = CGF.EmitDeclRefLValue(E).getPointer();
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
        break;
      }
    }
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
    Ptr = CGF.EmitMemberExpr(E).getPointer();
    FillBaseExprDimsAndType(E);
  }

  void VisitUnaryDeref(const UnaryOperator *E) {
    Ptr = CGF.EmitUnaryOpLValue(E).getPointer();
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

static void EmitCopyCtorFunc(CodeGenModule &CGM,
                             llvm::Value *V,
                             const CXXConstructExpr *CtorE,
                             const VarDecl *CopyD,
                             const VarDecl *InitD,
                             SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {
  const std::string BundleCopyName = "QUAL.OSS.COPY";
  const CXXConstructorDecl *CtorD = cast<CXXConstructorDecl>(CtorE->getConstructor());
  // If we have already created the function we're done
  auto It = CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs.find(CtorD);
  if (It != CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(BundleCopyName, ArrayRef<llvm::Value*>{V, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(CopyD->getType());
  ImplicitParamDecl SrcArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamDecl::Other);

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
  LValue DstLV = CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(&DstArg),
                                             PQ->castAs<PointerType>());
  llvm::Value *NelemsValue =
      CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&NelemsArg), /*Volatile=*/false,
                       NelemsArg.getType(), NelemsArg.getLocation());

  // Find the end of the array.
  llvm::Value *SrcBegin = SrcLV.getPointer();
  llvm::Value *DstBegin = DstLV.getPointer();
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(DstBegin, NelemsValue,
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
    InitScope.addPrivate(InitD, [SrcCur, SrcLV]() -> Address {

      return Address(SrcCur, SrcLV.getAlignment());
    });
    (void)InitScope.Privatize();
    CGF.EmitExprAsInit(CtorE, CopyD,
                       CGF.MakeAddrLValue(DstCur, DstLV.getType(), DstLV.getAlignment()),
                       /*capturedByInit=*/false);
  }

  // Go to the next element. Move SrcBegin too
  llvm::Value *DstNext =
    CGF.Builder.CreateInBoundsGEP(DstCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arrayctor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  llvm::Value *SrcDest =
    CGF.Builder.CreateInBoundsGEP(SrcCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                                  "arrayctor.src.next");
  SrcCur->addIncoming(SrcDest, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs[CtorD] = Fn;

  TaskInfo.emplace_back(BundleCopyName, ArrayRef<llvm::Value*>{V, Fn});

}

static void EmitCtorFunc(CodeGenModule &CGM,
                         llvm::Value *V,
                         const VarDecl *CopyD,
                         SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {
  const std::string BundleInitName = "QUAL.OSS.INIT";
  const CXXConstructExpr *CtorE = cast<CXXConstructExpr>(CopyD->getInit());
  const CXXConstructorDecl *CtorD = cast<CXXConstructorDecl>(CtorE->getConstructor());

  GlobalDecl CtorGD(CtorD, Ctor_Complete);
  llvm::Value *CtorValue = CGM.getAddrOfCXXStructor(CtorGD);

  auto It = CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs.find(CtorD);
  if (It != CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(BundleInitName, ArrayRef<llvm::Value*>{V, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(CopyD->getType());
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamDecl::Other);

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
  llvm::Value *DstBegin = DstLV.getPointer();
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(DstBegin, NelemsValue,
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
  llvm::Value *DstNext =
    CGF.Builder.CreateInBoundsGEP(DstCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arrayctor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arrayctor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arrayctor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs[CtorD] = Fn;

  TaskInfo.emplace_back(BundleInitName, ArrayRef<llvm::Value*>{V, Fn});
}

static void EmitDtorFunc(CodeGenModule &CGM,
                         llvm::Value *V,
                         const VarDecl *CopyD,
                         SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {
  const std::string BundleDeinitName = "QUAL.OSS.DEINIT";

  QualType Q = CopyD->getType();

  const RecordType *RT = Q->getAs<RecordType>();
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());

  if (RD->hasTrivialDestructor())
    return;

  const CXXDestructorDecl *DtorD = RD->getDestructor();

  GlobalDecl DtorGD(DtorD, Dtor_Complete);
  llvm::Value *DtorValue = CGM.getAddrOfCXXStructor(DtorGD);

  auto It = CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs.find(DtorD);
  if (It != CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs.end()) {
    TaskInfo.emplace_back(BundleDeinitName, ArrayRef<llvm::Value*>{V, It->second});
    return;
  }

  ASTContext &C = CGM.getContext();
  FunctionArgList Args;

  QualType PQ = C.getPointerType(Q);
  ImplicitParamDecl DstArg(C, PQ, ImplicitParamDecl::Other);
  ImplicitParamDecl NelemsArg(C, C.getSizeType(), ImplicitParamDecl::Other);

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
  llvm::Value *DstBegin = DstLV.getPointer();
  llvm::Value *DstEnd = CGF.Builder.CreateInBoundsGEP(DstBegin, NelemsValue,
                                                      "arraydtor.dst.end");

  // Enter the loop, setting up a phi for the current location to initialize.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("arraydtor.loop");
  CGF.EmitBlock(LoopBB);
  llvm::PHINode *DstCur = CGF.Builder.CreatePHI(DstBegin->getType(), 2,
                                             "arraydtor.dst.cur");
  DstCur->addIncoming(DstBegin, EntryBB);

  CGF.EmitCXXDestructorCall(DtorD, Dtor_Complete,
                         /*ForVirtualBase=*/false, /*Delegating=*/false,
                         Address(DstCur, DstLV.getAlignment()), Q);

  // Go to the next element
  llvm::Value *DstNext =
    CGF.Builder.CreateInBoundsGEP(DstCur, llvm::ConstantInt::get(CGF.ConvertType(C.getSizeType()), 1),
                              "arraydtor.dst.next");
  DstCur->addIncoming(DstNext, CGF.Builder.GetInsertBlock());

  // Check whether that's the end of the loop.
  llvm::Value *Done = CGF.Builder.CreateICmpEQ(DstNext, DstEnd, "arraydtor.done");
  llvm::BasicBlock *ContBB = CGF.createBasicBlock("arraydtor.cont");
  CGF.Builder.CreateCondBr(Done, ContBB, LoopBB);

  CGF.EmitBlock(ContBB);

  CGF.FinishFunction();

  CGM.getOmpSsRuntime().GenericCXXNonPodMethodDefs[DtorD] = Fn;

  TaskInfo.emplace_back(BundleDeinitName, ArrayRef<llvm::Value*>{V, Fn});
}

static void EmitDSAShared(
  CodeGenFunction &CGF, const Expr *E,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList,
  llvm::DenseMap<const VarDecl *, Address> &RefMap) {

  const std::string BundleName = "QUAL.OSS.SHARED";

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
    llvm::Value *V;
    if (VD->getType()->isReferenceType()) {
      // Record Ref Address to be reused in task body and other clasues
      LValue LV = CGF.EmitDeclRefLValue(DRE);
      RefMap.try_emplace(VD, LV.getAddress());
      V = LV.getPointer();
      TaskInfo.emplace_back(BundleName, V);
    } else {
      V = CGF.EmitDeclRefLValue(DRE).getPointer();
      TaskInfo.emplace_back(BundleName, V);
    }
    QualType Q = VD->getType();
    // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
    bool IsPtr = Q->isPointerType();
    SmallVector<llvm::Value *, 4> DimsWithValue;
    while (Q->isPointerType()) {
      Q = Q->getPointeeType();
    }
    if (Q->isVariableArrayType())
      GatherVLADims(CGF, V, Q, DimsWithValue, CapturedList, IsPtr);

    if (!DimsWithValue.empty())
      TaskInfo.emplace_back("QUAL.OSS.VLA.DIMS", DimsWithValue);

  } else if (const CXXThisExpr *ThisE = dyn_cast<CXXThisExpr>(E)) {
    TaskInfo.emplace_back(BundleName, CGF.EmitScalarExpr(ThisE));
  } else {
    llvm_unreachable("Unhandled expression");
  }
}

static void EmitDSAPrivate(
  CodeGenFunction &CGF, const OSSDSAPrivateDataTy &PDataTy,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList,
  llvm::DenseMap<const VarDecl *, Address> &RefMap) {

  const std::string BundleName = "QUAL.OSS.PRIVATE";

  const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(PDataTy.Ref);
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  llvm::Value *V;
  if (VD->getType()->isReferenceType()) {
    // Record Ref Address to be reused in task body and other clauses
    LValue LV = CGF.EmitDeclRefLValue(DRE);
    RefMap.try_emplace(VD, LV.getAddress());
    V = LV.getPointer();
    TaskInfo.emplace_back(BundleName, V);
  } else {
    V = CGF.EmitDeclRefLValue(DRE).getPointer();
    TaskInfo.emplace_back(BundleName, V);
  }
  QualType Q = VD->getType();
  // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
  bool IsPtr = Q->isPointerType();
  SmallVector<llvm::Value *, 4> DimsWithValue;
  while (Q->isPointerType()) {
    Q = Q->getPointeeType();
  }
  if (Q->isVariableArrayType())
    GatherVLADims(CGF, V, Q, DimsWithValue, CapturedList, IsPtr);

  if (!DimsWithValue.empty())
    TaskInfo.emplace_back("QUAL.OSS.VLA.DIMS", DimsWithValue);

  const DeclRefExpr *CopyE = cast<DeclRefExpr>(PDataTy.Copy);
  const VarDecl *CopyD = cast<VarDecl>(CopyE->getDecl());

  if (!CopyD->getType().isPODType(CGF.getContext())) {
    EmitCtorFunc(CGF.CGM, V, CopyD, TaskInfo);
    EmitDtorFunc(CGF.CGM, V, CopyD, TaskInfo);
  }
}

static void EmitDSAFirstprivate(
  CodeGenFunction &CGF, const OSSDSAFirstprivateDataTy &FpDataTy,
  SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
  SmallVectorImpl<llvm::Value*> &CapturedList,
  llvm::DenseMap<const VarDecl *, Address> &RefMap) {

  const std::string BundleName = "QUAL.OSS.FIRSTPRIVATE";

  const DeclRefExpr *DRE = cast<DeclRefExpr>(FpDataTy.Ref);
  const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
  llvm::Value *V;
  if (VD->getType()->isReferenceType()) {
    // Record Ref Address to be reused in task body and other clauses
    LValue LV = CGF.EmitDeclRefLValue(DRE);
    RefMap.try_emplace(VD, LV.getAddress());
    V = LV.getPointer();
    TaskInfo.emplace_back(BundleName, V);
  } else {
    V = CGF.EmitDeclRefLValue(DRE).getPointer();
    TaskInfo.emplace_back(BundleName, V);
  }
  QualType Q = VD->getType();
  // int (**p)[sizex][sizey] -> we need to capture sizex sizey only
  bool IsPtr = Q->isPointerType();
  SmallVector<llvm::Value *, 4> DimsWithValue;
  while (Q->isPointerType()) {
    Q = Q->getPointeeType();
  }
  if (Q->isVariableArrayType())
    GatherVLADims(CGF, V, Q, DimsWithValue, CapturedList, IsPtr);

  if (!DimsWithValue.empty())
    TaskInfo.emplace_back("QUAL.OSS.VLA.DIMS", DimsWithValue);

  // TODO: is this sufficient to skip COPY/DEINIT? (task functions)
  if (FpDataTy.Copy) {
    const DeclRefExpr *CopyE = cast<DeclRefExpr>(FpDataTy.Copy);
    const VarDecl *CopyD = cast<VarDecl>(CopyE->getDecl());
    const DeclRefExpr *InitE = cast<DeclRefExpr>(FpDataTy.Init);
    const VarDecl *InitD = cast<VarDecl>(InitE->getDecl());

    if (!CopyD->getType().isPODType(CGF.getContext())) {
      const CXXConstructExpr *CtorE = cast<CXXConstructExpr>(CopyD->getAnyInitializer());

      EmitCopyCtorFunc(CGF.CGM, V, CtorE, CopyD, InitD, TaskInfo);
      EmitDtorFunc(CGF.CGM, V, CopyD, TaskInfo);
    }
  }
}

static void EmitDependency(StringRef Name, CodeGenFunction &CGF, const OSSDepDataTy &Dep,
                           SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo) {

  // C long -> LLVM long
  llvm::Type *OSSArgTy = CGF.ConvertType(CGF.getContext().LongTy);

  OSSDependVisitor DepVisitor(CGF, Dep.OSSSyntax);
  DepVisitor.Visit(Dep.E);

  SmallVector<llvm::Value*, 4> DepData;

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
             CGF.CGM
               .getDataLayout()
               .getTypeSizeInBits(CGF
                                  .ConvertType(BaseElementTy))/8;

  DepData.push_back(Ptr);
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
      Dim = CGF.Builder.CreateMul(Dim,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
      IdxStart = CGF.Builder.CreateMul(IdxStart,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
      IdxEnd = CGF.Builder.CreateMul(IdxEnd,
                                  llvm::ConstantInt::getSigned(OSSArgTy,
                                                               BaseElementSize));
    }
    DepData.push_back(Dim);
    DepData.push_back(IdxStart);
    DepData.push_back(IdxEnd);
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
    Dim = CGF.Builder.CreateMul(Dim,
                                llvm::ConstantInt::getSigned(OSSArgTy,
                                                             BaseElementSize));
    IdxStart = CGF.Builder.CreateMul(IdxStart,
                                llvm::ConstantInt::getSigned(OSSArgTy,
                                                             BaseElementSize));
    IdxEnd = CGF.Builder.CreateMul(IdxEnd,
                                llvm::ConstantInt::getSigned(OSSArgTy,
                                                             BaseElementSize));
  }
  DepData.push_back(Dim);
  DepData.push_back(IdxStart);
  DepData.push_back(IdxEnd);

  TaskInfo.emplace_back(Name, makeArrayRef(DepData));
}

void CGOmpSsRuntime::emitTaskwaitCall(CodeGenFunction &CGF,
                                      SourceLocation Loc) {
  llvm::Value *Callee = CGM.getIntrinsic(llvm::Intrinsic::directive_marker);
  CGF.Builder.CreateCall(Callee,
                         {},
                         {llvm::OperandBundleDef("DIR.OSS",
                                                 llvm::ConstantDataArray::getString(CGM.getLLVMContext(),
                                                                                    "TASKWAIT"))});
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

llvm::DenseMap<const VarDecl *, Address> &CGOmpSsRuntime::getTaskRefMap() {
  return TaskStack.back().RefMap;
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
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

RValue CGOmpSsRuntime::emitTaskFunction(CodeGenFunction &CGF,
                                        const FunctionDecl *FD,
                                        const CallExpr *CE,
                                        ReturnValueSlot ReturnValue) {
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Ctx = CGM.getContext();

  SourceLocation Loc = CE->getBeginLoc();

  SmallVector<Expr *, 4> ParmCopies;
  SmallVector<Expr *, 4> FirstprivateCopies;
  SmallVector<Expr *, 4> SharedCopies;

  CodeGenFunction::OSSPrivateScope InitScope(CGF);

  InTaskEmission = true;
  // NOTE: this differs a bit from the regular use where InTaskEmission
  // means building bundles, and inTaskBody() meaning emiting the code
  // of the task. Since we build the call_arg before that and RefMap
  // is inside TaskContext we do this here.
  TaskStack.push_back(TaskContext());

  auto ArgI = CE->arg_begin();
  auto ParI = FD->param_begin(); 
  while (ArgI != CE->arg_end()) {

    // QualType ParQ = (*ParI)->getType();
    QualType ParQ = (*ParI)->getType().getUnqualifiedType();
    if (ParQ->isArrayType())
      ParQ = Ctx.getBaseElementType(ParQ).getCanonicalType();

    // The a new VarDecl like ParamArgDecl, but in context of function call
    auto *ParmDecl =
      VarDecl::Create(Ctx,
                      const_cast<DeclContext *>(cast<DeclContext>(CGF.CurCodeDecl)),
                      Loc,
                      Loc,
                      &Ctx.Idents.get("call_arg"),
                      ParQ,
                      Ctx.getTrivialTypeSourceInfo(ParQ, Loc),
                      SC_Auto);
    ParmDecl->setImplicit();
    ParmDecl->setReferenced();
    ParmDecl->markUsed(Ctx);
    ParmDecl->setInitStyle(VarDecl::CInit);
    ParmDecl->setInit(const_cast<Expr *>(*ArgI));

    CGF.EmitVarDecl(*ParmDecl);

    Expr *ParmRef = DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), SourceLocation(), ParmDecl,
        /*RefersToEnclosingVariableOrCapture=*/false, Loc, ParQ.getNonReferenceType(), VK_LValue);

    if (!(*ParI)->getType()->isReferenceType()) {
      ParmRef =
        ImplicitCastExpr::Create(Ctx, ParmRef->getType(), CK_LValueToRValue,
                                 ParmRef, /*BasePath=*/nullptr,
                                 VK_RValue);
      FirstprivateCopies.push_back(ParmRef);
    } else {
      // We want to pass references as shared so task can modify the original value
      SharedCopies.push_back(ParmRef);
    }
    ParmCopies.push_back(ParmRef);

    LValue ParmLV = CGF.EmitLValue(ParmRef);

    InitScope.addPrivate(*ParI, [&ParmLV]() -> Address { return ParmLV.getAddress(); });
    // We do need to do this every time because the next param may use the previous one
    (void)InitScope.Privatize();

    ++ArgI;
    ++ParI;
  }
  llvm::Value *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Value *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back("DIR.OSS", llvm::ConstantDataArray::getString(CGM.getLLVMContext(), "TASK"));

  bool IsMethodCall = false;
  if (const auto *CXXE = dyn_cast<CXXMemberCallExpr>(CE)) {
    IsMethodCall = true;
    const Expr *callee = CXXE->getCallee()->IgnoreParens();
    const MemberExpr *ME = cast<MemberExpr>(callee);
    const Expr *Base = ME->getBase();
    LValue This = CGF.EmitLValue(Base);
    TaskInfo.emplace_back("QUAL.OSS.FIRSTPRIVATE", This.getPointer());
  }

  // NOTE: this should do only one iteration
  for (const auto *Attr : FD->specific_attrs<OSSTaskDeclAttr>()) {
    SmallVector<llvm::Value*, 4> CapturedList;
    for (const Expr *E : SharedCopies) {
      EmitDSAShared(CGF, E, TaskInfo, CapturedList, TaskStack.back().RefMap);
    }
    for (const Expr *E : FirstprivateCopies) {
      OSSDSAFirstprivateDataTy FpDataTy;
      // Ignore ImplicitCast built for the new function call
      FpDataTy.Ref = E->IgnoreImpCasts();
      FpDataTy.Copy = FpDataTy.Init = nullptr;
      EmitDSAFirstprivate(CGF, FpDataTy, TaskInfo, CapturedList, TaskStack.back().RefMap);
    }
    if (const Expr *E = Attr->getIfExpr()) {
      TaskInfo.emplace_back("QUAL.OSS.IF", CGF.EvaluateExprAsBool(E));
    }
    if (const Expr *E = Attr->getFinalExpr()) {
      TaskInfo.emplace_back("QUAL.OSS.FINAL", CGF.EvaluateExprAsBool(E));
    }
    if (const Expr *E = Attr->getCostExpr()) {
      llvm::Value *V = CGF.EmitScalarExpr(E);
      CapturedList.push_back(V);
      TaskInfo.emplace_back("QUAL.OSS.COST", V);
    }
    if (const Expr *E = Attr->getPriorityExpr()) {
      llvm::Value *V = CGF.EmitScalarExpr(E);
      CapturedList.push_back(V);
      TaskInfo.emplace_back("QUAL.OSS.PRIORITY", V);
    }
    // in()
    for (const Expr *E : Attr->ins()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.IN", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->outs()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.OUT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->inouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.INOUT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->concurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.CONCURRENT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->commutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.COMMUTATIVE", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.WEAKIN", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.WEAKOUT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->weakInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = true;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.WEAKINOUT", CGF, Dep, TaskInfo);
    }
    // depend(in :)
    for (const Expr *E : Attr->depIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.IN", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.OUT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.INOUT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depConcurrents()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.CONCURRENT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depCommutatives()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.COMMUTATIVE", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakIns()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.WEAKIN", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakOuts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.WEAKOUT", CGF, Dep, TaskInfo);
    }
    for (const Expr *E : Attr->depWeakInouts()) {
      OSSDepDataTy Dep;
      Dep.OSSSyntax = false;
      Dep.E = E;
      EmitDependency("QUAL.OSS.DEP.WEAKINOUT", CGF, Dep, TaskInfo);
    }
    if (!CapturedList.empty())
      TaskInfo.emplace_back("QUAL.OSS.CAPTURED", CapturedList);
  }
  InTaskEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::makeArrayRef(TaskInfo));

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

  // From EmitCallExpr
  RValue RV;
  if (IsMethodCall) {
    RV = CGF.EmitCXXMemberCallExpr(cast<CXXMemberCallExpr>(CE), ReturnValue);
  } else {
    // Regular function call
    CGCallee callee = CGF.EmitCallee(CE->getCallee());

    CallExpr *NewCE = CallExpr::Create(Ctx, const_cast<Expr *>(CE->getCallee()), ParmCopies,
                                       Ctx.VoidTy, VK_RValue, SourceLocation());

    RV = CGF.EmitCall(CE->getCallee()->getType(), callee, NewCE, ReturnValue);
  }

  CGF.Builder.CreateCall(ExitCallee, Result);

  // Pop Task Stack
  TaskStack.pop_back();
  TaskAllocaInsertPt->eraseFromParent();

  return RV;
}

void CGOmpSsRuntime::emitTaskCall(CodeGenFunction &CGF,
                                  const OSSExecutableDirective &D,
                                  SourceLocation Loc,
                                  const OSSTaskDataTy &Data) {

  llvm::Value *EntryCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_entry);
  llvm::Value *ExitCallee = CGM.getIntrinsic(llvm::Intrinsic::directive_region_exit);
  SmallVector<llvm::OperandBundleDef, 8> TaskInfo;
  TaskInfo.emplace_back("DIR.OSS", llvm::ConstantDataArray::getString(CGM.getLLVMContext(), "TASK"));

  SmallVector<llvm::Value*, 4> CapturedList;

  TaskStack.push_back(TaskContext());

  InTaskEmission = true;
  for (const Expr *E : Data.DSAs.Shareds) {
    EmitDSAShared(CGF, E, TaskInfo, CapturedList, TaskStack.back().RefMap);
  }
  for (const OSSDSAPrivateDataTy &PDataTy : Data.DSAs.Privates) {
    EmitDSAPrivate(CGF, PDataTy, TaskInfo, CapturedList, getTaskRefMap());
  }
  for (const OSSDSAFirstprivateDataTy &FpDataTy : Data.DSAs.Firstprivates) {
    EmitDSAFirstprivate(CGF, FpDataTy, TaskInfo, CapturedList, getTaskRefMap());
  }

  if (Data.Cost) {
    llvm::Value *V = CGF.EmitScalarExpr(Data.Cost);
    CapturedList.push_back(V);
    TaskInfo.emplace_back("QUAL.OSS.COST", V);
  }
  if (Data.Priority) {
    llvm::Value *V = CGF.EmitScalarExpr(Data.Priority);
    CapturedList.push_back(V);
    TaskInfo.emplace_back("QUAL.OSS.PRIORITY", V);
  }

  if (!CapturedList.empty())
    TaskInfo.emplace_back("QUAL.OSS.CAPTURED", CapturedList);

  for (const OSSDepDataTy &Dep : Data.Deps.Ins) {
    EmitDependency("QUAL.OSS.DEP.IN", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Outs) {
    EmitDependency("QUAL.OSS.DEP.OUT", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Inouts) {
    EmitDependency("QUAL.OSS.DEP.INOUT", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Concurrents) {
    EmitDependency("QUAL.OSS.DEP.CONCURRENT", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.Commutatives) {
    EmitDependency("QUAL.OSS.DEP.COMMUTATIVE", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakIns) {
    EmitDependency("QUAL.OSS.DEP.WEAKIN", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakOuts) {
    EmitDependency("QUAL.OSS.DEP.WEAKOUT", CGF, Dep, TaskInfo);
  }
  for (const OSSDepDataTy &Dep : Data.Deps.WeakInouts) {
    EmitDependency("QUAL.OSS.DEP.WEAKINOUT", CGF, Dep, TaskInfo);
  }

  if (Data.If)
    TaskInfo.emplace_back("QUAL.OSS.IF", CGF.EvaluateExprAsBool(Data.If));
  if (Data.Final)
    TaskInfo.emplace_back("QUAL.OSS.FINAL", CGF.EvaluateExprAsBool(Data.Final));

  InTaskEmission = false;

  llvm::Instruction *Result =
    CGF.Builder.CreateCall(EntryCallee, {}, llvm::makeArrayRef(TaskInfo));

  // Push Task Stack
  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  llvm::Instruction *TaskAllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "taskallocapt", Result->getParent());
  setTaskInsertPt(TaskAllocaInsertPt);

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
  TaskAllocaInsertPt->eraseFromParent();

}
