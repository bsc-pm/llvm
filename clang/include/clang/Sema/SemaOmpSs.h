//===----- SemaOmpSs.h -- Semantic Analysis for OmpSs constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for OmpSs constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAOMPSS_H
#define LLVM_CLANG_SEMA_SEMAOMPSS_H

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclOmpSs.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprOmpSs.h"
#include "clang/AST/OmpSsClause.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include <optional>
#include <string>
#include <utility>

namespace clang {
class MultiLevelTemplateArgumentList;

class SemaOmpSs : public SemaBase {
public:
  SemaOmpSs(Sema &S);

  friend class Parser;
  friend class Sema;

  using DeclGroupPtrTy = OpaquePtr<DeclGroupRef>;

public:

  // OmpSs
  ExprResult ActOnOSSArraySectionExpr(Expr *Base, SourceLocation LBLoc,
                                      Expr *LowerBound, SourceLocation ColonLoc,
                                      Expr *LengthUpper, SourceLocation RBLoc,
                                      bool ColonForm = true);
  ExprResult ActOnOSSArrayShapingExpr(Expr *Base, ArrayRef<Expr *> Shapes,
                                      SourceLocation LBLoc,
                                      SourceLocation RBLoc);

  // OmpSs-2
  // Same behaviour as CheckShadow, but different diag messages.
  void OSSCheckShadow(
    NamedDecl *D, NamedDecl *ShadowedDecl, const LookupResult &R);
  void OSSCheckShadow(Scope *S, VarDecl *D);

  // Used in template instantiation
  void InstantiateOSSDeclareTaskAttr(
    const MultiLevelTemplateArgumentList &TemplateArgs,
    const OSSTaskDeclAttr &Attr, Decl *New);

  // This RAII manages the scope where we allow array shaping expressions
  class AllowShapingsRAII {
    SemaOmpSs &SemaOmpSsRef;
  public:
    AllowShapingsRAII(SemaOmpSs &S, llvm::function_ref<bool()> CondFun)
      : SemaOmpSsRef(S)
    { SemaOmpSsRef.AllowShapings = CondFun(); }
    ~AllowShapingsRAII() { SemaOmpSsRef.AllowShapings = false; }
  };
  ExprResult
  VerifyBooleanConditionWithCleanups(Expr *Condition,
                                     SourceLocation StartLoc);
  ExprResult
  CheckNonNegativeIntegerValue(Expr *ValExpr,
                               OmpSsClauseKind CKind,
                               bool StrictlyPositive,
                               bool Outline);
  ExprResult
  CheckSignedIntegerValue(Expr *ValExpr, bool Outline);
  ExprResult
  CheckIsConstCharPtrConvertibleExpr(Expr *E, bool ConstConstraint = false);
  ExprResult
  VerifyPositiveIntegerConstant(Expr *E,
                                OmpSsClauseKind CKind,
                                bool StrictlyPositive);
  OmpSsDirectiveKind GetCurrentOmpSsDirective() const;
  // Used to distinguish between for and while taskiter
  void SetTaskiterKind(OmpSsDirectiveKind);
  bool IsEndOfTaskloop() const;
  ExprResult PerformOmpSsImplicitIntegerConversion(SourceLocation OpLoc,
                                                   Expr *Op);
  /// Called on start of new data sharing attribute block.
  void StartOmpSsDSABlock(OmpSsDirectiveKind K,
                          Scope *CurScope,
                          SourceLocation Loc);
  /// Called on end of data sharing attribute block.
  void EndOmpSsDSABlock(Stmt *CurDirective);

  // Check if there are conflicts between depend() and other DSA clauses
  // This must happen before parsing the statement (if any) so child tasks
  // can see DSA derived from 'depend' clauses
  void ActOnOmpSsAfterClauseGathering(SmallVectorImpl<OSSClause *>& Clauses);

  /// Check if the current region is an OmpSs loop region and if it is,
  /// mark loop control variable, used in \p Init for loop initialization, as
  /// private by default.
  /// \param Init First part of the for loop.
  void ActOnOmpSsLoopInitialization(SourceLocation ForLoc, Stmt *Init);

  Expr *ActOnOmpSsMultiDepIterator(Scope *S, StringRef Name, SourceLocation Loc);
  ExprResult ActOnOmpSsMultiDepIteratorInitListExpr(InitListExpr *InitList);
  ExprResult ActOnOSSMultiDepExpression(
    SourceLocation Loc, SourceLocation RLoc, ArrayRef<Expr *> MultiDepIterators,
    ArrayRef<Expr *> MultiDepInits, ArrayRef<Expr *> MultiDepSizes,
    ArrayRef<Expr *> MultiDepSteps, ArrayRef<bool> MultiDepSizeOrSection,
    Expr *DepExpr);

  /// Called on well-formed '#pragma oss assert'.
  DeclGroupPtrTy ActOnOmpSsAssertDirective(SourceLocation Loc, Expr *E);

  /// Check if the specified type is allowed to be used in 'oss declare
  /// reduction' construct.
  QualType ActOnOmpSsDeclareReductionType(SourceLocation TyLoc,
                                          TypeResult ParsedType);
  /// Called on start of '#pragma oss declare reduction'.
  DeclGroupPtrTy ActOnOmpSsDeclareReductionDirectiveStart(
      Scope *S, DeclContext *DC, DeclarationName Name,
      ArrayRef<std::pair<QualType, SourceLocation>> ReductionTypes,
      AccessSpecifier AS, Decl *PrevDeclInScope = nullptr);
  /// Initialize declare reduction construct initializer.
  void ActOnOmpSsDeclareReductionCombinerStart(Scope *S, Decl *D);
  /// Finish current declare reduction construct initializer.
  void ActOnOmpSsDeclareReductionCombinerEnd(Decl *D, Expr *Combiner);
  /// Initialize declare reduction construct initializer.
  /// \return oss_priv variable.
  VarDecl *ActOnOmpSsDeclareReductionInitializerStart(Scope *S, Decl *D);
  /// Finish current declare reduction construct initializer.
  void ActOnOmpSsDeclareReductionInitializerEnd(Decl *D, Expr *Initializer,
                                                VarDecl *OssPrivParm);
  /// Called at the end of '#pragma oss declare reduction'.
  DeclGroupPtrTy ActOnOmpSsDeclareReductionDirectiveEnd(
      Scope *S, DeclGroupPtrTy DeclReductions, bool IsValid);


  // Used to push a fake FunctionScopeInfo
  void ActOnOmpSsExecutableDirectiveStart();
  // Used to pop the fake FunctionScopeInfo
  void ActOnOmpSsExecutableDirectiveEnd();
  StmtResult ActOnOmpSsExecutableDirective(ArrayRef<OSSClause *> Clauses,
      const DeclarationNameInfo &DirName, OmpSsDirectiveKind Kind, Stmt *AStmt,
      SourceLocation StartLoc, SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss taskwait'.
  StmtResult ActOnOmpSsTaskwaitDirective(ArrayRef<OSSClause *> Clauses,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss taskwait'.
  StmtResult ActOnOmpSsReleaseDirective(ArrayRef<OSSClause *> Clauses,
                                        SourceLocation StartLoc,
                                        SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss task' after parsing of the
  /// associated statement.
  StmtResult ActOnOmpSsTaskDirective(ArrayRef<OSSClause *> Clauses,
                                     Stmt *AStmt,
                                     SourceLocation StartLoc,
                                     SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss critical' after parsing of the
  /// associated statement.
  StmtResult ActOnOmpSsCriticalDirective(const DeclarationNameInfo &DirName,
                                         ArrayRef<OSSClause *> Clauses,
                                         Stmt *AStmt, SourceLocation StartLoc,
                                         SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss task for' after parsing of the
  /// associated statement.
  StmtResult
  ActOnOmpSsTaskForDirective(ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                             SourceLocation StartLoc, SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss taskiter' after parsing of the
  /// associated statement.
  StmtResult
  ActOnOmpSsTaskIterDirective(ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                             SourceLocation StartLoc, SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss taskloop' after parsing of the
  /// associated statement.
  StmtResult
  ActOnOmpSsTaskLoopDirective(ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                              SourceLocation StartLoc, SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss taskloop for' after parsing of the
  /// associated statement.
  StmtResult
  ActOnOmpSsTaskLoopForDirective(ArrayRef<OSSClause *> Clauses, Stmt *AStmt,
                                 SourceLocation StartLoc, SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss atomic' after parsing of the
  /// associated statement.
  StmtResult ActOnOmpSsAtomicDirective(ArrayRef<OSSClause *> Clauses,
                                       Stmt *AStmt, SourceLocation StartLoc,
                                       SourceLocation EndLoc);

  /// Called on well-formed '\#pragma oss task' after parsing of
  /// the associated method/function.
  DeclGroupPtrTy ActOnOmpSsDeclareTaskDirective(
      DeclGroupPtrTy DG,
      Expr *Immediate, Expr *Microtask,
      Expr *If, Expr *Final, Expr *Cost, Expr *Priority,
      Expr *Shmem, Expr *Onready, bool Wait,
      unsigned Device, SourceLocation DeviceLoc,
      ArrayRef<Expr *> Labels,
      ArrayRef<Expr *> Ins, ArrayRef<Expr *> Outs, ArrayRef<Expr *> Inouts,
      ArrayRef<Expr *> Concurrents, ArrayRef<Expr *> Commutatives,
      ArrayRef<Expr *> WeakIns, ArrayRef<Expr *> WeakOuts,
      ArrayRef<Expr *> WeakInouts,
      ArrayRef<Expr *> WeakConcurrents, ArrayRef<Expr *> WeakCommutatives,
      ArrayRef<Expr *> DepIns, ArrayRef<Expr *> DepOuts, ArrayRef<Expr *> DepInouts,
      ArrayRef<Expr *> DepConcurrents, ArrayRef<Expr *> DepCommutatives,
      ArrayRef<Expr *> DepWeakIns, ArrayRef<Expr *> DepWeakOuts,
      ArrayRef<Expr *> DepWeakInouts,
      ArrayRef<Expr *> DepWeakConcurrents, ArrayRef<Expr *> DepWeakCommutatives,
      ArrayRef<unsigned> ReductionListSizes,
      ArrayRef<Expr *> Reductions,
      ArrayRef<unsigned> ReductionClauseType,
      ArrayRef<CXXScopeSpec> ReductionCXXScopeSpecs,
      ArrayRef<DeclarationNameInfo> ReductionIds,
      ArrayRef<Expr *> Ndranges, SourceLocation NdrangeLoc,
      SourceRange SR,
      ArrayRef<Expr *> UnresolvedReductions = std::nullopt);

  OSSClause *ActOnOmpSsVarListClause(
      OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
      SourceLocation StartLoc, SourceLocation LParenLoc,
      SourceLocation ColonLoc, SourceLocation EndLoc,
      ArrayRef<OmpSsDependClauseKind> DepKinds,
      SourceLocation DepLoc,
      CXXScopeSpec &ReductionIdScopeSpec,
      DeclarationNameInfo &ReductionId);

  OSSClause *ActOnOmpSsFixedListClause(
      OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
      SourceLocation StartLoc, SourceLocation LParenLoc,
      SourceLocation EndLoc);

  OSSClause *ActOnOmpSsSimpleClause(OmpSsClauseKind Kind,
                                    unsigned Argument,
                                    SourceLocation ArgumentLoc,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);

  /// Called on well-formed 'default' clause.
  OSSClause *ActOnOmpSsDefaultClause(llvm::oss::DefaultKind Kind,
                                      SourceLocation KindLoc,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);

  /// Called on well-formed 'device' clause.
  OSSClause *ActOnOmpSsDeviceClause(OmpSsDeviceClauseKind Kind,
                                      SourceLocation KindLoc,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);

  /// Called on well-formed 'shared' clause.
  /// isImplicit is used to handle CXXThisExpr generated from the compiler
  OSSClause *ActOnOmpSsSharedClause(ArrayRef<Expr *> Vars,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc,
                                    bool isImplicit=false);

  /// Called on well-formed 'private' clause.
  OSSClause *ActOnOmpSsPrivateClause(ArrayRef<Expr *> Vars,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);

  /// Called on well-formed 'firstprivate' clause.
  OSSClause *ActOnOmpSsFirstprivateClause(ArrayRef<Expr *> Vars,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc);

  /// Called on well-formed 'ndrange' clause.
  OSSClause *ActOnOmpSsNdrangeClause(ArrayRef<Expr *> Vars,
                                     SourceLocation StartLoc,
                                     SourceLocation LParenLoc,
                                     SourceLocation EndLoc);

  // Checks depend kinds for errors
  // if no errors DepKindsOrdered is like
  // { OSSC_DEPEND_in, OSSC_DEPEND_weak }
  // { OSSC_DEPEND_in }
  bool ActOnOmpSsDependKinds(ArrayRef<OmpSsDependClauseKind> DepKinds,
                             SmallVectorImpl<OmpSsDependClauseKind> &DepKindsOrdered,
                             SourceLocation DepLoc);

  /// Called on well-formed 'depend' clause.
  OSSClause *
  ActOnOmpSsDependClause(ArrayRef<OmpSsDependClauseKind> DepKinds, SourceLocation DepLoc,
                          SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
                          SourceLocation StartLoc, SourceLocation LParenLoc,
                          SourceLocation EndLoc, bool OSSSyntax = false);

  /// Called on well-formed 'depend' clause.
  OSSClause *
  ActOnOmpSsReductionClause(OmpSsClauseKind Kind, ArrayRef<Expr *> VarList,
                         SourceLocation StartLoc, SourceLocation LParenLoc,
                         SourceLocation ColonLoc,
                         SourceLocation EndLoc,
                         CXXScopeSpec &ReductionIdScopeSpec,
                         const DeclarationNameInfo &ReductionId,
                         ArrayRef<Expr *> UnresolvedReductions = std::nullopt);

  OSSClause *ActOnOmpSsSingleExprClause(OmpSsClauseKind Kind,
                                        Expr *Expr,
                                        SourceLocation StartLoc,
                                        SourceLocation LParenLoc,
                                        SourceLocation EndLoc);

  /// Called on well-formed 'immediate' clause.
  OSSClause *ActOnOmpSsImmediateClause(Expr *Condition, SourceLocation StartLoc,
                                SourceLocation LParenLoc,
                                SourceLocation EndLoc);
  /// Called on well-formed 'microtask' clause.
  OSSClause *ActOnOmpSsMicrotaskClause(Expr *Condition, SourceLocation StartLoc,
                                SourceLocation LParenLoc,
                                SourceLocation EndLoc);
  /// Called on well-formed 'if' clause.
  OSSClause *ActOnOmpSsIfClause(Expr *Condition, SourceLocation StartLoc,
                                SourceLocation LParenLoc,
                                SourceLocation EndLoc);
  /// Called on well-formed 'final' clause.
  OSSClause *ActOnOmpSsFinalClause(Expr *Condition, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'cost' clause.
  OSSClause *ActOnOmpSsCostClause(Expr *E, SourceLocation StartLoc,
                                  SourceLocation LParenLoc,
                                  SourceLocation EndLoc);
  /// Called on well-formed 'priority' clause.
  OSSClause *ActOnOmpSsPriorityClause(Expr *E, SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);
  /// Called on well-formed 'label' clause.
  OSSClause *ActOnOmpSsLabelClause(ArrayRef<Expr *> VarList, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'shmem' clause.
  OSSClause *ActOnOmpSsShmemClause(Expr *E, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'onready' clause.
  OSSClause *ActOnOmpSsOnreadyClause(Expr *E, SourceLocation StartLoc,
                                   SourceLocation LParenLoc,
                                   SourceLocation EndLoc);
  /// Called on well-formed 'chunksize' clause.
  OSSClause *ActOnOmpSsChunksizeClause(Expr *E, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);
  /// Called on well-formed 'grainsize' clause.
  OSSClause *ActOnOmpSsGrainsizeClause(Expr *E, SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc);

  /// Called on well-formed 'unroll' clause.
  OSSClause *ActOnOmpSsUnrollClause(Expr *E, SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc);

  /// Called on well-formed 'collapse' clause.
  OSSClause *ActOnOmpSsCollapseClause(Expr *E, SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc);

  OSSClause *ActOnOmpSsClause(OmpSsClauseKind Kind, SourceLocation StartLoc,
                              SourceLocation EndLoc);
  /// Called on well-formed 'wait' clause.
  OSSClause *ActOnOmpSsWaitClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'update' clause.
  OSSClause *ActOnOmpSsUpdateClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'read' clause.
  OSSClause *ActOnOmpSsReadClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'write' clause.
  OSSClause *ActOnOmpSsWriteClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'capture' clause.
  OSSClause *ActOnOmpSsCaptureClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'compare' clause.
  OSSClause *ActOnOmpSsCompareClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'seq_cst' clause.
  OSSClause *ActOnOmpSsSeqCstClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'acq_rel' clause.
  OSSClause *ActOnOmpSsAcqRelClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'acquire' clause.
  OSSClause *ActOnOmpSsAcquireClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'release' clause.
  OSSClause *ActOnOmpSsReleaseClause(SourceLocation StartLoc,
                                    SourceLocation EndLoc);
  /// Called on well-formed 'relaxed' clause.
  OSSClause *ActOnOmpSsRelaxedClause(SourceLocation StartLoc,
                                     SourceLocation EndLoc);
private:
  void *VarDataSharingAttributesStackOmpSs;

  /// Initialization of data-sharing attributes stack.
  void InitDataSharingAttributesStackOmpSs();
  void DestroyDataSharingAttributesStackOmpSs();

  bool AllowShapings;

};


} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAOMPSS_H

