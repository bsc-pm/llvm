//===- StmtOmpSs.h - Classes for OmpSs directives  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines OmpSs AST classes for executable directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTOMPSS_H
#define LLVM_CLANG_AST_STMTOMPSS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OmpSsClause.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for directives.
//===----------------------------------------------------------------------===//

/// This is a basic class for representing single OmpSs executable
/// directive.
///
class OSSExecutableDirective : public Stmt {
  friend class ASTStmtReader;
  /// Kind of the directive.
  OmpSsDirectiveKind Kind;
  /// Starting location of the directive (directive keyword).
  SourceLocation StartLoc;
  /// Ending location of the directive.
  SourceLocation EndLoc;

  /// Get the clauses storage.
  MutableArrayRef<OSSClause *> getClauses() {
    if (!Data)
      return std::nullopt;
    return Data->getClauses();
  }
protected:
  /// Data, associated with the directive.
  OSSChildren *Data = nullptr;

  /// Build instance of directive of class \a K.
  ///
  /// \param SC Statement class.
  /// \param K Kind of OmpSs directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  ///
  OSSExecutableDirective(StmtClass SC, OmpSsDirectiveKind K,
                         SourceLocation StartLoc, SourceLocation EndLoc)
      : Stmt(SC), Kind(K), StartLoc(std::move(StartLoc)),
        EndLoc(std::move(EndLoc)) {}

  template <typename T, typename... Params>
  static T *createDirective(const ASTContext &C, ArrayRef<OSSClause *> Clauses,
                            Stmt *AssociatedStmt, unsigned NumChildren,
                            Params &&... P) {
    void *Mem =
        C.Allocate(sizeof(T) + OSSChildren::size(Clauses.size(), AssociatedStmt,
                                                 NumChildren),
                   alignof(T));

    auto *Data = OSSChildren::Create(reinterpret_cast<T *>(Mem) + 1, Clauses,
                                     AssociatedStmt, NumChildren);
    auto *Inst = new (Mem) T(std::forward<Params>(P)...);
    Inst->Data = Data;
    return Inst;
  }

  template <typename T, typename... Params>
  static T *createEmptyDirective(const ASTContext &C, unsigned NumClauses,
                                 bool HasAssociatedStmt, unsigned NumChildren,
                                 Params &&... P) {
    void *Mem =
        C.Allocate(sizeof(T) + OSSChildren::size(NumClauses, HasAssociatedStmt,
                                                 NumChildren),
                   alignof(T));
    auto *Data =
        OSSChildren::CreateEmpty(reinterpret_cast<T *>(Mem) + 1, NumClauses,
                                 HasAssociatedStmt, NumChildren);
    auto *Inst = new (Mem) T(std::forward<Params>(P)...);
    Inst->Data = Data;
    return Inst;
  }

  template <typename T>
  static T *createEmptyDirective(const ASTContext &C, unsigned NumClauses,
                                 bool HasAssociatedStmt = false,
                                 unsigned NumChildren = 0) {
    void *Mem =
        C.Allocate(sizeof(T) + OSSChildren::size(NumClauses, HasAssociatedStmt,
                                                 NumChildren),
                   alignof(T));
    auto *Data =
        OSSChildren::CreateEmpty(reinterpret_cast<T *>(Mem) + 1, NumClauses,
                                 HasAssociatedStmt, NumChildren);
    auto *Inst = new (Mem) T;
    Inst->Data = Data;
    return Inst;
  }

public:
  /// Iterates over a filtered subrange of clauses applied to a
  /// directive.
  ///
  /// This iterator visits only clauses of type SpecificClause.
  template <typename SpecificClause>
  class specific_clause_iterator
      : public llvm::iterator_adaptor_base<
            specific_clause_iterator<SpecificClause>,
            ArrayRef<OSSClause *>::const_iterator, std::forward_iterator_tag,
            const SpecificClause *, ptrdiff_t, const SpecificClause *,
            const SpecificClause *> {
    ArrayRef<OSSClause *>::const_iterator End;

    void SkipToNextClause() {
      while (this->I != End && !isa<SpecificClause>(*this->I))
        ++this->I;
    }

  public:
    explicit specific_clause_iterator(ArrayRef<OSSClause *> Clauses)
        : specific_clause_iterator::iterator_adaptor_base(Clauses.begin()),
          End(Clauses.end()) {
      SkipToNextClause();
    }

    const SpecificClause *operator*() const {
      return cast<SpecificClause>(*this->I);
    }
    const SpecificClause *operator->() const { return **this; }

    specific_clause_iterator &operator++() {
      ++this->I;
      SkipToNextClause();
      return *this;
    }
  };

  template <typename SpecificClause>
  static llvm::iterator_range<specific_clause_iterator<SpecificClause>>
  getClausesOfKind(ArrayRef<OSSClause *> Clauses) {
    return {specific_clause_iterator<SpecificClause>(Clauses),
            specific_clause_iterator<SpecificClause>(
                llvm::ArrayRef(Clauses.end(), (size_t)0))};
  }

  template <typename SpecificClause>
  llvm::iterator_range<specific_clause_iterator<SpecificClause>>
  getClausesOfKind() const {
    return getClausesOfKind<SpecificClause>(clauses());
  }

  /// Gets a single clause of the specified kind associated with the
  /// current directive iff there is only one clause of this kind (and assertion
  /// is fired if there is more than one clause is associated with the
  /// directive). Returns nullptr if no clause of this kind is associated with
  /// the directive.
  template <typename SpecificClause>
  const SpecificClause *getSingleClause() const {
    auto Clauses = getClausesOfKind<SpecificClause>();

    if (Clauses.begin() != Clauses.end()) {
      assert(std::next(Clauses.begin()) == Clauses.end() &&
             "There are at least 2 clauses of the specified kind");
      return *Clauses.begin();
    }
    return nullptr;
  }

  /// Returns true if the current directive has one or more clauses of a
  /// specific kind.
  template <typename SpecificClause>
  bool hasClausesOfKind() const {
    auto Clauses = getClausesOfKind<SpecificClause>();
    return Clauses.begin() != Clauses.end();
  }

  /// Returns starting location of directive kind.
  SourceLocation getBeginLoc() const { return StartLoc; }
  /// Returns ending location of directive.
  SourceLocation getEndLoc() const { return EndLoc; }

  /// Set starting location of directive kind.
  ///
  /// \param Loc New starting location of directive.
  ///
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  /// Set ending location of directive.
  ///
  /// \param Loc New ending location of directive.
  ///
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  /// Get number of clauses.
  unsigned getNumClauses() const {
    if (!Data)
      return 0;
    return Data->getNumClauses();
  }

  /// Returns specified clause.
  ///
  /// \param i Number of clause.
  ///
  OSSClause *getClause(unsigned i) const { return clauses()[i]; }

  /// Returns true if directive has associated statement.
  bool hasAssociatedStmt() const { return Data && Data->hasAssociatedStmt(); }

  /// Returns statement associated with the directive.
  const Stmt *getAssociatedStmt() const {
    assert(hasAssociatedStmt() && "no associated statement.");
    return *child_begin();
  }
  Stmt *getAssociatedStmt() {
    assert(hasAssociatedStmt() && "no associated statement.");
    return *child_begin();
  }

  OmpSsDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstOSSExecutableDirectiveConstant &&
           S->getStmtClass() <= lastOSSExecutableDirectiveConstant;
  }

  child_range children() {
    if (!Data)
      return child_range(child_iterator(), child_iterator());
    return Data->getAssociatedStmtAsRange();
  }

  const_child_range children() const {
    return const_cast<OSSExecutableDirective *>(this)->children();
  }

  ArrayRef<OSSClause *> clauses() const {
    if (!Data)
      return std::nullopt;
    return Data->getClauses();
  }
};

/// This represents '#pragma oss taskwait' directive.
///
/// \code
/// #pragma oss taskwait
/// \endcode
///
class OSSTaskwaitDirective : public OSSExecutableDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSTaskwaitDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSExecutableDirective(OSSTaskwaitDirectiveClass, llvm::oss::OSSD_taskwait,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSTaskwaitDirective()
      : OSSExecutableDirective(OSSTaskwaitDirectiveClass, llvm::oss::OSSD_taskwait,
                               SourceLocation(), SourceLocation()) {}

public:
  /// Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  ///
  static OSSTaskwaitDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OSSClause *> Clauses);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OSSTaskwaitDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                           EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskwaitDirectiveClass;
  }
};

/// This represents '#pragma oss release' directive.
///
/// \code
/// #pragma oss release
/// \endcode
///
class OSSReleaseDirective : public OSSExecutableDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSReleaseDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSExecutableDirective(OSSReleaseDirectiveClass, llvm::oss::OSSD_release,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSReleaseDirective()
      : OSSExecutableDirective(OSSReleaseDirectiveClass, llvm::oss::OSSD_release,
                               SourceLocation(), SourceLocation()) {}

public:
  /// Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  ///
  static OSSReleaseDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OSSClause *> Clauses);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OSSReleaseDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                           EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSReleaseDirectiveClass;
  }
};

class OSSTaskDirective : public OSSExecutableDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSTaskDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSExecutableDirective(OSSTaskDirectiveClass, llvm::oss::OSSD_task,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSTaskDirective()
      : OSSExecutableDirective(OSSTaskDirectiveClass, llvm::oss::OSSD_task,
                               SourceLocation(), SourceLocation()) {}

public:
  /// Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  ///
  static OSSTaskDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OSSClause *> Clauses,
                                  Stmt *AStmt);

  /// Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OSSTaskDirective *CreateEmpty(const ASTContext &C, unsigned NumClauses,
                                       EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskDirectiveClass;
  }
};

/// This represents '#pragma oss critical' directive.
///
/// \code
/// #pragma oss critical
/// \endcode
///
class OSSCriticalDirective : public OSSExecutableDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Name of the directive.
  DeclarationNameInfo DirName;
  /// Build directive with the given start and end location.
  ///
  /// \param Name Name of the directive.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSCriticalDirective(const DeclarationNameInfo &Name, SourceLocation StartLoc,
                       SourceLocation EndLoc)
      : OSSExecutableDirective(OSSCriticalDirectiveClass,
                               llvm::oss::OSSD_critical, StartLoc, EndLoc),
        DirName(Name) {}

  /// Build an empty directive.
  ///
  explicit OSSCriticalDirective()
      : OSSExecutableDirective(OSSCriticalDirectiveClass,
                               llvm::oss::OSSD_critical, SourceLocation(),
                               SourceLocation()) {}

  /// Set name of the directive.
  ///
  /// \param Name Name of the directive.
  ///
  void setDirectiveName(const DeclarationNameInfo &Name) { DirName = Name; }

public:
  /// Creates directive.
  ///
  /// \param C AST context.
  /// \param Name Name of the directive.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static OSSCriticalDirective *
  Create(const ASTContext &C, const DeclarationNameInfo &Name,
         SourceLocation StartLoc, SourceLocation EndLoc,
         ArrayRef<OSSClause *> Clauses, Stmt *AssociatedStmt);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OSSCriticalDirective *CreateEmpty(const ASTContext &C,
                                           unsigned NumClauses, EmptyShell);

  /// Return name of the directive.
  ///
  DeclarationNameInfo getDirectiveName() const { return DirName; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSCriticalDirectiveClass;
  }
};

/// This is a common base class for loop directives ('oss task for',
/// oss taskloop', 'oss taskloop for' etc.).
/// It is responsible for the loop code generation.
///
class OSSLoopDirective : public OSSExecutableDirective {
  friend class ASTStmtReader;
  /// The Induction variable of the loop directive
  Expr **IndVarExpr;
  /// The lower bound of the loop directive
  Expr **LowerBoundExpr;
  /// The upper bound of the loop directive
  Expr **UpperBoundExpr;
  /// The step of the loop directive
  Expr **StepExpr;
  /// The type of comparison used in the loop (<, <=, >=, >)
  /// NOTE: optional is used to handle the != eventually
  std::optional<bool>* TestIsLessOp;
  /// The type of comparison is strict (<, >)
  bool* TestIsStrictOp;
  /// The number of collapsed loops
  unsigned NumCollapses;
protected:
  /// Build instance of loop directive of class \a Kind.
  ///
  /// \param SC Statement class.
  /// \param Kind Kind of OpenMP directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  OSSLoopDirective(StmtClass SC, OmpSsDirectiveKind Kind,
                   SourceLocation StartLoc, SourceLocation EndLoc,
                   unsigned NumCollapses)
      : OSSExecutableDirective(SC, Kind, StartLoc, EndLoc),
        NumCollapses(NumCollapses)
        {}

  /// Sets the iteration variable used in the loop.
  ///
  /// \param IV The induction variable expression.
  ///
  void setIterationVariable(Expr **IV) { IndVarExpr = IV; }
  /// Sets the lower bound used in the loop.
  ///
  /// \param LB The lower bound expression.
  ///
  void setLowerBound(Expr **LB) { LowerBoundExpr = LB; }
  /// Sets the upper bound used in the loop.
  ///
  /// \param UB The upper bound expression.
  ///
  void setUpperBound(Expr **LB) { UpperBoundExpr = LB; }
  /// Sets the step used in the loop.
  ///
  /// \param Step The step expression.
  ///
  void setStep(Expr **Step) { StepExpr = Step; }
  /// Sets the loop comparison type.
  ///
  /// \param IsLessOp True if is < or <=. false otherwise
  ///
  void setIsLessOp(std::optional<bool> *IsLessOp) {
    TestIsLessOp = IsLessOp;
  }
  /// Sets if the loop comparison type is strict.
  ///
  /// \param IsStrict True < or >. false otherwise
  ///
  void setIsStrictOp(bool *IsStrictOp) { TestIsStrictOp = IsStrictOp; }

public:
  struct HelperExprs {
    Expr *IndVar;
    Expr *LB;
    Expr *UB;
    Expr *Step;
    std::optional<bool> TestIsLessOp;
    bool TestIsStrictOp;
  };

  /// Returns the induction variable expression of the loop.
  Expr **getIterationVariable() const { return IndVarExpr; }
  /// Returns the lower bound expression of the loop.
  Expr **getLowerBound() const { return LowerBoundExpr; }
  /// Returns the upper bound expression of the loop.
  Expr **getUpperBound() const { return UpperBoundExpr; }
  /// Returns the step expression of the loop.
  Expr **getStep() const { return StepExpr; }
  /// Returns True is the loop comparison type is < or <=.
  std::optional<bool> *getIsLessOp() const { return TestIsLessOp; }
  /// Returns True is the loop comparison type is < or >.
  bool *getIsStrictOp() const { return TestIsStrictOp; }

  unsigned getNumCollapses() const { return NumCollapses; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskForDirectiveClass ||
           T->getStmtClass() == OSSTaskLoopDirectiveClass ||
           T->getStmtClass() == OSSTaskLoopForDirectiveClass;
  }
};

class OSSTaskForDirective : public OSSLoopDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  OSSTaskForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                      unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskForDirectiveClass, llvm::oss::OSSD_task_for,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskForDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskForDirectiveClass, llvm::oss::OSSD_task_for,
                         SourceLocation(), SourceLocation(), NumCollapses) {}

public:
  /// Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OSSTaskForDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OSSClause *> Clauses,
                                  Stmt *AStmt,
                                  const SmallVectorImpl<HelperExprs> &Exprs);

  /// Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  static OSSTaskForDirective *CreateEmpty(
      const ASTContext &C, unsigned NumClauses,
      unsigned NumCollapses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskForDirectiveClass;
  }
};

class OSSTaskIterDirective : public OSSLoopDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  OSSTaskIterDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                      unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskIterDirectiveClass, llvm::oss::OSSD_taskiter,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  ///
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskIterDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskIterDirectiveClass, llvm::oss::OSSD_taskiter,
                         SourceLocation(), SourceLocation(), NumCollapses) {}

public:
  /// Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OSSTaskIterDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OSSClause *> Clauses,
                                  Stmt *AStmt,
                                  const SmallVectorImpl<HelperExprs> &Exprs);

  /// Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  static OSSTaskIterDirective *CreateEmpty(
      const ASTContext &C, unsigned NumClauses,
      unsigned NumCollapses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskIterDirectiveClass;
  }
};

class OSSTaskLoopDirective : public OSSLoopDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSTaskLoopDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                       unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskLoopDirectiveClass, llvm::oss::OSSD_taskloop,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskLoopDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskLoopDirectiveClass, llvm::oss::OSSD_taskloop,
                         SourceLocation(), SourceLocation(), NumCollapses) {}

public:
  /// Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OSSTaskLoopDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OSSClause *> Clauses,
                                  Stmt *AStmt,
                                  const SmallVectorImpl<HelperExprs> &Exprs);

  /// Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  static OSSTaskLoopDirective *CreateEmpty(
      const ASTContext &C, unsigned NumClauses,
      unsigned NumCollapses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskLoopDirectiveClass;
  }
};

class OSSTaskLoopForDirective : public OSSLoopDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  OSSTaskLoopForDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                          unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskLoopForDirectiveClass, llvm::oss::OSSD_taskloop_for,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskLoopForDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskLoopForDirectiveClass, llvm::oss::OSSD_taskloop_for,
                         SourceLocation(), SourceLocation(), NumCollapses) {}

public:
  /// Creates directive with a list of \a Clauses.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AStmt Statement, associated with the directive.
  /// \param Exprs Helper expressions for CodeGen.
  ///
  static OSSTaskLoopForDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                  SourceLocation EndLoc,
                                  ArrayRef<OSSClause *> Clauses,
                                  Stmt *AStmt,
                                  const SmallVectorImpl<HelperExprs> &Exprs);

  /// Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  static OSSTaskLoopForDirective *CreateEmpty(
      const ASTContext &C, unsigned NumClauses,
      unsigned NumCollapses, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskLoopForDirectiveClass;
  }
};

/// This represents '#pragma omp atomic' directive.
///
/// \code
/// #pragma oss atomic capture
/// \endcode
/// In this example directive '#pragma oss atomic' has clause 'capture'.
///
class OSSAtomicDirective : public OSSExecutableDirective {
  friend class ASTStmtReader;
  friend class OSSExecutableDirective;

  struct FlagTy {
    /// Used for 'atomic update' or 'atomic capture' constructs. They may
    /// have atomic expressions of forms:
    /// \code
    /// x = x binop expr;
    /// x = expr binop x;
    /// \endcode
    /// This field is 1 for the first form of the expression and 0 for the
    /// second. Required for correct codegen of non-associative operations (like
    /// << or >>).
    uint8_t IsXLHSInRHSPart : 1;
    /// Used for 'atomic update' or 'atomic capture' constructs. They may
    /// have atomic expressions of forms:
    /// \code
    /// v = x; <update x>;
    /// <update x>; v = x;
    /// \endcode
    /// This field is 1 for the first(postfix) form of the expression and 0
    /// otherwise.
    uint8_t IsPostfixUpdate : 1;
    /// 1 if 'v' is updated only when the condition is false (compare capture
    /// only).
    uint8_t IsFailOnly : 1;
  } Flags;

  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSAtomicDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSExecutableDirective(OSSAtomicDirectiveClass, llvm::oss::OSSD_atomic,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSAtomicDirective()
      : OSSExecutableDirective(OSSAtomicDirectiveClass, llvm::oss::OSSD_atomic,
                               SourceLocation(), SourceLocation()) {}

  enum DataPositionTy : size_t {
    POS_X = 0,
    POS_V,
    POS_E,
    POS_UpdateExpr,
    POS_D,
    POS_Cond,
    POS_R,
  };

  /// Set 'x' part of the associated expression/statement.
  void setX(Expr *X) { Data->getChildren()[DataPositionTy::POS_X] = X; }
  /// Set helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  void setUpdateExpr(Expr *UE) {
    Data->getChildren()[DataPositionTy::POS_UpdateExpr] = UE;
  }
  /// Set 'v' part of the associated expression/statement.
  void setV(Expr *V) { Data->getChildren()[DataPositionTy::POS_V] = V; }
  /// Set 'r' part of the associated expression/statement.
  void setR(Expr *R) { Data->getChildren()[DataPositionTy::POS_R] = R; }
  /// Set 'expr' part of the associated expression/statement.
  void setExpr(Expr *E) { Data->getChildren()[DataPositionTy::POS_E] = E; }
  /// Set 'd' part of the associated expression/statement.
  void setD(Expr *D) { Data->getChildren()[DataPositionTy::POS_D] = D; }
  /// Set conditional expression in `atomic compare`.
  void setCond(Expr *C) { Data->getChildren()[DataPositionTy::POS_Cond] = C; }

public:
  struct Expressions {
    /// 'x' part of the associated expression/statement.
    Expr *X = nullptr;
    /// 'v' part of the associated expression/statement.
    Expr *V = nullptr;
    // 'r' part of the associated expression/statement.
    Expr *R = nullptr;
    /// 'expr' part of the associated expression/statement.
    Expr *E = nullptr;
    /// UE Helper expression of the form:
    /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
    /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
    Expr *UE = nullptr;
    /// 'd' part of the associated expression/statement.
    Expr *D = nullptr;
    /// Conditional expression in `atomic compare` construct.
    Expr *Cond = nullptr;
    /// True if UE has the first form and false if the second.
    bool IsXLHSInRHSPart;
    /// True if original value of 'x' must be stored in 'v', not an updated one.
    bool IsPostfixUpdate;
    /// True if 'v' is updated only when the condition is false (compare capture
    /// only).
    bool IsFailOnly;
  };

  /// Creates directive with a list of \a Clauses and 'x', 'v' and 'expr'
  /// parts of the atomic construct (see Section 2.12.6, atomic Construct, for
  /// detailed description of 'x', 'v' and 'expr').
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param Clauses List of clauses.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param Exprs Associated expressions or statements.
  static OSSAtomicDirective *Create(const ASTContext &C,
                                    SourceLocation StartLoc,
                                    SourceLocation EndLoc,
                                    ArrayRef<OSSClause *> Clauses,
                                    Stmt *AssociatedStmt, Expressions Exprs);

  /// Creates an empty directive with the place for \a NumClauses
  /// clauses.
  ///
  /// \param C AST context.
  /// \param NumClauses Number of clauses.
  ///
  static OSSAtomicDirective *CreateEmpty(const ASTContext &C,
                                         unsigned NumClauses, EmptyShell);

  /// Get 'x' part of the associated expression/statement.
  Expr *getX() {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_X]);
  }
  const Expr *getX() const {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_X]);
  }
  /// Get helper expression of the form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' or
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  Expr *getUpdateExpr() {
    return cast_or_null<Expr>(
        Data->getChildren()[DataPositionTy::POS_UpdateExpr]);
  }
  const Expr *getUpdateExpr() const {
    return cast_or_null<Expr>(
        Data->getChildren()[DataPositionTy::POS_UpdateExpr]);
  }
  /// Return true if helper update expression has form
  /// 'OpaqueValueExpr(x) binop OpaqueValueExpr(expr)' and false if it has form
  /// 'OpaqueValueExpr(expr) binop OpaqueValueExpr(x)'.
  bool isXLHSInRHSPart() const { return Flags.IsXLHSInRHSPart; }
  /// Return true if 'v' expression must be updated to original value of
  /// 'x', false if 'v' must be updated to the new value of 'x'.
  bool isPostfixUpdate() const { return Flags.IsPostfixUpdate; }
  /// Return true if 'v' is updated only when the condition is evaluated false
  /// (compare capture only).
  bool isFailOnly() const { return Flags.IsFailOnly; }
  /// Get 'v' part of the associated expression/statement.
  Expr *getV() {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_V]);
  }
  const Expr *getV() const {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_V]);
  }
  /// Get 'r' part of the associated expression/statement.
  Expr *getR() {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_R]);
  }
  const Expr *getR() const {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_R]);
  }
  /// Get 'expr' part of the associated expression/statement.
  Expr *getExpr() {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_E]);
  }
  const Expr *getExpr() const {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_E]);
  }
  /// Get 'd' part of the associated expression/statement.
  Expr *getD() {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_D]);
  }
  Expr *getD() const {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_D]);
  }
  /// Get the 'cond' part of the source atomic expression.
  Expr *getCondExpr() {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_Cond]);
  }
  Expr *getCondExpr() const {
    return cast_or_null<Expr>(Data->getChildren()[DataPositionTy::POS_Cond]);
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSAtomicDirectiveClass;
  }
};

} // end namespace clang

#endif
