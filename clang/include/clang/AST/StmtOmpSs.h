//===- StmtOmpSs.h - Classes for OmpSs directives  ------------*- C++ -*-===//
//
//                     The LLVM Cossiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
                llvm::makeArrayRef(Clauses.end(), 0))};
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
      : OSSExecutableDirective(OSSTaskwaitDirectiveClass, OSSD_taskwait,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSTaskwaitDirective()
      : OSSExecutableDirective(OSSTaskwaitDirectiveClass, OSSD_taskwait,
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
      : OSSExecutableDirective(OSSReleaseDirectiveClass, OSSD_release,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSReleaseDirective()
      : OSSExecutableDirective(OSSReleaseDirectiveClass, OSSD_release,
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
      : OSSExecutableDirective(OSSTaskDirectiveClass, OSSD_task,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSTaskDirective()
      : OSSExecutableDirective(OSSTaskDirectiveClass, OSSD_task,
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
  llvm::Optional<bool>* TestIsLessOp;
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
  void setIsLessOp(llvm::Optional<bool> *IsLessOp) {
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
    llvm::Optional<bool> TestIsLessOp;
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
  llvm::Optional<bool> *getIsLessOp() const { return TestIsLessOp; }
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
      : OSSLoopDirective(OSSTaskForDirectiveClass, OSSD_task_for,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskForDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskForDirectiveClass, OSSD_task_for,
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
      : OSSLoopDirective(OSSTaskIterDirectiveClass, OSSD_taskiter,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  ///
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskIterDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskIterDirectiveClass, OSSD_taskiter,
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
      : OSSLoopDirective(OSSTaskLoopDirectiveClass, OSSD_taskloop,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskLoopDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskLoopDirectiveClass, OSSD_taskloop,
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
      : OSSLoopDirective(OSSTaskLoopForDirectiveClass, OSSD_taskloop_for,
                         StartLoc, EndLoc, NumCollapses) {}

  /// Build an empty directive.
  /// \param NumCollapses Number of collapsed loops from 'collapse' clause.
  ///
  explicit OSSTaskLoopForDirective(unsigned NumCollapses)
      : OSSLoopDirective(OSSTaskLoopForDirectiveClass, OSSD_taskloop_for,
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

} // end namespace clang

#endif
