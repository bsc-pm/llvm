//===- OmpSsClause.h - Classes for OmpSs clauses --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file defines OmpSs AST classes for clauses.
/// There are clauses for executable directives, clauses for declarative
/// directives and clauses which can be used in both kinds of directives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_OMPSSCLAUSE_H
#define LLVM_CLANG_AST_OMPSSCLAUSE_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtIterator.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Frontend/OmpSs/OSSConstants.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <utility>

namespace clang {

class ASTContext;

//===----------------------------------------------------------------------===//
// AST classes for clauses.
//===----------------------------------------------------------------------===//

/// This is a basic class for representing single OmpSs clause.
class OSSClause {
  /// Starting location of the clause (the clause keyword).
  SourceLocation StartLoc;

  /// Ending location of the clause.
  SourceLocation EndLoc;

  /// Kind of the clause.
  OmpSsClauseKind Kind;

protected:
  OSSClause(OmpSsClauseKind K, SourceLocation StartLoc, SourceLocation EndLoc)
      : StartLoc(StartLoc), EndLoc(EndLoc), Kind(K) {}

public:
  /// Returns the starting location of the clause.
  SourceLocation getBeginLoc() const { return StartLoc; }

  /// Returns the ending location of the clause.
  SourceLocation getEndLoc() const { return EndLoc; }

  /// Sets the starting location of the clause.
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }

  /// Sets the ending location of the clause.
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  /// Returns kind of OmpSs clause (private, shared, reduction, etc.).
  OmpSsClauseKind getClauseKind() const { return Kind; }

  bool isImplicit() const { return StartLoc.isInvalid(); }

  using child_iterator = StmtIterator;
  using const_child_iterator = ConstStmtIterator;
  using child_range = llvm::iterator_range<child_iterator>;
  using const_child_range = llvm::iterator_range<const_child_iterator>;

  child_range children();
  const_child_range children() const {
    auto Children = const_cast<OSSClause *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }

  static bool classof(const OSSClause *) { return true; }
};

/// This represents clauses with the list of variables like 'private',
/// 'firstprivate', 'shared', or 'depend' clauses in the
/// '#pragma oss ...' directives.
template <class T> class OSSVarListClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Number of variables in the list.
  unsigned NumVars;

protected:
  /// Build a clause with \a N variables
  ///
  /// \param K Kind of the clause.
  /// \param StartLoc Starting location of the clause (the clause keyword).
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSVarListClause(OmpSsClauseKind K, SourceLocation StartLoc,
                   SourceLocation LParenLoc, SourceLocation EndLoc, unsigned N)
      : OSSClause(K, StartLoc, EndLoc), LParenLoc(LParenLoc), NumVars(N) {}

  /// Fetches list of variables associated with this clause.
  MutableArrayRef<Expr *> getVarRefs() {
    return MutableArrayRef<Expr *>(
        static_cast<T *>(this)->template getTrailingObjects<Expr *>(), NumVars);
  }

  /// Sets the list of variables for this clause.
  void setVarRefs(ArrayRef<Expr *> VL) {
    assert(VL.size() == NumVars &&
           "Number of variables is not the same as the preallocated buffer");
    std::copy(VL.begin(), VL.end(),
              static_cast<T *>(this)->template getTrailingObjects<Expr *>());
  }

public:
  using varlist_iterator = MutableArrayRef<Expr *>::iterator;
  using varlist_const_iterator = ArrayRef<const Expr *>::iterator;
  using varlist_range = llvm::iterator_range<varlist_iterator>;
  using varlist_const_range = llvm::iterator_range<varlist_const_iterator>;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }

  varlist_range varlists() {
    return varlist_range(varlist_begin(), varlist_end());
  }
  varlist_const_range varlists() const {
    return varlist_const_range(varlist_begin(), varlist_end());
  }

  varlist_iterator varlist_begin() { return getVarRefs().begin(); }
  varlist_iterator varlist_end() { return getVarRefs().end(); }
  varlist_const_iterator varlist_begin() const { return getVarRefs().begin(); }
  varlist_const_iterator varlist_end() const { return getVarRefs().end(); }

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Fetches list of all variables in the clause.
  ArrayRef<const Expr *> getVarRefs() const {
    return llvm::ArrayRef(
        static_cast<const T *>(this)->template getTrailingObjects<Expr *>(),
        NumVars);
  }
};

/// Contains data for OmpSs-2 directives: clauses, children
/// expressions/statements (helpers for codegen) and associated statement, if
/// any.
class OSSChildren final
    : private llvm::TrailingObjects<OSSChildren, OSSClause *, Stmt *> {
  friend TrailingObjects;
  friend class OSSClauseReader;
  friend class OSSExecutableDirective;

  /// Numbers of clauses.
  unsigned NumClauses = 0;
  /// Number of child expressions/stmts.
  unsigned NumChildren = 0;
  /// true if the directive has associated statement.
  bool HasAssociatedStmt = false;

  /// Define the sizes of each trailing object array except the last one. This
  /// is required for TrailingObjects to work properly.
  size_t numTrailingObjects(OverloadToken<OSSClause *>) const {
    return NumClauses;
  }

  OSSChildren() = delete;

  OSSChildren(unsigned NumClauses, unsigned NumChildren, bool HasAssociatedStmt)
      : NumClauses(NumClauses), NumChildren(NumChildren),
        HasAssociatedStmt(HasAssociatedStmt) {}

  static size_t size(unsigned NumClauses, bool HasAssociatedStmt,
                     unsigned NumChildren);

  static OSSChildren *Create(void *Mem, ArrayRef<OSSClause *> Clauses);
  static OSSChildren *Create(void *Mem, ArrayRef<OSSClause *> Clauses, Stmt *S,
                             unsigned NumChildren = 0);
  static OSSChildren *CreateEmpty(void *Mem, unsigned NumClauses,
                                  bool HasAssociatedStmt = false,
                                  unsigned NumChildren = 0);

public:
  unsigned getNumClauses() const { return NumClauses; }
  unsigned getNumChildren() const { return NumChildren; }
  bool hasAssociatedStmt() const { return HasAssociatedStmt; }

  /// Set associated statement.
  void setAssociatedStmt(Stmt *S) {
    getTrailingObjects<Stmt *>()[NumChildren] = S;
  }

  void setChildren(ArrayRef<Stmt *> Children);

  /// Sets the list of variables for this clause.
  ///
  /// \param Clauses The list of clauses for the directive.
  ///
  void setClauses(ArrayRef<OSSClause *> Clauses);

  /// Returns statement associated with the directive.
  const Stmt *getAssociatedStmt() const {
    return const_cast<OSSChildren *>(this)->getAssociatedStmt();
  }
  Stmt *getAssociatedStmt() {
    assert(HasAssociatedStmt &&
           "Expected directive with the associated statement.");
    return getTrailingObjects<Stmt *>()[NumChildren];
  }

  /// Get the clauses storage.
  MutableArrayRef<OSSClause *> getClauses() {
    return llvm::MutableArrayRef(getTrailingObjects<OSSClause *>(),
                                     NumClauses);
  }
  ArrayRef<OSSClause *> getClauses() const {
    return const_cast<OSSChildren *>(this)->getClauses();
  }

  MutableArrayRef<Stmt *> getChildren();
  ArrayRef<Stmt *> getChildren() const {
    return const_cast<OSSChildren *>(this)->getChildren();
  }

  Stmt::child_range getAssociatedStmtAsRange() {
    if (!HasAssociatedStmt)
      return Stmt::child_range(Stmt::child_iterator(), Stmt::child_iterator());
    return Stmt::child_range(&getTrailingObjects<Stmt *>()[NumChildren],
                             &getTrailingObjects<Stmt *>()[NumChildren + 1]);
  }
};

/// This represents 'immediate' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task immediate(a > 5)
/// \endcode
/// In this example directive '#pragma oss task' has simple 'immediate'
/// clause with condition 'a > 5'.
class OSSImmediateClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Condition of the 'immediate' clause.
  Stmt *Condition = nullptr;

  /// Set condition.
  void setCondition(Expr *Cond) { Condition = Cond; }

public:
  /// Build 'immediate' clause with condition \a Cond.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Cond Condition of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSImmediateClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
                 SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_immediate, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// Build an empty clause.
  OSSImmediateClause()
      : OSSClause(llvm::oss::OSSC_immediate, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  child_range children() { return child_range(&Condition, &Condition + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_immediate;
  }
};

/// This represents 'microtask' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task microtask(a > 5)
/// \endcode
/// In this example directive '#pragma oss task' has simple 'microtask'
/// clause with condition 'a > 5'.
class OSSMicrotaskClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Condition of the 'microtask' clause.
  Stmt *Condition = nullptr;

  /// Set condition.
  void setCondition(Expr *Cond) { Condition = Cond; }

public:
  /// Build 'microtask' clause with condition \a Cond.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Cond Condition of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSMicrotaskClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
                 SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_microtask, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// Build an empty clause.
  OSSMicrotaskClause()
      : OSSClause(llvm::oss::OSSC_microtask, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  child_range children() { return child_range(&Condition, &Condition + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_microtask;
  }
};

/// This represents 'if' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task if(a > 5)
/// \endcode
/// In this example directive '#pragma oss task' has simple 'if'
/// clause with condition 'a > 5'.
class OSSIfClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Condition of the 'if' clause.
  Stmt *Condition = nullptr;

  /// Set condition.
  void setCondition(Expr *Cond) { Condition = Cond; }

public:
  /// Build 'if' clause with condition \a Cond.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Cond Condition of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSIfClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
                 SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_if, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// Build an empty clause.
  OSSIfClause()
      : OSSClause(llvm::oss::OSSC_if, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  child_range children() { return child_range(&Condition, &Condition + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_if;
  }
};

/// This represents 'final' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task final(a > 5)
/// \endcode
/// In this example directive '#pragma oss task' has simple 'final'
/// clause with condition 'a > 5'.
class OSSFinalClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Condition of the 'final' clause.
  Stmt *Condition = nullptr;

  /// Set condition.
  void setCondition(Expr *Cond) { Condition = Cond; }

public:
  /// Build 'final' clause with condition \a Cond.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param Cond Condition of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSFinalClause(Expr *Cond, SourceLocation StartLoc, SourceLocation LParenLoc,
                 SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_final, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// Build an empty clause.
  OSSFinalClause()
      : OSSClause(llvm::oss::OSSC_final, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  child_range children() { return child_range(&Condition, &Condition + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_final;
  }
};

/// This represents 'cost' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task cost(foo(N))
/// \endcode
/// In this example directive '#pragma oss task' has simple 'cost'
/// clause with expression 'foo(N)'.
class OSSCostClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'cost' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'cost' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSCostClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_cost, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSCostClause()
      : OSSClause(llvm::oss::OSSC_cost, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_cost;
  }
};

/// This represents 'priority' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task priority(foo(N))
/// \endcode
/// In this example directive '#pragma oss task' has simple 'priority'
/// clause with expression 'foo(N)'.
class OSSPriorityClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'priority' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'priority' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSPriorityClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_priority, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSPriorityClause()
      : OSSClause(llvm::oss::OSSC_priority, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_priority;
  }
};

/// This represents 'label' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task label("string-literal") // T1
/// #pragma oss task label(s)                // T2
/// \endcode
/// In this example directive '#pragma oss task' T1 has a 'label' 'string-literal' and
/// a T2 a 'label' the string contained in variable 's'
class OSSLabelClause final
    : public OSSVarListClause<OSSLabelClause>,
      private llvm::TrailingObjects<OSSLabelClause, Expr *> {
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSLabelClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OSSVarListClause<OSSLabelClause>(llvm::oss::OSSC_label, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSLabelClause(unsigned N)
      : OSSVarListClause<OSSLabelClause>(llvm::oss::OSSC_label, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  static OSSLabelClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSLabelClause *CreateEmpty(const ASTContext &C, unsigned N);

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_label;
  }
};

/// This represents 'wait' clause in the '#pragma oss task' directive.
///
/// \code
/// #pragma oss task wait
/// \endcode
class OSSWaitClause : public OSSClause {
public:
  /// Build 'wait' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSWaitClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_wait, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSWaitClause()
      : OSSClause(llvm::oss::OSSC_wait, SourceLocation(), SourceLocation()) {}

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_wait;
  }
};

/// This represents 'update' clause in the '#pragma oss taskiter|atomic' directive.
///
/// \code
/// #pragma oss taskiter update
/// \endcode
class OSSUpdateClause : public OSSClause {
public:
  /// Build 'update' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSUpdateClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_update, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSUpdateClause()
      : OSSClause(llvm::oss::OSSC_update, SourceLocation(), SourceLocation()) {}

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_update;
  }
};

/// This represents 'onready' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task onready(foo(N))
/// \endcode
/// In this example directive '#pragma oss task' has simple 'onready'
/// clause with expression 'foo(N)'.
class OSSOnreadyClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'onready' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'onready' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSOnreadyClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_onready, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSOnreadyClause()
      : OSSClause(llvm::oss::OSSC_onready, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_onready;
  }
};

/// This represents 'chunksize' clause in the
/// '#pragma oss {task for|taskloop|taskloop for}' directive.
///
/// \code
/// #pragma oss task for chunksize(foo(N))
/// \endcode
/// In this example directive '#pragma oss task for' has simple 'chunksize'
/// clause with expression 'foo(N)'.
class OSSChunksizeClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'chunksize' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'chunksize' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSChunksizeClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_chunksize, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSChunksizeClause()
      : OSSClause(llvm::oss::OSSC_chunksize, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_chunksize;
  }
};

/// This represents 'grainsize' clause in the
/// '#pragma oss {task for|taskloop|taskloop for}' directive.
///
/// \code
/// #pragma oss task for grainsize(foo(N))
/// \endcode
/// In this example directive '#pragma oss task for' has simple 'grainsize'
/// clause with expression 'foo(N)'.
class OSSGrainsizeClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'grainsize' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'grainsize' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSGrainsizeClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_grainsize, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSGrainsizeClause()
      : OSSClause(llvm::oss::OSSC_grainsize, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_grainsize;
  }
};

/// This represents 'unroll' clause in the
/// '#pragma oss taskiter' directive.
///
/// \code
/// #pragma oss taskiter unroll(foo(N))
/// \endcode
/// In this example directive '#pragma oss taskiter' has simple 'unroll'
/// clause with expression 'foo(N)'.
class OSSUnrollClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'unroll' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'unroll' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSUnrollClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_unroll, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSUnrollClause()
      : OSSClause(llvm::oss::OSSC_unroll, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_unroll;
  }
};

/// This represents 'collapse' clause in the
/// '#pragma oss {task for|taskloop|taskloop for}' directive.
/// directive.
///
/// \code
/// #pragma oss taskloop collapse(3)
/// \endcode
/// In this example directive '#pragma oss taskloop' has clause 'collapse'
/// with single expression '3'.
/// The parameter must be a constant positive integer expression, it specifies
/// the number of nested loops that should be collapsed into a single iteration
/// space.
class OSSCollapseClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Number of for-loops.
  Stmt *NumForLoops = nullptr;

  /// Set the number of associated for-loops.
  void setNumForLoops(Expr *Num) { NumForLoops = Num; }

public:
  /// Build 'collapse' clause.
  ///
  /// \param Num Expression associated with this clause.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  OSSCollapseClause(Expr *Num, SourceLocation StartLoc,
                    SourceLocation LParenLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_collapse, StartLoc, EndLoc),
        LParenLoc(LParenLoc), NumForLoops(Num) {}

  /// Build an empty clause.
  explicit OSSCollapseClause()
      : OSSClause(llvm::oss::OSSC_collapse, SourceLocation(),
                  SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Return the number of associated for-loops.
  Expr *getNumForLoops() const { return cast_or_null<Expr>(NumForLoops); }

  child_range children() { return child_range(&NumForLoops, &NumForLoops + 1); }

  const_child_range children() const {
    return const_child_range(&NumForLoops, &NumForLoops + 1);
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_collapse;
  }
};

/// This represents 'default' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task default(shared)
/// \endcode
/// In this example directive '#pragma oss task' has simple 'default'
/// clause with kind 'shared'.
class OSSDefaultClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// A kind of the 'default' clause.
  llvm::oss::DefaultKind Kind = llvm::oss::OSS_DEFAULT_unknown;

  /// Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// Set kind of the clauses.
  ///
  /// \param K Argument of clause.
  void setDefaultKind(llvm::oss::DefaultKind K) { Kind = K; }

  /// Set argument location.
  ///
  /// \param KLoc Argument location.
  void setDefaultKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
  /// Build 'default' clause with argument \a A ('none' or 'shared').
  ///
  /// \param A Argument of the clause ('none' or 'shared').
  /// \param ALoc Starting location of the argument.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  OSSDefaultClause(llvm::oss::DefaultKind A, SourceLocation ALoc,
                   SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_default, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// Build an empty clause.
  OSSDefaultClause()
      : OSSClause(llvm::oss::OSSC_default, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns kind of the clause.
  llvm::oss::DefaultKind getDefaultKind() const { return Kind; }

  /// Returns location of clause kind.
  SourceLocation getDefaultKindKwLoc() const { return KindKwLoc; }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_default;
  }
};

/// This represents clause 'private' in the '#pragma oss ...' directives.
///
/// \code
/// #pragma oss task private(a,b)
/// \endcode
/// In this example directive '#pragma oss task' has clause 'private'
/// with the variables 'a' and 'b'.
class OSSPrivateClause final
    : public OSSVarListClause<OSSPrivateClause>,
      private llvm::TrailingObjects<OSSPrivateClause, Expr *> {
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSPrivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OSSVarListClause<OSSPrivateClause>(llvm::oss::OSSC_private, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSPrivateClause(unsigned N)
      : OSSVarListClause<OSSPrivateClause>(llvm::oss::OSSC_private, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

  /// Sets the list of references to private copies with initializers for
  /// new private variables.
  /// \param VL List of references.
  void setPrivateCopies(ArrayRef<Expr *> VL);

  /// Gets the list of references to private copies with initializers for
  /// new private variables.
  MutableArrayRef<Expr *> getPrivateCopies() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivateCopies() const {
    return llvm::ArrayRef(varlist_end(), varlist_size());
  }

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param PrivateVL List of references to private copies with initializers.
  static OSSPrivateClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc,
                                 ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSPrivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  using private_copies_iterator = MutableArrayRef<Expr *>::iterator;
  using private_copies_const_iterator = ArrayRef<const Expr *>::iterator;
  using private_copies_range = llvm::iterator_range<private_copies_iterator>;
  using private_copies_const_range =
      llvm::iterator_range<private_copies_const_iterator>;

  private_copies_range private_copies() {
    return private_copies_range(getPrivateCopies().begin(),
                                getPrivateCopies().end());
  }
  private_copies_const_range private_copies() const {
    return private_copies_const_range(getPrivateCopies().begin(),
                                      getPrivateCopies().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_private;
  }
};

/// This represents clause 'firstprivate' in the '#pragma oss ...'
/// directives.
///
/// \code
/// #pragma oss task firstprivate(a,b)
/// \endcode
/// In this example directive '#pragma oss task' has clause 'firstprivate'
/// with the variables 'a' and 'b'.
class OSSFirstprivateClause final
    : public OSSVarListClause<OSSFirstprivateClause>,
      private llvm::TrailingObjects<OSSFirstprivateClause, Expr *> {
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSFirstprivateClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OSSVarListClause<OSSFirstprivateClause>(llvm::oss::OSSC_firstprivate, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSFirstprivateClause(unsigned N)
      : OSSVarListClause<OSSFirstprivateClause>(llvm::oss::OSSC_firstprivate, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

  /// Sets the list of references to private copies with initializers for
  /// new private variables.
  /// \param VL List of references.
  void setPrivateCopies(ArrayRef<Expr *> VL);

  /// Gets the list of references to private copies with initializers for
  /// new private variables.
  MutableArrayRef<Expr *> getPrivateCopies() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getPrivateCopies() const {
    return llvm::ArrayRef(varlist_end(), varlist_size());
  }

  /// Sets the list of references to initializer variables for new
  /// private variables.
  /// \param VL List of references.
  void setInits(ArrayRef<Expr *> VL);

  /// Gets the list of references to initializer variables for new
  /// private variables.
  MutableArrayRef<Expr *> getInits() {
    return MutableArrayRef<Expr *>(getPrivateCopies().end(), varlist_size());
  }
  ArrayRef<const Expr *> getInits() const {
    return llvm::ArrayRef(getPrivateCopies().end(), varlist_size());
  }

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  /// \param PrivateVL List of references to private copies with initializers.
  /// \param InitVL List of references to auto generated variables used for
  /// initialization.
  static OSSFirstprivateClause *
  Create(const ASTContext &C, SourceLocation StartLoc,
         SourceLocation LParenLoc, SourceLocation EndLoc,
         ArrayRef<Expr *> VL, ArrayRef<Expr *> PrivateVL, ArrayRef<Expr *> InitVL);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSFirstprivateClause *CreateEmpty(const ASTContext &C, unsigned N);

  using private_copies_iterator = MutableArrayRef<Expr *>::iterator;
  using private_copies_const_iterator = ArrayRef<const Expr *>::iterator;
  using private_copies_range = llvm::iterator_range<private_copies_iterator>;
  using private_copies_const_range =
      llvm::iterator_range<private_copies_const_iterator>;

  private_copies_range private_copies() {
    return private_copies_range(getPrivateCopies().begin(),
                                getPrivateCopies().end());
  }
  private_copies_const_range private_copies() const {
    return private_copies_const_range(getPrivateCopies().begin(),
                                      getPrivateCopies().end());
  }

  using inits_iterator = MutableArrayRef<Expr *>::iterator;
  using inits_const_iterator = ArrayRef<const Expr *>::iterator;
  using inits_range = llvm::iterator_range<inits_iterator>;
  using inits_const_range = llvm::iterator_range<inits_const_iterator>;

  inits_range inits() {
    return inits_range(getInits().begin(), getInits().end());
  }
  inits_const_range inits() const {
    return inits_const_range(getInits().begin(), getInits().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_firstprivate;
  }
};

/// This represents clause 'shared' in the '#pragma oss ...' directives.
///
/// \code
/// #pragma oss task shared(a,b)
/// \endcode
/// In this example directive '#pragma oss task' has clause 'shared'
/// with the variables 'a' and 'b'.
class OSSSharedClause final
    : public OSSVarListClause<OSSSharedClause>,
      private llvm::TrailingObjects<OSSSharedClause, Expr *> {
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSSharedClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OSSVarListClause<OSSSharedClause>(llvm::oss::OSSC_shared, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSSharedClause(unsigned N)
      : OSSVarListClause<OSSSharedClause>(llvm::oss::OSSC_shared, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  static OSSSharedClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSSharedClause *CreateEmpty(const ASTContext &C, unsigned N);

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_shared;
  }
};

/// This represents implicit clause 'depend' for the '#pragma oss task'
/// directive.
///
/// \code
/// #pragma oss task depend(in:a,b)
/// \endcode
/// In this example directive '#pragma oss task' with clause 'depend' with the
/// variables 'a' and 'b' with dependency 'in'.
class OSSDependClause final
    : public OSSVarListClause<OSSDependClause>,
      private llvm::TrailingObjects<OSSDependClause, Expr *> {
  friend class OSSClauseReader;
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Dependency types in parsing order
  SmallVector<OmpSsDependClauseKind, 2> DepKinds;
  /// Dependency types ordered.
  /// { OSSC_DEPEND_in }
  /// { OSSC_DEPEND_in, OSSC_DEPEND_weak }
  SmallVector<OmpSsDependClauseKind, 2> DepKindsOrdered;

  /// Dependency type location.
  SourceLocation DepLoc;

  /// Colon location.
  SourceLocation ColonLoc;

  /// OpenMP or OmpSs-2 syntax.
  bool OSSSyntax;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSDependClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N, bool OSSSyntax)
      : OSSVarListClause<OSSDependClause>(llvm::oss::OSSC_depend, StartLoc, LParenLoc,
                                          EndLoc, N), OSSSyntax(OSSSyntax)
                                          {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSDependClause(unsigned N)
      : OSSVarListClause<OSSDependClause>(llvm::oss::OSSC_depend, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N), DepKinds(1, OSSC_DEPEND_unknown),
                                          DepKindsOrdered(1, OSSC_DEPEND_unknown),
                                          OSSSyntax(false)
                                          {}

  /// Set dependency kinds.
  void setDependencyKinds(ArrayRef<OmpSsDependClauseKind> K) {
    for (const OmpSsDependClauseKind& CK : K) {
      DepKinds.push_back(CK);
    }
  }

  /// Set dependency kinds.
  void setDependencyKindsOrdered(ArrayRef<OmpSsDependClauseKind> K) {
    for (const OmpSsDependClauseKind& CK : K) {
      DepKindsOrdered.push_back(CK);
    }
  }

  /// Set dependency kind and its location.
  void setDependencyLoc(SourceLocation Loc) { DepLoc = Loc; }

  /// Set colon location.
  void setColonLoc(SourceLocation Loc) { ColonLoc = Loc; }

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param DepKind Dependency type.
  /// \param DepKind Dependency type ordered.
  /// \param DepLoc Location of the dependency type.
  /// \param ColonLoc Colon location.
  /// \param VL List of references to the variables.
  /// \param OSSSyntax True if it's an OmpSs-2 depend clause (i.e. weakin())
  /// clause.
  static OSSDependClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc,
                                 ArrayRef<OmpSsDependClauseKind> DepKinds,
                                 ArrayRef<OmpSsDependClauseKind> DepKindsOrdered,
                                 SourceLocation DepLoc, SourceLocation ColonLoc,
                                 ArrayRef<Expr *> VL,
                                 bool OSSSyntax);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSDependClause *CreateEmpty(const ASTContext &C, unsigned N);

  /// Get dependency types.
  ArrayRef<OmpSsDependClauseKind> getDependencyKinds() const { return DepKinds; }

  /// Get dependency types.
  ArrayRef<OmpSsDependClauseKind> getDependencyKindsOrdered() const { return DepKindsOrdered; }

  /// Get dependency type location.
  SourceLocation getDependencyLoc() const { return DepLoc; }

  /// Get colon location.
  SourceLocation getColonLoc() const { return ColonLoc; }

  bool isOSSSyntax() const { return OSSSyntax; }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_depend;
  }
};


/// This represents clause 'reduction' in the '#pragma oss task'
/// directive.
///
/// \code
/// #pragma oss task reduction(+:a,b)
/// \endcode
/// In this example directive '#pragma oss task' has clause 'reduction'
/// with operator '+' and the variables 'a' and 'b'.
class OSSReductionClause final
    : public OSSVarListClause<OSSReductionClause>,
      private llvm::TrailingObjects<OSSReductionClause, Expr *> {
  friend class OSSClauseReader;
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Location of ':'.
  SourceLocation ColonLoc;

  /// Nested name specifier for C++.
  NestedNameSpecifierLoc QualifierLoc;

  /// Name of custom operator.
  DeclarationNameInfo NameInfo;

  /// Reduction Kind for each variable
  // TODO: this can be simplified since in a reduction clause
  // all variables have the same kind
  SmallVector<BinaryOperatorKind, 4> ReductionKinds;

  /// Tells if the reduction is weak 'weakreduction'
  bool IsWeak;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param ColonLoc Location of ':'.
  /// \param N Number of the variables in the clause.
  /// \param QualifierLoc The nested-name qualifier with location information
  /// \param NameInfo The full name info for reduction identifier.
  /// \param IsWeak Specifies if the reduction is weak.
  OSSReductionClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                     SourceLocation ColonLoc, SourceLocation EndLoc, unsigned N,
                     NestedNameSpecifierLoc QualifierLoc,
                     const DeclarationNameInfo &NameInfo,
                     bool IsWeak)
      : OSSVarListClause<OSSReductionClause>(llvm::oss::OSSC_reduction, StartLoc,
                                             LParenLoc, EndLoc, N),
        ColonLoc(ColonLoc),
        QualifierLoc(QualifierLoc), NameInfo(NameInfo),
        IsWeak(IsWeak) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSReductionClause(unsigned N)
      : OSSVarListClause<OSSReductionClause>(llvm::oss::OSSC_reduction, SourceLocation(),
                                             SourceLocation(), SourceLocation(),
                                             N)
        {}

  /// Sets location of ':' symbol in clause.
  void setColonLoc(SourceLocation CL) { ColonLoc = CL; }

  /// Sets the name info for specified reduction identifier.
  void setNameInfo(DeclarationNameInfo DNI) { NameInfo = DNI; }

  /// Sets the nested name specifier.
  void setQualifierLoc(NestedNameSpecifierLoc NSL) { QualifierLoc = NSL; }

  /// Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent data-sharing of the reduction
  /// variable.
  void setSimpleExprs(ArrayRef<Expr *> SimpleExprs);

  /// Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent LHS expression in the final
  /// reduction expression performed by the reduction clause.
  void setLHSExprs(ArrayRef<Expr *> LHSExprs);

  /// Get the list of helper LHS expressions.
  MutableArrayRef<Expr *> getLHSExprs() {
    return MutableArrayRef<Expr *>(varlist_end(), varlist_size());
  }
  ArrayRef<const Expr *> getLHSExprs() const {
    return llvm::ArrayRef(varlist_end(), varlist_size());
  }

  /// Set list of helper expressions, required for proper codegen of the
  /// clause. These expressions represent RHS expression in the final
  /// reduction expression performed by the reduction clause.
  /// Also, variables in these expressions are used for proper initialization of
  /// reduction copies.
  void setRHSExprs(ArrayRef<Expr *> RHSExprs);

  /// Get the list of helper destination expressions.
  MutableArrayRef<Expr *> getRHSExprs() {
    return MutableArrayRef<Expr *>(getLHSExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getRHSExprs() const {
    return llvm::ArrayRef(getLHSExprs().end(), varlist_size());
  }

  /// Set list of helper reduction expressions, required for proper
  /// codegen of the clause. These expressions are binary expressions or
  /// operator/custom reduction call that calculates new value from source
  /// helper expressions to destination helper expressions.
  void setReductionOps(ArrayRef<Expr *> ReductionOps);

  /// Get the list of helper reduction expressions.
  MutableArrayRef<Expr *> getReductionOps() {
    return MutableArrayRef<Expr *>(getRHSExprs().end(), varlist_size());
  }
  ArrayRef<const Expr *> getReductionOps() const {
    return llvm::ArrayRef(getRHSExprs().end(), varlist_size());
  }

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param ColonLoc Location of ':'.
  /// \param EndLoc Ending location of the clause.
  /// \param VL The variables in the clause.
  /// \param QualifierLoc The nested-name qualifier with location information
  /// \param NameInfo The full name info for reduction identifier.
  /// \param LHSExprs This list represents LHSs of the reduction expressions.
  /// \param RHSExprs This list represents RHSs of the reduction expressions.
  /// Also, variables in these expressions are used for proper initialization of
  /// reduction copies.
  /// \param ReductionOps List of helper expressions that represents reduction
  /// expressions:
  /// \code
  /// LHSExprs binop RHSExprs;
  /// operator binop(LHSExpr, RHSExpr);
  /// <CustomCombiner>(<CombinerOut>, <CombinerIn>);
  /// <CustomInit>(<InitPriv>, <InitOrig>);
  /// \endcode
  /// Required for proper codegen of final reduction operation performed by the
  /// reduction clause.
  static OSSReductionClause *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation LParenLoc,
         SourceLocation ColonLoc, SourceLocation EndLoc, ArrayRef<Expr *> VL,
         NestedNameSpecifierLoc QualifierLoc, const DeclarationNameInfo &NameInfo,
         ArrayRef<Expr *> LHSExprs, ArrayRef<Expr *> RHSExprs,
         ArrayRef<Expr *> ReductionOps, ArrayRef<BinaryOperatorKind> ReductionKinds,
         bool IsWeak);

  /// Creates an empty clause with the place for \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSReductionClause *CreateEmpty(const ASTContext &C, unsigned N);

  /// Gets location of ':' symbol in clause.
  SourceLocation getColonLoc() const { return ColonLoc; }

  /// Gets the name info for specified reduction identifier.
  const DeclarationNameInfo &getNameInfo() const { return NameInfo; }

  /// Gets the nested name specifier.
  NestedNameSpecifierLoc getQualifierLoc() const { return QualifierLoc; }

  ArrayRef<BinaryOperatorKind> getReductionKinds() const { return ReductionKinds; }

  bool isWeak() const { return IsWeak; }

  using helper_expr_iterator = MutableArrayRef<Expr *>::iterator;
  using helper_expr_const_iterator = ArrayRef<const Expr *>::iterator;
  using helper_expr_range = llvm::iterator_range<helper_expr_iterator>;
  using helper_expr_const_range =
      llvm::iterator_range<helper_expr_const_iterator>;

  helper_expr_const_range lhs_exprs() const {
    return helper_expr_const_range(getLHSExprs().begin(), getLHSExprs().end());
  }

  helper_expr_range lhs_exprs() {
    return helper_expr_range(getLHSExprs().begin(), getLHSExprs().end());
  }

  helper_expr_const_range rhs_exprs() const {
    return helper_expr_const_range(getRHSExprs().begin(), getRHSExprs().end());
  }

  helper_expr_range rhs_exprs() {
    return helper_expr_range(getRHSExprs().begin(), getRHSExprs().end());
  }

  helper_expr_const_range reduction_ops() const {
    return helper_expr_const_range(getReductionOps().begin(),
                                   getReductionOps().end());
  }

  helper_expr_range reduction_ops() {
    return helper_expr_range(getReductionOps().begin(),
                             getReductionOps().end());
  }

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  const_child_range children() const {
    auto Children = const_cast<OSSReductionClause *>(this)->children();
    return const_child_range(Children.begin(), Children.end());
  }

  child_range used_children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }
  const_child_range used_children() const {
    auto Children = const_cast<OSSReductionClause *>(this)->used_children();
    return const_child_range(Children.begin(), Children.end());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_reduction;
  }
};

/// This represents 'device' clause in the '#pragma oss ...' directive.
///
/// \code
/// #pragma oss task device(cuda)
/// \endcode
/// In this example directive '#pragma oss task' has simple 'device'
/// clause with kind 'cuda'.
class OSSDeviceClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// A kind of the 'device' clause.
  OmpSsDeviceClauseKind Kind = OSSC_DEVICE_unknown;

  /// Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// Set kind of the clauses.
  ///
  /// \param K Argument of clause.
  void setDeviceKind(OmpSsDeviceClauseKind K) { Kind = K; }

  /// Set argument location.
  ///
  /// \param KLoc Argument location.
  void setDeviceKindKwLoc(SourceLocation KLoc) { KindKwLoc = KLoc; }

public:
  /// Build 'device' clause with argument \a A ('smp' or 'cuda').
  ///
  /// \param A Argument of the clause ('none' or 'shared').
  /// \param ALoc Starting location of the argument.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  OSSDeviceClause(OmpSsDeviceClauseKind A, SourceLocation ALoc,
                   SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_device, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// Build an empty clause.
  OSSDeviceClause()
      : OSSClause(llvm::oss::OSSC_device, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns kind of the clause.
  OmpSsDeviceClauseKind getDeviceKind() const { return Kind; }

  /// Returns location of clause kind.
  SourceLocation getDeviceKindKwLoc() const { return KindKwLoc; }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_device;
  }
};

/// This represents 'ndrange' clause in the
/// '#pragma oss task' directive.
///
/// \code
/// #pragma oss task ndrange(1, N, 128)
/// \endcode
///
/// The syntax of ndrange is
/// \code
///   ndrange(N, global-list, local-list)
/// \endcode
/// Each X-list has as much as N elements
///
/// In this example directive '#pragma oss task' has simple 'ndrange'
/// clause with expression 'foo(1, N, 128)'.
class OSSNdrangeClause final
    : public OSSVarListClause<OSSNdrangeClause>,
      private llvm::TrailingObjects<OSSNdrangeClause, Expr *> {
  friend OSSVarListClause;
  friend TrailingObjects;

  /// Build clause with number of variables \a N.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param N Number of the variables in the clause.
  OSSNdrangeClause(SourceLocation StartLoc, SourceLocation LParenLoc,
                  SourceLocation EndLoc, unsigned N)
      : OSSVarListClause<OSSNdrangeClause>(llvm::oss::OSSC_ndrange, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSNdrangeClause(unsigned N)
      : OSSVarListClause<OSSNdrangeClause>(llvm::oss::OSSC_ndrange, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N) {}

public:
  /// Creates clause with a list of variables \a VL.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param EndLoc Ending location of the clause.
  /// \param VL List of references to the variables.
  static OSSNdrangeClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc, ArrayRef<Expr *> VL);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSNdrangeClause *CreateEmpty(const ASTContext &C, unsigned N);

  child_range children() {
    return child_range(reinterpret_cast<Stmt **>(varlist_begin()),
                       reinterpret_cast<Stmt **>(varlist_end()));
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_ndrange;
  }
};

/// This represents 'shmem' clause in the
/// '#pragma oss task directive.
///
/// \code
/// #pragma oss task device(cuda) ndrange(1, 1, 1) shmem(foo(N))
/// \endcode
/// In this example directive '#pragma oss task for' has simple 'shmem'
/// clause with expression 'foo(N)'.
class OSSShmemClause : public OSSClause {
  friend class OSSClauseReader;

  /// Location of '('.
  SourceLocation LParenLoc;

  /// Expression of the 'shmem' clause.
  Stmt *Expression = nullptr;

  /// Set expression.
  void setExpression(Expr *E) { Expression = E; }

public:
  /// Build 'shmem' clause with expression \a E.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param LParenLoc Location of '('.
  /// \param E Expression of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSShmemClause(Expr *E, SourceLocation StartLoc, SourceLocation LParenLoc,
                SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_shmem, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSShmemClause()
      : OSSClause(llvm::oss::OSSC_shmem, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_shmem;
  }
};

/// This represents 'read' clause in the '#pragma oss atomic' directive.
///
/// \code
/// #pragma oss atomic read
/// \endcode
/// In this example directive '#pragma oss atomic' has 'read' clause.
class OSSReadClause : public OSSClause {
public:
  /// Build 'read' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSReadClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_read, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSReadClause()
      : OSSClause(llvm::oss::OSSC_read, SourceLocation(), SourceLocation()) {}

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_read;
  }
};

/// This represents 'write' clause in the '#pragma oss atomic' directive.
///
/// \code
/// #pragma oss atomic write
/// \endcode
/// In this example directive '#pragma oss atomic' has 'write' clause.
class OSSWriteClause : public OSSClause {
public:
  /// Build 'write' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSWriteClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_write, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSWriteClause()
      : OSSClause(llvm::oss::OSSC_write, SourceLocation(), SourceLocation()) {}

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_write;
  }
};

/// This represents 'capture' clause in the '#pragma oss atomic'
/// directive.
///
/// \code
/// #pragma oss atomic capture
/// \endcode
/// In this example directive '#pragma oss atomic' has 'capture' clause.
class OSSCaptureClause : public OSSClause {
public:
  /// Build 'capture' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSCaptureClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_capture, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSCaptureClause()
      : OSSClause(llvm::oss::OSSC_capture, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_capture;
  }
};

/// This represents 'compare' clause in the '#pragma oss atomic'
/// directive.
///
/// \code
/// #pragma oss atomic compare
/// \endcode
/// In this example directive '#pragma oss atomic' has 'compare' clause.
class OSSCompareClause final : public OSSClause {
public:
  /// Build 'compare' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSCompareClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_compare, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSCompareClause()
      : OSSClause(llvm::oss::OSSC_compare, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_compare;
  }
};

/// This represents 'seq_cst' clause in the '#pragma oss atomic'
/// directive.
///
/// \code
/// #pragma oss atomic seq_cst
/// \endcode
/// In this example directive '#pragma oss atomic' has 'seq_cst' clause.
class OSSSeqCstClause : public OSSClause {
public:
  /// Build 'seq_cst' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSSeqCstClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_seq_cst, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSSeqCstClause()
      : OSSClause(llvm::oss::OSSC_seq_cst, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_seq_cst;
  }
};

/// This represents 'acq_rel' clause in the '#pragma oss atomic'
/// directives.
///
/// \code
/// #pragma oss atomic acq_rel
/// \endcode
/// In this example directive '#pragma oss atomic' has 'acq_rel' clause.
class OSSAcqRelClause final : public OSSClause {
public:
  /// Build 'ack_rel' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSAcqRelClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_acq_rel, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSAcqRelClause()
      : OSSClause(llvm::oss::OSSC_acq_rel, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_acq_rel;
  }
};

/// This represents 'acquire' clause in the '#pragma oss atomic'
/// directives.
///
/// \code
/// #pragma oss atomic acquire
/// \endcode
/// In this example directive '#pragma oss atomic' has 'acquire' clause.
class OSSAcquireClause final : public OSSClause {
public:
  /// Build 'acquire' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSAcquireClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_acquire, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSAcquireClause()
      : OSSClause(llvm::oss::OSSC_acquire, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_acquire;
  }
};

/// This represents 'release' clause in the '#pragma oss atomic'
/// directives.
///
/// \code
/// #pragma oss atomic release
/// \endcode
/// In this example directive '#pragma oss atomic' has 'release' clause.
class OSSReleaseClause final : public OSSClause {
public:
  /// Build 'release' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSReleaseClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_release, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSReleaseClause()
      : OSSClause(llvm::oss::OSSC_release, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_release;
  }
};

/// This represents 'relaxed' clause in the '#pragma oss atomic'
/// directives.
///
/// \code
/// #pragma oss atomic relaxed
/// \endcode
/// In this example directive '#pragma oss atomic' has 'relaxed' clause.
class OSSRelaxedClause final : public OSSClause {
public:
  /// Build 'relaxed' clause.
  ///
  /// \param StartLoc Starting location of the clause.
  /// \param EndLoc Ending location of the clause.
  OSSRelaxedClause(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSClause(llvm::oss::OSSC_relaxed, StartLoc, EndLoc) {}

  /// Build an empty clause.
  OSSRelaxedClause()
      : OSSClause(llvm::oss::OSSC_relaxed, SourceLocation(), SourceLocation()) {
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  child_range used_children() {
    return child_range(child_iterator(), child_iterator());
  }
  const_child_range used_children() const {
    return const_child_range(const_child_iterator(), const_child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == llvm::oss::OSSC_relaxed;
  }
};

/// This class implements a simple visitor for OSSClause
/// subclasses.
template<class ImplClass, template <typename> class Ptr, typename RetTy>
class OSSClauseVisitorBase {
public:
#define PTR(CLASS) Ptr<CLASS>
#define DISPATCH(CLASS) \
  return static_cast<ImplClass*>(this)->Visit##CLASS(static_cast<PTR(CLASS)>(S))

#define GEN_CLANG_CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class)                                         \
  RetTy Visit##Class(PTR(Class) S) { DISPATCH(Class); }
#include "llvm/Frontend/OmpSs/OSS.inc"

  RetTy Visit(PTR(OSSClause) S) {
    // Top switch clause: visit each OMPClause.
    switch (S->getClauseKind()) {
#define GEN_CLANG_CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class)                                         \
  case llvm::oss::Clause::Enum:                                                \
    return Visit##Class(static_cast<PTR(Class)>(S));
#define CLAUSE_NO_CLASS(Enum, Str)                                             \
  case llvm::oss::Clause::Enum:                                                \
    break;
#include "llvm/Frontend/OmpSs/OSS.inc"
    }
  }
  // Base case, ignore it. :)
  RetTy VisitOSSClause(PTR(OSSClause) Node) { return RetTy(); }
#undef PTR
#undef DISPATCH
};

template <typename T>
using const_ptr = std::add_pointer_t<std::add_const_t<T>>;

template<class ImplClass, typename RetTy = void>
class OSSClauseVisitor :
      public OSSClauseVisitorBase <ImplClass, std::add_pointer_t, RetTy> {};
template<class ImplClass, typename RetTy = void>
class ConstOSSClauseVisitor :
      public OSSClauseVisitorBase <ImplClass, const_ptr, RetTy> {};

class OSSClausePrinter final : public OSSClauseVisitor<OSSClausePrinter> {
  raw_ostream &OS;
  const PrintingPolicy &Policy;

  /// Process clauses with list of variables.
  template <typename T> void VisitOSSClauseList(T *Node, char StartSym);
  /// Process motion clauses.
  template <typename T> void VisitOSSMotionClause(T *Node);

public:
  OSSClausePrinter(raw_ostream &OS, const PrintingPolicy &Policy)
      : OS(OS), Policy(Policy) {}

#define GEN_CLANG_CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class) void Visit##Class(Class *S);
#include "llvm/Frontend/OmpSs/OSS.inc"
};


} // namespace clang

#endif // LLVM_CLANG_AST_OMPSSCLAUSE_H
