//===- OmpSsClause.h - Classes for OmpSs clauses --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
    return llvm::makeArrayRef(
        static_cast<const T *>(this)->template getTrailingObjects<Expr *>(),
        NumVars);
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
      : OSSClause(OSSC_if, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// Build an empty clause.
  OSSIfClause()
      : OSSClause(OSSC_if, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  child_range children() { return child_range(&Condition, &Condition + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == OSSC_if;
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
      : OSSClause(OSSC_final, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Condition(Cond) {}

  /// Build an empty clause.
  OSSFinalClause()
      : OSSClause(OSSC_final, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns condition.
  Expr *getCondition() const { return cast_or_null<Expr>(Condition); }

  child_range children() { return child_range(&Condition, &Condition + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == OSSC_final;
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
      : OSSClause(OSSC_cost, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSCostClause()
      : OSSClause(OSSC_cost, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == OSSC_cost;
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
      : OSSClause(OSSC_priority, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Expression(E) {}

  /// Build an empty clause.
  OSSPriorityClause()
      : OSSClause(OSSC_priority, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns expression.
  Expr *getExpression() const { return cast_or_null<Expr>(Expression); }

  child_range children() { return child_range(&Expression, &Expression + 1); }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == OSSC_priority;
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
  OmpSsDefaultClauseKind Kind = OSSC_DEFAULT_unknown;

  /// Start location of the kind in source code.
  SourceLocation KindKwLoc;

  /// Set kind of the clauses.
  ///
  /// \param K Argument of clause.
  void setDefaultKind(OmpSsDefaultClauseKind K) { Kind = K; }

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
  OSSDefaultClause(OmpSsDefaultClauseKind A, SourceLocation ALoc,
                   SourceLocation StartLoc, SourceLocation LParenLoc,
                   SourceLocation EndLoc)
      : OSSClause(OSSC_default, StartLoc, EndLoc), LParenLoc(LParenLoc),
        Kind(A), KindKwLoc(ALoc) {}

  /// Build an empty clause.
  OSSDefaultClause()
      : OSSClause(OSSC_default, SourceLocation(), SourceLocation()) {}

  /// Sets the location of '('.
  void setLParenLoc(SourceLocation Loc) { LParenLoc = Loc; }

  /// Returns the location of '('.
  SourceLocation getLParenLoc() const { return LParenLoc; }

  /// Returns kind of the clause.
  OmpSsDefaultClauseKind getDefaultKind() const { return Kind; }

  /// Returns location of clause kind.
  SourceLocation getDefaultKindKwLoc() const { return KindKwLoc; }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  static bool classof(const OSSClause *T) {
    return T->getClauseKind() == OSSC_default;
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
      : OSSVarListClause<OSSPrivateClause>(OSSC_private, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSPrivateClause(unsigned N)
      : OSSVarListClause<OSSPrivateClause>(OSSC_private, SourceLocation(),
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
    return llvm::makeArrayRef(varlist_end(), varlist_size());
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
    return T->getClauseKind() == OSSC_private;
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
      : OSSVarListClause<OSSFirstprivateClause>(OSSC_firstprivate, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSFirstprivateClause(unsigned N)
      : OSSVarListClause<OSSFirstprivateClause>(OSSC_firstprivate, SourceLocation(),
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
    return llvm::makeArrayRef(varlist_end(), varlist_size());
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
    return llvm::makeArrayRef(getPrivateCopies().end(), varlist_size());
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
    return T->getClauseKind() == OSSC_firstprivate;
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
      : OSSVarListClause<OSSSharedClause>(OSSC_shared, StartLoc, LParenLoc,
                                          EndLoc, N) {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSSharedClause(unsigned N)
      : OSSVarListClause<OSSSharedClause>(OSSC_shared, SourceLocation(),
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
    return T->getClauseKind() == OSSC_shared;
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

  /// Dependency type (one of in, out, inout).
  SmallVector<OmpSsDependClauseKind, 2> DepKinds;

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
      : OSSVarListClause<OSSDependClause>(OSSC_depend, StartLoc, LParenLoc,
                                          EndLoc, N), OSSSyntax(OSSSyntax)
                                          {}

  /// Build an empty clause.
  ///
  /// \param N Number of variables.
  explicit OSSDependClause(unsigned N)
      : OSSVarListClause<OSSDependClause>(OSSC_depend, SourceLocation(),
                                          SourceLocation(), SourceLocation(),
                                          N), DepKinds(1, OSSC_DEPEND_unknown),
                                          OSSSyntax(false)
                                          {}

  /// Set dependency kind.
  void setDependencyKinds(ArrayRef<OmpSsDependClauseKind> K) {
    for (const OmpSsDependClauseKind& CK : K) {
      DepKinds.push_back(CK);
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
  /// \param DepLoc Location of the dependency type.
  /// \param ColonLoc Colon location.
  /// \param VL List of references to the variables.
  /// \param OSSSyntax True if it's an OmpSs-2 depend clause (i.e. weakin())
  /// clause.
  static OSSDependClause *Create(const ASTContext &C, SourceLocation StartLoc,
                                 SourceLocation LParenLoc,
                                 SourceLocation EndLoc,
                                 ArrayRef<OmpSsDependClauseKind> DepKinds,
                                 SourceLocation DepLoc, SourceLocation ColonLoc,
                                 ArrayRef<Expr *> VL,
                                 bool OSSSyntax);

  /// Creates an empty clause with \a N variables.
  ///
  /// \param C AST context.
  /// \param N The number of variables.
  static OSSDependClause *CreateEmpty(const ASTContext &C, unsigned N);

  /// Get dependency types.
  ArrayRef<OmpSsDependClauseKind> getDependencyKind() const { return DepKinds; }

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
    return T->getClauseKind() == OSSC_depend;
  }
};

/// This class implements a simple visitor for OSSClause
/// subclasses.
template<class ImplClass, template <typename> class Ptr, typename RetTy>
class OSSClauseVisitorBase {
public:
#define PTR(CLASS) typename Ptr<CLASS>::type
#define DISPATCH(CLASS) \
  return static_cast<ImplClass*>(this)->Visit##CLASS(static_cast<PTR(CLASS)>(S))

#define OMPSS_CLAUSE(Name, Class)                              \
  RetTy Visit ## Class (PTR(Class) S) { DISPATCH(Class); }
#include "clang/Basic/OmpSsKinds.def"

  RetTy Visit(PTR(OSSClause) S) {
    // Top switch clause: visit each OSSClause.
    switch (S->getClauseKind()) {
    default: llvm_unreachable("Unknown clause kind!");
#define OMPSS_CLAUSE(Name, Class)                              \
    case OSSC_ ## Name : return Visit ## Class(static_cast<PTR(Class)>(S));
#include "clang/Basic/OmpSsKinds.def"
    }
  }
  // Base case, ignore it. :)
  RetTy VisitOSSClause(PTR(OSSClause) Node) { return RetTy(); }
#undef PTR
#undef DISPATCH
};

template <typename T>
using const_ptr = typename std::add_pointer<typename std::add_const<T>::type>;

template<class ImplClass, typename RetTy = void>
class OSSClauseVisitor :
      public OSSClauseVisitorBase <ImplClass, std::add_pointer, RetTy> {};
template<class ImplClass, typename RetTy = void>
class ConstOSSClauseVisitor :
      public OSSClauseVisitorBase <ImplClass, const_ptr, RetTy> {};
} // namespace clang

#endif // LLVM_CLANG_AST_OMPSSCLAUSE_H
