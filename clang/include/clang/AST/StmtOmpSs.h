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

#include "clang/AST/Expr.h"
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
protected:
  /// Build instance of directive of class \a K.
  ///
  /// \param SC Statement class.
  /// \param K Kind of OmpSs directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  ///
  template <typename T>
  OSSExecutableDirective(const T *, StmtClass SC, OmpSsDirectiveKind K,
                         SourceLocation StartLoc, SourceLocation EndLoc)
      : Stmt(SC), Kind(K), StartLoc(std::move(StartLoc)),
        EndLoc(std::move(EndLoc)) {}

public:

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

  OmpSsDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstOSSExecutableDirectiveConstant &&
           S->getStmtClass() <= lastOSSExecutableDirectiveConstant;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
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
  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  OSSTaskwaitDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : OSSExecutableDirective(this, OSSTaskwaitDirectiveClass, OSSD_taskwait,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit OSSTaskwaitDirective()
      : OSSExecutableDirective(this, OSSTaskwaitDirectiveClass, OSSD_taskwait,
                               SourceLocation(), SourceLocation()) {}

public:
  /// Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  ///
  static OSSTaskwaitDirective *
  Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static OSSTaskwaitDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSTaskwaitDirectiveClass;
  }
};

} // end namespace clang

#endif
