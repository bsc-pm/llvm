//===- DeclOmpSs.h - Classes for representing OmpSs directives -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines OmpSs nodes for declarative directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLOMPSS_H
#define LLVM_CLANG_AST_DECLOMPSS_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/OmpSsClause.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"

namespace clang {

/// This represents '#pragma oss declare reduction ...' directive.
/// For example, in the following, declared reduction 'foo' for types 'int' and
/// 'float':
///
/// \code
/// #pragma oss declare reduction (foo : int,float : omp_out += omp_in) \
///                     initializer (omp_priv = 0)
/// \endcode
///
/// Here 'omp_out += omp_in' is a combiner and 'omp_priv = 0' is an initializer.
class OSSDeclareReductionDecl final : public ValueDecl, public DeclContext {
  // This class stores some data in DeclContext::OSSDeclareReductionDeclBits
  // to save some space. Use the provided accessors to access it.
public:
  enum InitKind {
    CallInit,   // Initialized by function call.
    DirectInit, // omp_priv(<expr>)
    CopyInit    // omp_priv = <expr>
  };

private:
  friend class ASTDeclReader;
  /// Combiner for declare reduction construct.
  Expr *Combiner = nullptr;
  /// Initializer for declare reduction construct.
  Expr *Initializer = nullptr;
  /// In parameter of the combiner.
  Expr *In = nullptr;
  /// Out parameter of the combiner.
  Expr *Out = nullptr;
  /// Priv parameter of the initializer.
  Expr *Priv = nullptr;
  /// Orig parameter of the initializer.
  Expr *Orig = nullptr;

  /// Reference to the previous declare reduction construct in the same
  /// scope with the same name. Required for proper templates instantiation if
  /// the declare reduction construct is declared inside compound statement.
  LazyDeclPtr PrevDeclInScope;

  virtual void anchor();

  OSSDeclareReductionDecl(Kind DK, DeclContext *DC, SourceLocation L,
                          DeclarationName Name, QualType Ty,
                          OSSDeclareReductionDecl *PrevDeclInScope);

  void setPrevDeclInScope(OSSDeclareReductionDecl *Prev) {
    PrevDeclInScope = Prev;
  }

public:
  /// Create declare reduction node.
  static OSSDeclareReductionDecl *
  Create(ASTContext &C, DeclContext *DC, SourceLocation L, DeclarationName Name,
         QualType T, OSSDeclareReductionDecl *PrevDeclInScope);
  /// Create deserialized declare reduction node.
  static OSSDeclareReductionDecl *CreateDeserialized(ASTContext &C,
                                                     unsigned ID);

  /// Get combiner expression of the declare reduction construct.
  Expr *getCombiner() { return Combiner; }
  const Expr *getCombiner() const { return Combiner; }
  /// Get In variable of the combiner.
  Expr *getCombinerIn() { return In; }
  const Expr *getCombinerIn() const { return In; }
  /// Get Out variable of the combiner.
  Expr *getCombinerOut() { return Out; }
  const Expr *getCombinerOut() const { return Out; }
  /// Set combiner expression for the declare reduction construct.
  void setCombiner(Expr *E) { Combiner = E; }
  /// Set combiner In and Out vars.
  void setCombinerData(Expr *InE, Expr *OutE) {
    In = InE;
    Out = OutE;
  }

  /// Get initializer expression (if specified) of the declare reduction
  /// construct.
  Expr *getInitializer() { return Initializer; }
  const Expr *getInitializer() const { return Initializer; }
  /// Get initializer kind.
  InitKind getInitializerKind() const {
    return static_cast<InitKind>(OSSDeclareReductionDeclBits.InitializerKind);
  }
  /// Get Orig variable of the initializer.
  Expr *getInitOrig() { return Orig; }
  const Expr *getInitOrig() const { return Orig; }
  /// Get Priv variable of the initializer.
  Expr *getInitPriv() { return Priv; }
  const Expr *getInitPriv() const { return Priv; }
  /// Set initializer expression for the declare reduction construct.
  void setInitializer(Expr *E, InitKind IK) {
    Initializer = E;
    OSSDeclareReductionDeclBits.InitializerKind = IK;
  }
  /// Set initializer Orig and Priv vars.
  void setInitializerData(Expr *OrigE, Expr *PrivE) {
    Orig = OrigE;
    Priv = PrivE;
  }

  /// Get reference to previous declare reduction construct in the same
  /// scope with the same name.
  OSSDeclareReductionDecl *getPrevDeclInScope();
  const OSSDeclareReductionDecl *getPrevDeclInScope() const;

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OSSDeclareReduction; }
  static DeclContext *castToDeclContext(const OSSDeclareReductionDecl *D) {
    return static_cast<DeclContext *>(const_cast<OSSDeclareReductionDecl *>(D));
  }
  static OSSDeclareReductionDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<OSSDeclareReductionDecl *>(
        const_cast<DeclContext *>(DC));
  }
};

/// This represents '#pragma oss assert ...' directive.
///
/// \code
/// #pragma oss assert("stacksize=4M")
/// int main() {}
/// \endcode
///
class OSSAssertDecl final
    : public Decl, private llvm::TrailingObjects<OSSAssertDecl, Expr *> {
  friend class ASTDeclReader;
  friend TrailingObjects;

  unsigned NumVars;

  virtual void anchor();

  OSSAssertDecl(Kind DK, DeclContext *DC, SourceLocation L) :
    Decl(DK, DC, L), NumVars(0) { }

  ArrayRef<const Expr *> getVars() const {
    return llvm::makeArrayRef(getTrailingObjects<Expr *>(), NumVars);
  }

  MutableArrayRef<Expr *> getVars() {
    return MutableArrayRef<Expr *>(getTrailingObjects<Expr *>(), NumVars);
  }

  void setVars(ArrayRef<Expr *> VL);

public:
  static OSSAssertDecl *Create(
    ASTContext &C, DeclContext *DC, SourceLocation L, ArrayRef<Expr *> VL);

  static OSSAssertDecl *CreateDeserialized(
    ASTContext &C, unsigned ID, unsigned N);

  typedef MutableArrayRef<Expr *>::iterator varlist_iterator;
  typedef ArrayRef<const Expr *>::iterator varlist_const_iterator;
  typedef llvm::iterator_range<varlist_iterator> varlist_range;
  typedef llvm::iterator_range<varlist_const_iterator> varlist_const_range;

  unsigned varlist_size() const { return NumVars; }
  bool varlist_empty() const { return NumVars == 0; }

  varlist_range varlists() {
    return varlist_range(varlist_begin(), varlist_end());
  }
  varlist_const_range varlists() const {
    return varlist_const_range(varlist_begin(), varlist_end());
  }
  varlist_iterator varlist_begin() { return getVars().begin(); }
  varlist_iterator varlist_end() { return getVars().end(); }
  varlist_const_iterator varlist_begin() const { return getVars().begin(); }
  varlist_const_iterator varlist_end() const { return getVars().end(); }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OSSAssert; }
};

} // end namespace clang

#endif
