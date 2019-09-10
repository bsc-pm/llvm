//===--- ExprOmpSs.h - Classes for representing expressions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Expr interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_EXPROMPSS_H
#define LLVM_CLANG_AST_EXPROMPSS_H

#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"

namespace clang {
/// OmpSs 4.0 [2.4, Array Sections].
/// To specify an array section in an OmpSs construct, array subscript
/// expressions are extended with the following syntax:
/// \code
/// [ lower-bound : length ]
/// [ lower-bound : ]
/// [ : length ]
/// [ : ]
/// \endcode
/// The array section must be a subset of the original array.
/// Array sections are allowed on multidimensional arrays. Base language array
/// subscript expressions can be used to specify length-one dimensions of
/// multidimensional array sections.
/// The lower-bound and length are integral type expressions. When evaluated
/// they represent a set of integer values as follows:
/// \code
/// { lower-bound, lower-bound + 1, lower-bound + 2,... , lower-bound + length -
/// 1 }
/// \endcode
/// The lower-bound and length must evaluate to non-negative integers.
/// When the size of the array dimension is not known, the length must be
/// specified explicitly.
/// When the length is absent, it defaults to the size of the array dimension
/// minus the lower-bound.
/// When the lower-bound is absent it defaults to 0.
class OSSArraySectionExpr : public Expr {
  enum { BASE, LOWER_BOUND, LENGTH, END_EXPR };
  Stmt *SubExprs[END_EXPR];
  SourceLocation ColonLoc;
  SourceLocation RBracketLoc;

public:
  OSSArraySectionExpr(Expr *Base, Expr *LowerBound, Expr *Length, QualType Type,
                      ExprValueKind VK, ExprObjectKind OK,
                      SourceLocation ColonLoc, SourceLocation RBracketLoc)
      : Expr(
            OSSArraySectionExprClass, Type, VK, OK,
            Base->isTypeDependent() ||
                (LowerBound && LowerBound->isTypeDependent()) ||
                (Length && Length->isTypeDependent()),
            Base->isValueDependent() ||
                (LowerBound && LowerBound->isValueDependent()) ||
                (Length && Length->isValueDependent()),
            Base->isInstantiationDependent() ||
                (LowerBound && LowerBound->isInstantiationDependent()) ||
                (Length && Length->isInstantiationDependent()),
            Base->containsUnexpandedParameterPack() ||
                (LowerBound && LowerBound->containsUnexpandedParameterPack()) ||
                (Length && Length->containsUnexpandedParameterPack())),
        ColonLoc(ColonLoc), RBracketLoc(RBracketLoc) {
    SubExprs[BASE] = Base;
    SubExprs[LOWER_BOUND] = LowerBound;
    SubExprs[LENGTH] = Length;
  }

  /// Create an empty array section expression.
  explicit OSSArraySectionExpr(EmptyShell Shell)
      : Expr(OSSArraySectionExprClass, Shell) {}

  /// An array section can be written only as Base[LowerBound:Length].

  /// Get base of the array section.
  Expr *getBase() { return cast<Expr>(SubExprs[BASE]); }
  const Expr *getBase() const { return cast<Expr>(SubExprs[BASE]); }
  /// Set base of the array section.
  void setBase(Expr *E) { SubExprs[BASE] = E; }

  /// Return original type of the base expression for array section.
  static QualType getBaseOriginalType(const Expr *Base);

  /// Get lower bound of array section.
  Expr *getLowerBound() { return cast_or_null<Expr>(SubExprs[LOWER_BOUND]); }
  const Expr *getLowerBound() const {
    return cast_or_null<Expr>(SubExprs[LOWER_BOUND]);
  }
  /// Set lower bound of the array section.
  void setLowerBound(Expr *E) { SubExprs[LOWER_BOUND] = E; }

  /// Get length of array section.
  Expr *getLength() { return cast_or_null<Expr>(SubExprs[LENGTH]); }
  const Expr *getLength() const { return cast_or_null<Expr>(SubExprs[LENGTH]); }
  /// Set length of the array section.
  void setLength(Expr *E) { SubExprs[LENGTH] = E; }

  SourceLocation getBeginLoc() const LLVM_READONLY {
    return getBase()->getBeginLoc();
  }
  SourceLocation getEndLoc() const LLVM_READONLY { return RBracketLoc; }

  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  SourceLocation getRBracketLoc() const { return RBracketLoc; }
  void setRBracketLoc(SourceLocation L) { RBracketLoc = L; }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getBase()->getExprLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSArraySectionExprClass;
  }

  child_range children() {
    return child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }

  const_child_range children() const {
    return const_child_range(&SubExprs[BASE], &SubExprs[END_EXPR]);
  }
};

class OSSArrayShapingExpr final
  : public Expr,
    private llvm::TrailingObjects<OSSArrayShapingExpr, Stmt *> {

  friend TrailingObjects;

  unsigned NumShapes;

  SourceLocation BeginLoc;
  SourceLocation EndLoc;

  size_t numTrailingObjects(OverloadToken<Stmt *>) const {
    return NumShapes;
  }

  OSSArrayShapingExpr(QualType Type,
                      ExprValueKind VK, ExprObjectKind OK, unsigned N,
                      SourceLocation BeginLoc, SourceLocation EndLoc)
      : Expr( OSSArrayShapingExprClass, Type, VK, OK,
            false, false, false, false),
            NumShapes(N), BeginLoc(BeginLoc), EndLoc(EndLoc)
      {}

  /// Create an empty array section expression.
  explicit OSSArrayShapingExpr(EmptyShell Shell, unsigned N)
      : Expr(OSSArrayShapingExprClass, Shell), NumShapes(N) {}

public:

  static OSSArrayShapingExpr *Create(const ASTContext &C,
                                 QualType Type,
                                 ExprValueKind VK,
                                 ExprObjectKind OK,
                                 Expr *Base,
                                 ArrayRef<Expr *> ShapeList,
                                 SourceLocation BeginLoc,
                                 SourceLocation EndLoc) {
    void *Mem = C.Allocate(totalSizeToAlloc<Stmt *>(ShapeList.size() + 1));
    OSSArrayShapingExpr *Clause = new (Mem)
        OSSArrayShapingExpr(Type, VK, OK, ShapeList.size(), BeginLoc, EndLoc);
    Clause->setBase(Base);
    Clause->setShapes(ShapeList);
    return Clause;
  }

  /// Get base of the array section.
  Expr *getBase() { return cast<Expr>(getTrailingObjects<Stmt *>()[0]); }
  const Expr *getBase() const { return cast<Expr>(getTrailingObjects<Stmt *>()[0]); }
  /// Set base of the array section.
  void setBase(Expr *E) { getTrailingObjects<Stmt *>()[0] = E; }

  /// Get the shape of array shaping.
  MutableArrayRef<Stmt *> getShapes() {
    return MutableArrayRef<Stmt *>(
        getTrailingObjects<Stmt *>() + 1, NumShapes);
  }
  ArrayRef<const Stmt *> getShapes() const {
    return ArrayRef<Stmt *>(
        getTrailingObjects<Stmt *>() + 1, NumShapes);
  }
  /// Set the shape of the array shaping.
  void setShapes(ArrayRef<Expr *> VL) {
    std::copy(VL.begin(), VL.end(),
              getTrailingObjects<Stmt *>() + 1);
  }

  SourceLocation getBeginLoc() const LLVM_READONLY { return BeginLoc; }
  SourceLocation getEndLoc() const LLVM_READONLY { return EndLoc; }
  SourceRange getSourceRange() const LLVM_READONLY { return SourceRange(BeginLoc, EndLoc); }

  SourceLocation getExprLoc() const LLVM_READONLY {
    return getBase()->getBeginLoc();
  }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == OSSArrayShapingExprClass;
  }

  child_range children() {
    return child_range(getTrailingObjects<Stmt *>(), getTrailingObjects<Stmt *>() + NumShapes + 1);
  }

  const_child_range children() const {
    return const_child_range(getTrailingObjects<Stmt *>(), getTrailingObjects<Stmt *>() + NumShapes + 1);
  }
};
} // end namespace clang

#endif
