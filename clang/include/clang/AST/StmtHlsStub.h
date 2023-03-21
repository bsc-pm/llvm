//===- StmtHlsStub.h - Hls pragma directive stub ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the nodes for the stub HLS pragma directive
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMTHLSSTUB_H
#define LLVM_CLANG_AST_STMTHLSSTUB_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>

namespace clang {
class HlsDirective : public Stmt {
  friend class ASTStmtReader;
  /// Starting location of the directive (directive keyword).
  SourceLocation StartLoc;
  /// Ending location of the directive.
  SourceLocation EndLoc;
  size_t length;
  /// Build a stub directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param conent raw source content to print back.
  ///
  HlsDirective(SourceLocation StartLoc, SourceLocation EndLoc,
               StringRef &&content)
      : Stmt(StmtClass::HlsDirectiveClass), StartLoc(StartLoc), EndLoc(EndLoc) {
    std::copy(content.begin(), content.end(),
              reinterpret_cast<char *>(this) +
                  llvm::alignTo(sizeof(HlsDirective), alignof(void *)));
    length = content.size();
  }

public:
  const StringRef getContent() const {
    return StringRef(reinterpret_cast<const char *>(this) +
                         llvm::alignTo(sizeof(HlsDirective), alignof(void *)),
                     length);
  }

  template <class Alloc>
  static HlsDirective *Create(Alloc &C, SourceLocation StartLoc,
                              SourceLocation EndLoc,
                              llvm::StringRef &&content) {
    unsigned Size = llvm::alignTo(sizeof(HlsDirective), alignof(void *));
    void *Mem = C.Allocate(Size + content.size(), sizeof(void *));
    HlsDirective *Dir =
        new (Mem) HlsDirective(StartLoc, EndLoc, std::move(content));
    return Dir;
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

  static bool classof(const Stmt *S) {
    return S->getStmtClass() == HlsDirectiveClass;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }
};
} // namespace clang
#endif