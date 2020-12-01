//===-- lib/Semantics/check-oss-structure.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-oss-structure.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

llvm::StringRef OSSStructureChecker::getClauseName(llvm::oss::Clause clause) {
  return llvm::oss::getOmpSsClauseName(clause);
}

llvm::StringRef OSSStructureChecker::getDirectiveName(
    llvm::oss::Directive directive) {
  return llvm::oss::getOmpSsDirectiveName(directive);
}

} // namespace Fortran::semantics
