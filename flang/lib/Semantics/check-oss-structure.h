//===-- lib/Semantics/check-oss-structure.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OpenMP structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. TODO: invalid nesting of regions

#ifndef FORTRAN_SEMANTICS_CHECK_OSS_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_OSS_STRUCTURE_H_

#include "check-directive-structure.h"
#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"

using OSSDirectiveSet = Fortran::common::EnumSet<llvm::oss::Directive,
    llvm::oss::Directive_enumSize>;

using OSSClauseSet =
    Fortran::common::EnumSet<llvm::oss::Clause, llvm::oss::Clause_enumSize>;

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OmpSs/OSS.cpp.inc"

namespace llvm {
namespace oss {
    // TODO: Directive sets
} // namespace oss
} // namespace llvm

namespace Fortran::semantics {

class OSSStructureChecker
    : public DirectiveStructureChecker<llvm::oss::Directive, llvm::oss::Clause,
          parser::OSSClause, llvm::oss::Clause_enumSize> {
public:
  OSSStructureChecker(SemanticsContext &context)
      : DirectiveStructureChecker(context,
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OmpSs/OSS.cpp.inc"
        ) {
  }

  // TODO: add directives and clauses
  // void Enter(const parser::OmpSs... &);

private:

  llvm::StringRef getClauseName(llvm::oss::Clause clause) override;
  llvm::StringRef getDirectiveName(llvm::oss::Directive directive) override;

};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OSS_STRUCTURE_H_
