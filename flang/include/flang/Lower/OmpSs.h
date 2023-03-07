//===-- Lower/OmpSs.h -- lower OmpSs-2 directives --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OMPSS_H
#define FORTRAN_LOWER_OMPSS_H

#include "flang/Semantics/symbol.h"

namespace mlir {
class Value;
} // namespace mlir

namespace Fortran {
namespace parser {
struct OmpSsConstruct;
struct OmpSsOutlineTaskConstruct;
} // namespace parser

namespace semantics {
class Scope;
} // namespace semantics

namespace lower {

class AbstractConverter;
class StatementContext;
class SymMap;

namespace pft {
struct Evaluation;
} // namespace pft

struct ImplicitDSAs {
  std::vector<Fortran::evaluate::SymbolRef> sharedList;
  std::vector<Fortran::evaluate::SymbolRef> privateList;
  std::vector<Fortran::evaluate::SymbolRef> firstprivateList;
};

void genOmpSsConstruct(
  AbstractConverter &, pft::Evaluation &,
  const parser::OmpSsConstruct &, const Fortran::lower::ImplicitDSAs &,
  Fortran::semantics::SemanticsContext &);

mlir::Value genOmpSsTaskSubroutine(
  AbstractConverter &,
  Fortran::lower::pft::Evaluation &,
  const Fortran::semantics::Scope &,
  const Fortran::parser::OmpSsOutlineTaskConstruct &,
  Fortran::semantics::SemanticsContext &,
  const Fortran::evaluate::ProcedureRef &,
  StatementContext &);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OMPSS_H
