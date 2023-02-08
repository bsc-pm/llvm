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

// Use when clause falls under 'struct OmpClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_CLAUSE(X, Y) \
  void OSSStructureChecker::Enter(const parser::OSSClause::X &) { \
    CheckAllowed(llvm::oss::Clause::Y); \
  }

#define CHECK_REQ_CONSTANT_SCALAR_INT_CLAUSE(X, Y) \
  void OSSStructureChecker::Enter(const parser::OSSClause::X &c) { \
    CheckAllowed(llvm::oss::Clause::Y); \
    RequiresConstantPositiveParameter(llvm::oss::Clause::Y, c.v); \
  }

#define CHECK_REQ_SCALAR_INT_CLAUSE(X, Y) \
  void OSSStructureChecker::Enter(const parser::OSSClause::X &c) { \
    CheckAllowed(llvm::oss::Clause::Y); \
    RequiresPositiveParameter(llvm::oss::Clause::Y, c.v); \
  }

// Use when clause don't falls under 'struct OSSClause' in 'parse-tree.h'.
#define CHECK_SIMPLE_PARSER_CLAUSE(X, Y) \
  void OSSStructureChecker::Enter(const parser::X &) { \
    CheckAllowed(llvm::oss::Y); \
  }

void OSSStructureChecker::Enter(const parser::OmpSsConstruct &) {
}

void OSSStructureChecker::Enter(const parser::OmpSsLoopConstruct &x) {
  const auto &beginLoopDir{std::get<parser::OSSBeginLoopDirective>(x.t)};
  const auto &beginDir{std::get<parser::OSSLoopDirective>(beginLoopDir.t)};

  // check matching, End directive is optional
  if (const auto &endLoopDir{
          std::get<std::optional<parser::OSSEndLoopDirective>>(x.t)}) {
    const auto &endDir{
        std::get<parser::OSSLoopDirective>(endLoopDir.value().t)};

    CheckMatching<parser::OSSLoopDirective>(beginDir, endDir);
  }
  PushContextAndClauseSets(beginDir.source, beginDir.v);
}

void OSSStructureChecker::Leave(const parser::OmpSsLoopConstruct &) {
  dirContext_.pop_back();
}

void OSSStructureChecker::Enter(const parser::OSSEndLoopDirective &x) {
  const auto &dir{std::get<parser::OSSLoopDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  default:
    // no clauses are allowed
    break;
  }
}


void OSSStructureChecker::Enter(const parser::OmpSsBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OSSBeginBlockDirective>(x.t)};
  const auto &endBlockDir{std::get<parser::OSSEndBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OSSBlockDirective>(beginBlockDir.t)};
  const auto &endDir{std::get<parser::OSSBlockDirective>(endBlockDir.t)};
  const parser::Block &block{std::get<parser::Block>(x.t)};

  CheckMatching<parser::OSSBlockDirective>(beginDir, endDir);

  PushContextAndClauseSets(beginDir.source, beginDir.v);

  switch (beginDir.v) {
  default:
    CheckNoBranching(block, beginDir.v, beginDir.source);
    break;
  }
}

void OSSStructureChecker::Leave(const parser::OmpSsBlockConstruct &) {
  dirContext_.pop_back();
}

void OSSStructureChecker::Enter(const parser::OSSEndBlockDirective &x) {
  const auto &dir{std::get<parser::OSSBlockDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  default:
    // no clauses are allowed
    break;
  }
}


void OSSStructureChecker::Enter(const parser::OmpSsSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OSSSimpleStandaloneDirective>(x.t)};
  PushContextAndClauseSets(dir.source, dir.v);
}

void OSSStructureChecker::Leave(const parser::OmpSsSimpleStandaloneConstruct &) {
  dirContext_.pop_back();
}

void OSSStructureChecker::Enter(const parser::OmpSsSimpleOutlineTaskConstruct &x) {
  const auto &dir{std::get<parser::OSSSimpleOutlineTaskDirective>(x.t)};
  PushContextAndClauseSets(dir.source, dir.v);
} 

void OSSStructureChecker::Leave(const parser::OmpSsSimpleOutlineTaskConstruct &) {
  dirContext_.pop_back();
}

void OSSStructureChecker::Leave(const parser::OSSClauseList &) {
  CheckRequireAtLeastOneOf();
}

void OSSStructureChecker::Enter(const parser::OSSClause &x) {
  SetContextClause(x);
}


CHECK_SIMPLE_CLAUSE(If, OSSC_if)

CHECK_SIMPLE_CLAUSE(Final, OSSC_final)

CHECK_REQ_SCALAR_INT_CLAUSE(Cost, OSSC_cost)

CHECK_SIMPLE_CLAUSE(Priority, OSSC_priority)

CHECK_SIMPLE_CLAUSE(Label, OSSC_label)

CHECK_SIMPLE_CLAUSE(Wait, OSSC_wait)

CHECK_SIMPLE_CLAUSE(Update, OSSC_update)

CHECK_SIMPLE_CLAUSE(Onready, OSSC_onready)

CHECK_SIMPLE_CLAUSE(Default, OSSC_default)

CHECK_SIMPLE_CLAUSE(Device, OSSC_device)

CHECK_SIMPLE_CLAUSE(Private, OSSC_private)

CHECK_SIMPLE_CLAUSE(Firstprivate, OSSC_firstprivate)

CHECK_SIMPLE_CLAUSE(Shared, OSSC_shared)

CHECK_SIMPLE_CLAUSE(On, OSSC_on)

CHECK_SIMPLE_CLAUSE(In, OSSC_in)

CHECK_SIMPLE_CLAUSE(Out, OSSC_out)

CHECK_SIMPLE_CLAUSE(Inout, OSSC_inout)

CHECK_SIMPLE_CLAUSE(Concurrent, OSSC_concurrent)

CHECK_SIMPLE_CLAUSE(Commutative, OSSC_commutative)

CHECK_SIMPLE_CLAUSE(Weakin, OSSC_weakin)

CHECK_SIMPLE_CLAUSE(Weakout, OSSC_weakout)

CHECK_SIMPLE_CLAUSE(Weakinout, OSSC_weakinout)

CHECK_SIMPLE_CLAUSE(Weakconcurrent, OSSC_weakconcurrent)

CHECK_SIMPLE_CLAUSE(Weakcommutative, OSSC_weakcommutative)

CHECK_REQ_SCALAR_INT_CLAUSE(Grainsize, OSSC_grainsize)
CHECK_REQ_SCALAR_INT_CLAUSE(Chunksize, OSSC_chunksize)

CHECK_SIMPLE_CLAUSE(Unroll, OSSC_unroll)

CHECK_SIMPLE_CLAUSE(Collapse, OSSC_collapse)

CHECK_SIMPLE_CLAUSE(Ndrange, OSSC_ndrange)

CHECK_SIMPLE_CLAUSE(Read, OSSC_read)

CHECK_SIMPLE_CLAUSE(Write, OSSC_write)

CHECK_SIMPLE_CLAUSE(Capture, OSSC_capture)

CHECK_SIMPLE_CLAUSE(Compare, OSSC_compare)

CHECK_SIMPLE_CLAUSE(SeqCst, OSSC_seq_cst)

CHECK_SIMPLE_CLAUSE(AcqRel, OSSC_acq_rel)

CHECK_SIMPLE_CLAUSE(Acquire, OSSC_acquire)

CHECK_SIMPLE_CLAUSE(Release, OSSC_release)

CHECK_SIMPLE_CLAUSE(Relaxed, OSSC_relaxed)

CHECK_SIMPLE_CLAUSE(Unknown, OSSC_unknown)

void OSSStructureChecker::Enter(const parser::OSSClause::Depend &) {
  CheckAllowed(llvm::oss::Clause::OSSC_depend);
}

void OSSStructureChecker::Enter(const parser::OSSClause::Reduction &) {
  CheckAllowed(llvm::oss::Clause::OSSC_reduction);
}

void OSSStructureChecker::Enter(const parser::OSSClause::Weakreduction &) {
  CheckAllowed(llvm::oss::Clause::OSSC_weakreduction);
}

llvm::StringRef OSSStructureChecker::getClauseName(llvm::oss::Clause clause) {
  return llvm::oss::getOmpSsClauseName(clause);
}

llvm::StringRef OSSStructureChecker::getDirectiveName(
    llvm::oss::Directive directive) {
  return llvm::oss::getOmpSsDirectiveName(directive);
}

} // namespace Fortran::semantics
