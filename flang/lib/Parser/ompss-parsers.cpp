//===-- lib/Parser/ompss-parsers.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Top-level grammar specification for OmpSs-2.

#include "basic-parsers.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/parse-tree.h"

// OmpSs Directives and Clauses
namespace Fortran::parser {

constexpr auto startOSSLine = skipStuffBeforeStatement >> "!$OSS "_sptok;
constexpr auto endOSSLine = space >> endOfLine;

TYPE_PARSER(construct<OSSObject>(designator))
TYPE_PARSER(construct<OSSObjectList>(nonemptyList(Parser<OSSObject>{})))

TYPE_PARSER(construct<OSSDefaultClause>(
    "PRIVATE" >> pure(OSSDefaultClause::Type::Private) ||
    "FIRSTPRIVATE" >> pure(OSSDefaultClause::Type::Firstprivate) ||
    "SHARED" >> pure(OSSDefaultClause::Type::Shared) ||
    "NONE" >> pure(OSSDefaultClause::Type::None)))

TYPE_PARSER(construct<OSSDependSinkVecLength>(
    Parser<DefinedOperator>{}, scalarIntConstantExpr))

TYPE_PARSER(
    construct<OSSDependSinkVec>(name, maybe(Parser<OSSDependSinkVecLength>{})))

TYPE_PARSER(
    construct<OSSDependenceType>("IN"_id >> pure(OSSDependenceType::Type::In) ||
        "INOUT" >> pure(OSSDependenceType::Type::Inout) ||
        "OUT" >> pure(OSSDependenceType::Type::Out)))

TYPE_CONTEXT_PARSER("OmpSs-2 Depend clause"_en_US,
    construct<OSSDependClause>(construct<OSSDependClause::Sink>(
        "SINK :" >> nonemptyList(Parser<OSSDependSinkVec>{}))) ||
        construct<OSSDependClause>(
            construct<OSSDependClause::Source>("SOURCE"_tok)) ||
        construct<OSSDependClause>(construct<OSSDependClause::InOut>(
            Parser<OSSDependenceType>{}, ":" >> nonemptyList(designator))))

TYPE_PARSER(construct<OSSReductionOperator>(Parser<DefinedOperator>{}) ||
    construct<OSSReductionOperator>(Parser<ProcedureDesignator>{}))

TYPE_PARSER(construct<OSSReductionClause>(
    Parser<OSSReductionOperator>{} / ":", nonemptyList(designator)))

// Clauses
TYPE_PARSER(
    "COST" >> construct<OSSClause>(construct<OSSClause::Cost>(
                      parenthesized(scalarIntExpr))) ||
    "DEFAULT"_id >>
        construct<OSSClause>(parenthesized(Parser<OSSDefaultClause>{})) ||
    "DEPEND" >>
        construct<OSSClause>(parenthesized(Parser<OSSDependClause>{})) ||
    "FINAL" >> construct<OSSClause>(construct<OSSClause::Final>(
                   parenthesized(scalarLogicalExpr))) ||
    "FIRSTPRIVATE" >> construct<OSSClause>(construct<OSSClause::Firstprivate>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "IF" >> construct<OSSClause>(construct<OSSClause::If>(
                   parenthesized(scalarLogicalExpr))) ||
    "PRIVATE" >> construct<OSSClause>(construct<OSSClause::Private>(
                     parenthesized(Parser<OSSObjectList>{}))) ||
    "REDUCTION" >>
        construct<OSSClause>(parenthesized(Parser<OSSReductionClause>{})) ||
    "SHARED" >> construct<OSSClause>(construct<OSSClause::Shared>(
                    parenthesized(Parser<OSSObjectList>{}))))

TYPE_PARSER(sourced(construct<OSSClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OSSClause>{})))))

// Simple standalone directives
TYPE_PARSER(sourced(construct<OSSSimpleStandaloneDirective>(first(
    "TASKWAIT" >> pure(OSSSimpleStandaloneDirective::Directive::Taskwait)))))

TYPE_PARSER(sourced(construct<OmpSsSimpleStandaloneConstruct>(
    Parser<OSSSimpleStandaloneDirective>{}, Parser<OSSClauseList>{})))

// Block directives
TYPE_PARSER(construct<OSSBlockDirective>(
    first("TASK"_id >> pure(OSSBlockDirective::Directive::Task))))

TYPE_PARSER(sourced(construct<OSSBeginBlockDirective>(
    sourced(Parser<OSSBlockDirective>{}), Parser<OSSClauseList>{})))

TYPE_PARSER(
    startOSSLine >> sourced(construct<OSSEndBlockDirective>(
                        sourced("END"_tok >> Parser<OSSBlockDirective>{}),
                        Parser<OSSClauseList>{})))

TYPE_PARSER(construct<OmpSsBlockConstruct>(
    Parser<OSSBeginBlockDirective>{} / endOSSLine, block,
    Parser<OSSEndBlockDirective>{} / endOSSLine))

TYPE_PARSER(sourced(construct<OmpSsStandaloneConstruct>(
                Parser<OmpSsSimpleStandaloneConstruct>{})) /
    endOfLine)

// OmpSs-2 Top level Executable statement
TYPE_CONTEXT_PARSER("OmpSs-2 construct"_en_US,
    startOSSLine >>
        first(construct<OmpSsConstruct>(Parser<OmpSsBlockConstruct>{}),
            construct<OmpSsConstruct>(Parser<OmpSsStandaloneConstruct>{})))

} // namespace Fortran::parser
