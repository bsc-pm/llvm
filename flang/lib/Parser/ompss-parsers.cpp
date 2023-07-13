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

TYPE_PARSER(
    construct<OSSObject>(designator) || construct<OSSObject>("/" >> name / "/"))
TYPE_PARSER(construct<OSSObjectList>(nonemptyList(Parser<OSSObject>{})))

TYPE_PARSER(construct<OSSDefaultClause>(
    "PRIVATE" >> pure(OSSDefaultClause::Type::Private) ||
    "FIRSTPRIVATE" >> pure(OSSDefaultClause::Type::Firstprivate) ||
    "SHARED" >> pure(OSSDefaultClause::Type::Shared) ||
    "NONE" >> pure(OSSDefaultClause::Type::None)))

TYPE_PARSER(
    construct<OSSDependenceType>(first(
        "IN, WEAK" >> pure(OSSDependenceType::Type::Weakin),
        "IN"_id >> pure(OSSDependenceType::Type::In),
        "INOUT, WEAK" >> pure(OSSDependenceType::Type::Weakinout),
        "INOUT"_id >> pure(OSSDependenceType::Type::Inout),
        "INOUTSET, WEAK" >> pure(OSSDependenceType::Type::Weakinoutset),
        "INOUTSET" >> pure(OSSDependenceType::Type::Inoutset),
        "MUTEXINOUTSET, WEAK" >> pure(OSSDependenceType::Type::Weakmutexinoutset),
        "MUTEXINOUTSET" >> pure(OSSDependenceType::Type::Mutexinoutset),
        "OUT, WEAK" >> pure(OSSDependenceType::Type::Weakout),
        "OUT" >> pure(OSSDependenceType::Type::Out),
        "WEAK, IN"_id >> pure(OSSDependenceType::Type::Weakin),
        "WEAK, INOUT"_id >> pure(OSSDependenceType::Type::Weakinout),
        "WEAK, INOUTSET" >> pure(OSSDependenceType::Type::Weakinoutset),
        "WEAK, MUTEXINOUTSET" >> pure(OSSDependenceType::Type::Weakmutexinoutset),
        "WEAK, OUT" >> pure(OSSDependenceType::Type::Weakout))))

TYPE_CONTEXT_PARSER("OmpSs-2 Depend clause"_en_US,
    construct<OSSDependClause>(construct<OSSDependClause::InOut>(
        Parser<OSSDependenceType>{}, ":" >> Parser<OSSObjectList>{})))

TYPE_PARSER(construct<OSSReductionOperator>(Parser<DefinedOperator>{}) ||
    construct<OSSReductionOperator>(Parser<ProcedureDesignator>{}))

TYPE_PARSER(construct<OSSReductionClause>(
    Parser<OSSReductionOperator>{} / ":", nonemptyList(designator)))

// Clauses
TYPE_PARSER(
    "COST" >> construct<OSSClause>(construct<OSSClause::Cost>(
                      parenthesized(scalarIntExpr))) ||
    "CHUNKSIZE" >> construct<OSSClause>(construct<OSSClause::Chunksize>(
                       parenthesized(scalarIntExpr))) ||
    "COMMUTATIVE" >> construct<OSSClause>(construct<OSSClause::Commutative>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "CONCURRENT" >> construct<OSSClause>(construct<OSSClause::Concurrent>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "DEFAULT"_id >>
        construct<OSSClause>(parenthesized(Parser<OSSDefaultClause>{})) ||
    "DEPEND" >>
        construct<OSSClause>(parenthesized(Parser<OSSDependClause>{})) ||
    "FINAL" >> construct<OSSClause>(construct<OSSClause::Final>(
                   parenthesized(scalarLogicalExpr))) ||
    "FIRSTPRIVATE" >> construct<OSSClause>(construct<OSSClause::Firstprivate>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "GRAINSIZE" >> construct<OSSClause>(construct<OSSClause::Grainsize>(
                       parenthesized(scalarIntExpr))) ||
    "IF" >> construct<OSSClause>(construct<OSSClause::If>(
                   parenthesized(scalarLogicalExpr))) ||
    "IN"_id >> construct<OSSClause>(construct<OSSClause::In>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "INOUT" >> construct<OSSClause>(construct<OSSClause::Inout>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "LABEL" >> construct<OSSClause>(construct<OSSClause::Label>(
                      parenthesized(scalarDefaultCharExpr))) ||
    "ON" >> construct<OSSClause>(construct<OSSClause::Inout>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "ONREADY" >> construct<OSSClause>(construct<OSSClause::Onready>(
                    parenthesized(expr))) ||
    "OUT" >> construct<OSSClause>(construct<OSSClause::Out>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "PRIORITY" >> construct<OSSClause>(construct<OSSClause::Priority>(
                      parenthesized(scalarIntExpr))) ||
    "PRIVATE" >> construct<OSSClause>(construct<OSSClause::Private>(
                     parenthesized(Parser<OSSObjectList>{}))) ||
    "REDUCTION" >>
        construct<OSSClause>(parenthesized(Parser<OSSReductionClause>{})) ||
    "SHARED" >> construct<OSSClause>(construct<OSSClause::Shared>(
                    parenthesized(Parser<OSSObjectList>{}))) ||
    "SHMEM" >> construct<OSSClause>(construct<OSSClause::Shmem>(
                    parenthesized(scalarIntExpr))) ||
    "WAIT" >> construct<OSSClause>(construct<OSSClause::Wait>()) ||
    "WEAKCOMMUTATIVE" >> construct<OSSClause>(construct<OSSClause::Weakcommutative>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "WEAKCONCURRENT" >> construct<OSSClause>(construct<OSSClause::Weakconcurrent>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "WEAKIN"_id >> construct<OSSClause>(construct<OSSClause::Weakin>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "WEAKINOUT" >> construct<OSSClause>(construct<OSSClause::Weakinout>(
                          parenthesized(Parser<OSSObjectList>{}))) ||
    "WEAKOUT" >> construct<OSSClause>(construct<OSSClause::Weakout>(
                          parenthesized(Parser<OSSObjectList>{}))))

TYPE_PARSER(sourced(construct<OSSClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OSSClause>{})))))

// Simple standalone directives
TYPE_PARSER(sourced(construct<OSSSimpleStandaloneDirective>(first(
    "RELEASE" >> pure(llvm::oss::Directive::OSSD_release),
    "TASKWAIT" >> pure(llvm::oss::Directive::OSSD_taskwait)))))

TYPE_PARSER(sourced(construct<OmpSsSimpleStandaloneConstruct>(
    Parser<OSSSimpleStandaloneDirective>{}, Parser<OSSClauseList>{})))

TYPE_PARSER(sourced(construct<OmpSsStandaloneConstruct>(
                Parser<OmpSsSimpleStandaloneConstruct>{})) /
    endOfLine)

// Block directives
TYPE_PARSER(construct<OSSBlockDirective>(first(
    "TASK"_id >> pure(llvm::oss::Directive::OSSD_task))))

TYPE_PARSER(sourced(construct<OSSBeginBlockDirective>(
    sourced(Parser<OSSBlockDirective>{}), Parser<OSSClauseList>{})))

TYPE_PARSER(
    startOSSLine >> sourced(construct<OSSEndBlockDirective>(
                        sourced("END"_tok >> Parser<OSSBlockDirective>{}),
                        Parser<OSSClauseList>{})))

TYPE_PARSER(construct<OmpSsBlockConstruct>(
    Parser<OSSBeginBlockDirective>{} / endOSSLine, block,
    Parser<OSSEndBlockDirective>{} / endOSSLine))

// Loop directives
TYPE_PARSER(sourced(construct<OSSLoopDirective>(first(
    "TASK DO" >> pure(llvm::oss::Directive::OSSD_task_for),
    "TASKLOOP DO" >> pure(llvm::oss::Directive::OSSD_taskloop_for),
    "TASKLOOP" >> pure(llvm::oss::Directive::OSSD_taskloop)))))

TYPE_PARSER(sourced(construct<OSSBeginLoopDirective>(
    sourced(Parser<OSSLoopDirective>{}), Parser<OSSClauseList>{})))

TYPE_PARSER(
    startOSSLine >> sourced(construct<OSSEndLoopDirective>(
                        sourced("END"_tok >> Parser<OSSLoopDirective>{}),
                        Parser<OSSClauseList>{})))

TYPE_PARSER(construct<OmpSsLoopConstruct>(
    Parser<OSSBeginLoopDirective>{} / endOSSLine))

// Task outline directives
TYPE_PARSER(sourced(construct<OSSSimpleOutlineTaskDirective>(
    "TASK" >> pure(llvm::oss::Directive::OSSD_declare_task))))

TYPE_PARSER(sourced(construct<OmpSsSimpleOutlineTaskConstruct>(
    Parser<OSSSimpleOutlineTaskDirective>{}, Parser<OSSClauseList>{})))

TYPE_PARSER(
    construct<OmpSsOutlineTask>(
      statement(Parser<OmpSsOutlineTaskConstruct>()), indirect(Parser<SubroutineSubprogram>())))

TYPE_CONTEXT_PARSER("OmpSs-2 construct"_en_US,
    startOSSLine >>
        sourced(construct<OmpSsOutlineTaskConstruct>(
                Parser<OmpSsSimpleOutlineTaskConstruct>{})))

// OmpSs-2 Top level Executable statement
TYPE_CONTEXT_PARSER("OmpSs-2 construct"_en_US,
    startOSSLine >>
        first(
            construct<OmpSsConstruct>(Parser<OmpSsLoopConstruct>{}),
            construct<OmpSsConstruct>(Parser<OmpSsBlockConstruct>{}),
            construct<OmpSsConstruct>(Parser<OmpSsStandaloneConstruct>{})))

} // namespace Fortran::parser
