#!/usr/bin/env python

import sys
import json
import collections
import logging

IMPLEMENTED_LMULS = [1, 2, 4, 8]

# Prototype kinds
PRIMARIES = "evwm0us"
MODIFIERS = "PKCUIF"


# Types can be rendered in several ways
class Type:
    def __init__(self):
        pass

    def isPrimaryType(self):
        return False

    def isIntegerType(self):
        return False


class TypePrimary(Type):
    def __init__(self, letter):
        self.letter = letter

    def __str__(self):
        return self.letter

    def isPrimaryType(self):
        return True

    def isIntegerType(self):
        return self.letter in ["c", "s", "i", "l"]

    def render_for_name(self):
        if self.letter == "c":
            return "i8"
        elif self.letter == "s":
            return "i16"
        elif self.letter == "i":
            return "i32"
        elif self.letter == "l":
            return "i64"
        elif self.letter == "f":
            return "f32"
        elif self.letter == "d":
            return "f64"
        elif self.letter == "m":
            return "i1"
        else:
            raise Exception("Unknown letter {}".format(self.letter))

    def render_for_clang(self):
        if self.letter == "c":
            return "Sc"
        elif self.letter == "s":
            return "Ss"
        elif self.letter == "i":
            return "Si"
        elif self.letter == "l":
            return "SWi"
        elif self.letter == "f":
            return "f"
        elif self.letter == "d":
            return "d"
        elif self.letter == "m":
            return "b"
        else:
            raise Exception("Unknown letter {}".format(self.letter))


class TypeConstant(Type):
    def __init__(self, letter):
        self.letter = letter

    def __str__(self):
        return self.letter

    def render_for_name(self):
        raise Exception("No type constant can appear in the name")

    def render_for_clang(self):
        if self.letter == "0":
            return "v"
        elif self.letter == "s":
            return "SWi"
        elif self.letter == "u":
            return "UWi"
        else:
            raise Exception("Unknown type constant {}".format(self.letter))


class TypeVector(Type):
    def __init__(self, element_type, scale):
        self.element_type = element_type
        self.scale = scale

    def __str__(self):
        return "<" + self.scale + " x " + str(self.element_type) + ">"

    def render_for_name(self):
        t = self.element_type.render_for_name()
        return "{}x{}".format(self.scale, t)

    def render_for_clang(self):
        t = self.element_type.render_for_clang()
        return "QV{}{}".format(self.scale, t)


class TypeModified(Type):
    def __init__(self, modified_type, modifier):
        self.modified_type = modified_type
        self.modifier = modifier

    def __str__(self):
        return self.modifier + str(self.modified_type)

    def isIntegerType(self):
        return self.modified_type.isIntegerType()

    def isPrimaryType(self):
        return self.modified_type.isPrimaryType()

    def render_for_name(self):
        raise Exception("No modified types appear in the name")

    def render_for_clang(self):
        t = self.modified_type.render_for_clang()
        if self.modifier == "P":
            return "{}*".format(t)
        elif self.modifier == "C":
            return "{}C".format(t)
        elif self.modifier == "K":
            return t.replace("i", "Ii")
        elif self.modifier == "U":
            return "U{}".format(t.replace("S", ""))
        else:
            raise Exception("Unexpected modified type {}".format(
                self.modifier))


# Prototypes are evaluated using a type spec and a LMUL to generate a Type.
class Prototype:
    def __init__(self):
        pass

    def evaluate(self, type_spec, lmul):
        raise Exception("Method 'evaluate' must be overriden")


class PrototypePrimary(Prototype):
    def __init__(self, letter):
        assert (letter in PRIMARIES)
        self.letter = letter

    def __str__(self):
        return self.letter

    def __computeVector(self, type_spec, lmul):
        if type_spec == "c":
            scale = 8
        elif type_spec == "s":
            scale = 4
        elif type_spec in ["f", "i"]:
            scale = 2
        elif type_spec in ["l", "d"]:
            scale = 1
        else:
            raise Exception("Unhandled type_spec '{}'".format(type_spec))
        scale *= lmul
        return TypeVector(TypePrimary(type_spec), scale)

    def __computeWideVector(self, type_spec, lmul):
        if type_spec == 'c':
            scale = 8
            base_type = TypePrimary("s")
        elif type_spec == "s":
            scale = 4
            base_type = TypePrimary("i")
        elif type_spec in ["f", "i"]:
            if type_spec == "f":
                base_type = TypePrimary("d")
            else:
                base_type = TypePrimary("l")
            scale = 2
        else:
            raise Exception("Unhandled type_spec '{}'".format(type_spec))
        scale *= lmul
        return TypeVector(base_type, scale)

    def __computeMaskVector(self, type_spec, lmul):
        if type_spec == 'c':
            scale = 8
        elif type_spec == "s":
            scale = 4
        elif type_spec in ["f", "i"]:
            scale = 2
        elif type_spec in ["l", "d"]:
            scale = 1
        else:
            raise Exception("Unhandled type_spec '{}'".format(type_spec))
        scale *= lmul
        return TypeVector(TypePrimary("m"), scale)

    def evaluate(self, type_spec, lmul):
        if self.letter == "e":
            return TypePrimary(type_spec)
        elif self.letter == "v":
            return self.__computeVector(type_spec, lmul)
        elif self.letter == "w":
            return self.__computeWideVector(type_spec, lmul)
        elif self.letter == "m":
            return self.__computeMaskVector(type_spec, lmul)
        elif self.letter in ["0", "u", "s"]:
            return TypeConstant(self.letter)
        else:
            raise Exception("Unhandled letter '{}'".format(self.letter))


class PrototypeModifier(Prototype):
    def __init__(self, letter, prototype):
        assert (letter in MODIFIERS)
        self.letter = letter
        self.prototype = prototype

    def __str__(self):
        return self.letter + str(self.prototype)

    def evaluate(self, type_spec, lmul):
        t = self.prototype.evaluate(type_spec, lmul)
        if self.letter == "P":
            if not t.isPrimaryType():
                raise Exception(
                    "P modifier can only be applied to primary types. Current type is '{}'"
                    .format(t))
            return TypeModified(t, self.letter)
        elif self.letter == "C":
            return TypeModified(t, self.letter)
        elif self.letter in ["U", "K"]:
            if not t.isIntegerType():
                raise Exception(
                    "U or K can only be applied to integer types. Current type is '{}'"
                    .format(t))
            return TypeModified(t, self.letter)
        elif self.letter in ["I", "F"]:
            if not isinstance(t, TypeVector):
                raise Exception(
                    "F or I can only be applied to vector types. Current type is '{}'"
                    .format(t))
            element_type = t.element_type
            if not isinstance(element_type, TypePrimary):
                raise Exception(
                    "Expecting a primary type in a vector but type is '{}".
                    format(element_type))
            if element_type.letter in ["f", "i"]:
                if self.letter == "F":
                    new_element_type = TypePrimary("f")
                else:
                    new_element_type = TypePrimary("i")
            elif element_type.letter in ["d", "l"]:
                if self.letter == "F":
                    new_element_type = TypePrimary("d")
                else:
                    new_element_type = TypePrimary("l")
            elif element_type.letter in ["c", "s"] and self.letter == "I":
                new_element_type = TypePrimary(element_type.letter)
            else:
                raise Exception(
                    "Cannot convert element type '{}' to floating or integer vector"
                    .format(element_type.letter))
            return TypeVector(new_element_type, t.scale)


def process_tblgen_file(tablegen, input_tablegen):
    import subprocess
    t = subprocess.check_output([tablegen, "--dump-json", input_tablegen])
    return json.loads(t.decode("utf-8"))


def parse_single_prototype(prototype):
    # Reverse
    p = prototype[::-1]
    assert (p[0] in PRIMARIES)
    t = PrototypePrimary(p[0])
    p = p[1:]
    for l in p:
        assert (l in MODIFIERS)
        t = PrototypeModifier(l, t)
    return t


def parse_prototype_seq(prototype_seq):
    res = []
    current_prototype = ""
    for current_letter in prototype_seq:
        current_prototype += current_letter
        if current_letter in PRIMARIES:
            res.append(parse_single_prototype(current_prototype))
            current_prototype = ""
        elif current_letter in MODIFIERS:
            pass
        else:
            raise Exception("Invalid prototype letter {} in {}".format(
                current_letter, prototype_seq))

    if current_prototype != "":
        raise Exception("Prototype {} is incomplete".format(prototype_seq))
    return res


def render_type_for_name(prototype, type_spec, lmul):
    ty = prototype.evaluate(type_spec, lmul)
    return ty.render_for_name()


def render_types_for_name(suffix_types, type_spec, lmul):
    prototype = parse_prototype_seq(suffix_types)
    res = []
    for proto in prototype:
        res.append(render_type_for_name(proto, type_spec, lmul))
    return res


def render_type_for_clang(prototype, type_spec, lmul):
    ty = prototype.evaluate(type_spec, lmul)
    return ty.render_for_clang()


def compute_builtin_type_clang(builtin, prototype, type_spec, lmul):
    res = []
    for proto in prototype:
        res.append(render_type_for_clang(proto, type_spec, lmul))
    return "".join(res)


class InstantiatedBuiltin:
    def __init__(self, full_name, type_description, flags, builtin, prototype,
            masked = False, index_of_mask = None):
        self.full_name = full_name
        self.type_description = type_description
        self.flags = flags
        self.builtin = builtin
        self.prototype = prototype
        self.masked = masked
        self.index_of_mask = index_of_mask

    def __str__(self):
        return "EPI_BUILTIN({}, \"{}\", \"{}\")".format(self.full_name, \
            self.type_description, self.flags)


def compute_builtin_name(builtin, prototype, type_spec, lmul):
    res = "{}".format(builtin["Name"])
    if not builtin["Suffix"]:
        return res
    rendered_types = render_types_for_name(builtin["Suffix"], type_spec, lmul)
    res += "_" + "_".join(rendered_types)
    return res


def compute_single_builtin_defs(builtin, orig_prototype, type_spec, lmul):
    builtin_list = []
    prototype = orig_prototype[:]

    if builtin["HasVL"] != 0:
        # Add GVL operand
        prototype.append(PrototypePrimary("u"))

    full_name = compute_builtin_name(builtin, prototype, type_spec, lmul)
    type_description = compute_builtin_type_clang(builtin, prototype,
                                                  type_spec, lmul)
    flags = ""
    if builtin["HasSideEffects"] == 0:
        flags = "n"

    builtin_list.append(InstantiatedBuiltin(full_name, type_description,
        flags, builtin, prototype))

    if builtin["HasMask"] != 0:
        # Emit masked variant
        prototype_mask = orig_prototype[:]

        if builtin["HasMergeOperand"]:
            # Add merge operand
            prototype_mask.insert(1, prototype_mask[0])

        index_of_mask = len(prototype_mask)
        prototype_mask.append(PrototypePrimary("m"))

        if builtin["HasVL"] != 0:
            # Add GVL operand
            prototype_mask.append(PrototypePrimary("u"))

        type_description = compute_builtin_type_clang(builtin, prototype_mask,
                                                      type_spec, lmul)
        builtin_list.append(InstantiatedBuiltin(full_name + "_mask",
            type_description, flags, builtin, prototype_mask,
            masked = True, index_of_mask = index_of_mask))

    return builtin_list


def instantiate_builtins(j):
    all_builtins = j["!instanceof"]["EPIBuiltin"]
    result = []
    for builtin_instance in all_builtins:
        builtin = j[builtin_instance]
        prototype = parse_prototype_seq(builtin["Prototype"])
        for lmul in IMPLEMENTED_LMULS:
            if lmul in builtin["LMUL"]:
                for type_spec in builtin["TypeRange"]:
                    result += compute_single_builtin_defs(builtin, prototype, type_spec, lmul)
    return result;


def emit_builtins_def(out_file, j):
    # Ideally we should not generate repeated builtins but some
    # masked operations overlap while ranging over LMUL and they generate
    # repeated builtins.
    # We use an OrderedDict as a form of OrderedSet. This is not great
    # but avoids using external packages or having to roll our own
    # data-structure.
    builtin_set = collections.OrderedDict()
    builtin_names = {}

    error = False

    inst_builtins = instantiate_builtins(j)

    for b in inst_builtins:
        if b.full_name in builtin_names and str(b) not in builtin_set:
            logging.error("Builtin '{}' has already been defined as '{}' but this one is '{}'".format(b.full_name, str(builtin_names[b.full_name]), str(b)))
            error = True
        builtin_set[str(b)] = None
        builtin_names[b.full_name] = b

    if error:
        raise Exception("Errors found while emitting the builtins")

    out_file.write("""\
#if defined(BUILTIN) && !defined(EPI_BUILTIN)
#define EPI_BUILTIN(ID, TYPE, ATTRS) BUILTIN(ID, TYPE, ATTRS)
#endif

""")

    for b in builtin_set.keys():
        out_file.write(b + "\n")

    out_file.write("""
#undef BUILTIN
#undef EPI_BUILTIN
""")


def emit_codegen(out_file, j):
    inst_builtins = instantiate_builtins(j)

    code_set = {}
    for b in inst_builtins:
        code_case = ""
        if b.builtin["HasManualCodegen"] != 0:
            s = ""
            if b.masked:
                s = b.builtin["ManualCodegenMask"]
            else:
                s = b.builtin["ManualCodegen"]
            if not s.endswith("\n"):
                s += "\n"
            code_case += s;
        elif b.builtin["IntrinsicTypes"]:
            code_case += "  IntrinsicTypes = { ";
            intr_types = []
            source_intrinsic_types = b.builtin["IntrinsicTypes"][:]
            if b.masked:
                if b.builtin["HasMergeOperand"]:
                    # Skew the operands
                    for i in range(0, len(source_intrinsic_types)):
                        if source_intrinsic_types[i] >= 0:
                            source_intrinsic_types[i] += 1

            for t in source_intrinsic_types:
                if t == -1:
                    intr_types.append("ResultType");
                elif t == -2:
                    if b.masked:
                        intr_types.append("Ops[{}]->getType()".format(b.index_of_mask - 1));
                else:
                    if b.masked and t == b.index_of_mask:
                        raise Exception("Cannot refer to the mask operand like a regular operand")
                    intr_types.append("Ops[{}]->getType()".format(t));
            code_case += ", ".join(intr_types)
            code_case += " };\n";

        if not b.masked and b.builtin["IntrinsicName"]:
            code_case += "  ID = Intrinsic::epi_{};\n".format(b.builtin["IntrinsicName"]);
        if b.masked and b.builtin["IntrinsicNameMask"]:
            code_case += "  ID = Intrinsic::epi_{};\n".format(b.builtin["IntrinsicNameMask"]);
        code_case += "  break;\n"

        if code_case not in code_set:
            # Sometimes we may have repeated cases here (e.g. masks) so use a set
            code_set[code_case] = set([b.full_name])
        else:
            code_set[code_case].add(b.full_name)

    for (code, cases) in code_set.items():
        for case in cases:
            out_file.write("case RISCV::EPI_BI_{}:\n".format(case))
        out_file.write(code)

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser(description="Generate instruction table")
    args.add_argument("--mode", required=True,
            choices=["builtins-def", "codegen"], help="Mode of operation")
    args.add_argument("--tablegen", required=True, help="Path of tablegen")
    args.add_argument("--output-file", required=False, help="Output file. stdout otherwise")
    args.add_argument("input_tblgen",
                      help="File with the tablegen description")
    args = args.parse_args()

    out_file = sys.stdout
    if args.output_file:
        out_file = open(args.output_file, "w")

    j = process_tblgen_file(args.tablegen, args.input_tblgen)

    if args.mode == "builtins-def":
        emit_builtins_def(out_file, j)
    elif args.mode == "codegen":
        emit_codegen(out_file, j)
    else:
        raise Exception("Unexpected mode '{}".format(args.mode))

    out_file.close()
