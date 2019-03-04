#!/usr/bin/env python2

from preprocess_builtins import *
from builtin_parser import *
from type_render import *

import string

class TestTemplate(object):
    RESULT_ID = 0
    def __init__(self):
        pass

    def store_code(self, result_type):
        store_var = "p{}".format(TestTemplate.RESULT_ID)
        TestTemplate.RESULT_ID += 1

        store_decl = ""
        store_stmt = ""
        if result_type.scalable_vector:
            if result_type.basic_type == TypeBuilder.INT:
                if result_type.short:
                    store_decl = "short";
                    store_stmt = "__builtin_epi_vstore_i16({}, result);".format(store_var)
                elif result_type.long:
                    store_decl = "long";
                    store_stmt = "__builtin_epi_vstore_i64({}, result);".format(store_var)
                else:
                    store_decl = "int";
                    store_stmt = "__builtin_epi_vstore_i32({}, result);".format(store_var)
            elif result_type.basic_type == TypeBuilder.CHAR:
                store_decl = "signed char"
                store_stmt = "__builtin_epi_vstore_i8({}, result);".format(store_var)
            elif result_type.basic_type == TypeBuilder.BOOL:
                pass
                # FIXME - We can't store a mask, so we will need to convert it reinterpret to an integer first
                # TODO - We don't have reinterpret casts yet!
                # store_decl = "_Bool"
                # store_stmt = "__builtin_epi_vstore_i1({}, result);".format(store_var)
            elif result_type.basic_type == TypeBuilder.FLOAT:
                store_decl = "float"
                store_stmt = "__builtin_epi_vstore_f32({}, result);".format(store_var)
            elif result_type.basic_type == TypeBuilder.DOUBLE:
                store_decl += "double"
                store_stmt = "__builtin_epi_vstore_f64({}, result);".format(store_var)
            else:
                assert False, "Unreachable"
        else:
            if result_type.basic_type in [TypeBuilder.INT, TypeBuilder.CHAR]:
                if result_type.unsigned:
                    store_decl += "unsigned "
                elif (result_type.basic_type != TypeBuilder.CHAR and result_type.signed) \
                        or result_type.basic_type == TypeBuilder.INT:
                    store_decl += "signed "
                else:
                    assert result_type.basic_type == TypeBuilder.CHAR, "only char may be left without signedness"

                if result_type.short:
                    store_decl += "short "
                elif result_type.long:
                    if result_type.long == 1:
                        store_decl += "long "
                    elif result_type.long == 2:
                        store_decl += "long long "
                    else:
                        assert False, "unreachable"

                if result_type.basic_type == TypeBuilder.CHAR:
                    store_decl += "char"
                elif result_type.basic_type == TypeBuilder.INT:
                    store_decl += "int"
                else:
                    assert False, "unreachable"
            elif result_type.basic_type == TypeBuilder.BOOL:
                store_decl += "_Bool"
            elif result_type.basic_type == TypeBuilder.FLOAT:
                store_decl += "float"
            elif result_type.basic_type == TypeBuilder.DOUBLE:
                store_decl += "double"
            elif result_type.basic_type == TypeBuilder.VOID:
                store_decl += "void"
            else:
                assert False, "unreachable"
            store_stmt = "*{} = result;".format(store_var);
        if store_decl:
            store_decl = "{}* {};".format(store_decl, store_var);
        return (store_decl, store_stmt)

class UnaryTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  result = __builtin_epi_${intrinsic}(lhs);
  ${store_stmt}
}
"""

    def __init__(self):
        super(UnaryTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(UnaryTemplate.TEMPLATE).substitute(subs)

class UnaryMaskTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  __epi_i1 mask;
  result = __builtin_epi_${intrinsic}(lhs, mask);
  ${store_stmt}
}
"""

    def __init__(self):
        super(UnaryMaskTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(UnaryMaskTemplate.TEMPLATE).substitute(subs)

class BinaryTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  result = __builtin_epi_${intrinsic}(lhs, rhs);
  ${store_stmt}
}
"""

    def __init__(self):
        super(BinaryTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()
        subs["c_rhs_type"] = TypeRender(argument_types[1]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(BinaryTemplate.TEMPLATE).substitute(subs)

class BinaryMaskTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  __epi_i1 mask;
  result = __builtin_epi_${intrinsic}(lhs, rhs, mask);
  ${store_stmt}
}
"""

    def __init__(self):
        super(BinaryMaskTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()
        subs["c_rhs_type"] = TypeRender(argument_types[1]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(BinaryMaskTemplate.TEMPLATE).substitute(subs)

class TernaryTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  ${c_acc_type} acc;
  result = __builtin_epi_${intrinsic}(lhs, rhs, acc);
  ${store_stmt}
}
"""

    def __init__(self):
        super(TernaryTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()
        subs["c_rhs_type"] = TypeRender(argument_types[1]).render()
        subs["c_acc_type"] = TypeRender(argument_types[2]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(TernaryTemplate.TEMPLATE).substitute(subs)

class TernaryMaskTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  ${c_acc_type} acc;
  __epi_i1 mask;
  result = __builtin_epi_${intrinsic}(lhs, rhs, acc, mask);
  ${store_stmt}
}
"""

    def __init__(self):
        super(TernaryMaskTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()
        subs["c_rhs_type"] = TypeRender(argument_types[1]).render()
        subs["c_acc_type"] = TypeRender(argument_types[2]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(TernaryMaskTemplate.TEMPLATE).substitute(subs)


template_dict = {}

def EPI_FP_UNARY(string_name):
    global template_dict
    template_dict[string_name + "_f32"] = UnaryTemplate
    template_dict[string_name + "_f64"] = UnaryTemplate
    template_dict[string_name + "_f32"] = UnaryTemplate
    template_dict[string_name + "_f64"] = UnaryTemplate
    template_dict[string_name + "_f32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_f64_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_f32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_f64_mask"] = UnaryMaskTemplate

def EPI_FP_TO_INT_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_i32_f32"] = UnaryTemplate
    template_dict[string_name + "_i64_f64"] = UnaryTemplate
    template_dict[string_name + "_i32_f32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_i64_f64_mask"] = UnaryMaskTemplate

def EPI_INT_TO_FP_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_f32_i32"] = UnaryTemplate
    template_dict[string_name + "_f64_i64"] = UnaryTemplate
    template_dict[string_name + "_f32_i32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_f64_i64_mask"] = UnaryMaskTemplate

def EPI_FP_TO_FP_WIDEN_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_f64_f32"] = UnaryTemplate
    template_dict[string_name + "_f64_f32_mask"] = UnaryMaskTemplate

def EPI_FP_TO_FP_NARROW_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_f32_f64"] = UnaryTemplate
    template_dict[string_name + "_f32_f64_mask"] = UnaryMaskTemplate

def EPI_MASK_TO_SCALAR_INT_UNARY(string_name):
    global template_dict
    template_dict[string_name + "_i1"] = UnaryTemplate
    template_dict[string_name + "_i1_mask"] = UnaryMaskTemplate

def EPI_INT_TO_MASK_UNARY(string_name):
    global template_dict
    template_dict[string_name + "_i8"] = UnaryTemplate
    template_dict[string_name + "_i16"] = UnaryTemplate
    template_dict[string_name + "_i32"] = UnaryTemplate
    template_dict[string_name + "_i64"] = UnaryTemplate
    template_dict[string_name + "_i8_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_i16_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_i32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_i64_mask"] = UnaryMaskTemplate

EPI_MASK_UNARY = EPI_MASK_TO_SCALAR_INT_UNARY
EPI_INT_UNARY = EPI_INT_TO_MASK_UNARY

def EPI_INT_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_i8"] = BinaryTemplate
    template_dict[string_name + "_i16"] = BinaryTemplate
    template_dict[string_name + "_i32"] = BinaryTemplate
    template_dict[string_name + "_i64"] = BinaryTemplate
    template_dict[string_name + "_i8_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_i16_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_i32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_i64_mask"] = BinaryMaskTemplate
EPI_INT_RELATIONAL = EPI_INT_BINARY

def EPI_FP_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_f32"] = BinaryTemplate
    template_dict[string_name + "_f64"] = BinaryTemplate
    template_dict[string_name + "_f32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_f64_mask"] = BinaryMaskTemplate
EPI_FP_RELATIONAL = EPI_FP_BINARY

def EPI_MASK_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_i1"] = BinaryTemplate

def EPI_ANY_AND_INT_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_i8"] = BinaryTemplate
    template_dict[string_name + "_i16"] = BinaryTemplate
    template_dict[string_name + "_i32"] = BinaryTemplate
    template_dict[string_name + "_i64"] = BinaryTemplate

    template_dict[string_name + "_f32"] = BinaryTemplate
    template_dict[string_name + "_f64"] = BinaryTemplate

    template_dict[string_name + "_i8_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_i16_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_i32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_i64_mask"] = BinaryMaskTemplate

    template_dict[string_name + "_f32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_f64_mask"] = BinaryMaskTemplate
EPI_ANY_AND_MASK_BINARY = EPI_ANY_AND_INT_BINARY

def EPI_FP_TERNARY(string_name):
    global template_dict
    template_dict[string_name + "_f32"] = TernaryTemplate
    template_dict[string_name + "_f64"] = TernaryTemplate
    template_dict[string_name + "_f32_mask"] = TernaryMaskTemplate
    template_dict[string_name + "_f64_mask"] = TernaryMaskTemplate

EPI_INT_BINARY("vadd")
EPI_INT_BINARY("vsub")
EPI_INT_BINARY("vrsub")

EPI_INT_BINARY("vand")
EPI_INT_BINARY("vor")
EPI_INT_BINARY("vxor")

EPI_INT_BINARY("vsll")
EPI_INT_BINARY("vsrl")
EPI_INT_BINARY("vsra")

EPI_INT_RELATIONAL("vseq")
EPI_INT_RELATIONAL("vsne")
EPI_INT_RELATIONAL("vsltu")
EPI_INT_RELATIONAL("vslt")
EPI_INT_RELATIONAL("vsleu")
EPI_INT_RELATIONAL("vsle")
EPI_INT_RELATIONAL("vsgtu")
EPI_INT_RELATIONAL("vsgt")

EPI_INT_BINARY("vminu")
EPI_INT_BINARY("vmin")

EPI_INT_BINARY("vmaxu")
EPI_INT_BINARY("vmax")

EPI_INT_BINARY("vmul")
EPI_INT_BINARY("vmulh")
EPI_INT_BINARY("vmulhu")
EPI_INT_BINARY("vmulhsu")

EPI_INT_BINARY("vdivu")
EPI_INT_BINARY("vdiv")
EPI_INT_BINARY("vremu")
EPI_INT_BINARY("vrem")

EPI_INT_BINARY("vmerge")

EPI_INT_BINARY("vsaddu")
EPI_INT_BINARY("vsadd")
EPI_INT_BINARY("vssub")
EPI_INT_BINARY("vssubu")

EPI_INT_BINARY("vaadd")
EPI_INT_BINARY("vasub")

EPI_INT_BINARY("vsmul")

EPI_INT_BINARY("vssrl")
EPI_INT_BINARY("vssra")

EPI_FP_BINARY("vfadd")
EPI_FP_BINARY("vfsub")

EPI_FP_BINARY("vfmul")
EPI_FP_BINARY("vfdiv")

EPI_FP_TERNARY("vfmadd")
EPI_FP_TERNARY("vfnmadd")
EPI_FP_TERNARY("vfmsub")
EPI_FP_TERNARY("vfnmsub")
EPI_FP_TERNARY("vfmacc")
EPI_FP_TERNARY("vfnmacc")
EPI_FP_TERNARY("vfmsac")
EPI_FP_TERNARY("vfnmsac")

EPI_FP_UNARY("vfsqrt")

EPI_FP_BINARY("vfmin")
EPI_FP_BINARY("vfmax")

EPI_FP_BINARY("vfsgnj")
EPI_FP_BINARY("vfsgnjn")
EPI_FP_BINARY("vfsgnjx")

EPI_FP_RELATIONAL("vfeq")
EPI_FP_RELATIONAL("vfne")
EPI_FP_RELATIONAL("vflt")
EPI_FP_RELATIONAL("vflte")
EPI_FP_RELATIONAL("vfgt")
EPI_FP_RELATIONAL("vfgte")
EPI_FP_RELATIONAL("vford")

EPI_FP_BINARY("vfmerge")

EPI_FP_TO_INT_CONVERSION("vfcvt_xu_f")
EPI_FP_TO_INT_CONVERSION("vfcvt_x_f")
EPI_INT_TO_FP_CONVERSION("vfcvt_f_xu")
EPI_INT_TO_FP_CONVERSION("vfcvt_f_x")

EPI_FP_TO_FP_WIDEN_CONVERSION("vfwcvt_f_f")
EPI_FP_TO_FP_NARROW_CONVERSION("vfncvt_f_f")

EPI_INT_BINARY("vredsum")
EPI_INT_BINARY("vredmaxu")
EPI_INT_BINARY("vredmax")
EPI_INT_BINARY("vredmin")
EPI_INT_BINARY("vredminu")
EPI_INT_BINARY("vredand")
EPI_INT_BINARY("vredor")
EPI_INT_BINARY("vredxor")

EPI_FP_BINARY("vfredosum")
EPI_FP_BINARY("vfredsum")
EPI_FP_BINARY("vfredmax")
EPI_FP_BINARY("vfredmin")

EPI_MASK_BINARY("vmand")
EPI_MASK_BINARY("vmnand")
EPI_MASK_BINARY("vmandnot")
EPI_MASK_BINARY("vmxor")
EPI_MASK_BINARY("vmor")
EPI_MASK_BINARY("vmnor")
EPI_MASK_BINARY("vmornot")
EPI_MASK_BINARY("vmxnor")

EPI_MASK_TO_SCALAR_INT_UNARY("vmpopc")
EPI_MASK_TO_SCALAR_INT_UNARY("vmfirst")

EPI_MASK_UNARY("vmsbf")
EPI_MASK_UNARY("vmsif")
EPI_MASK_UNARY("vmsof")

EPI_INT_TO_MASK_UNARY("vmiota")

EPI_ANY_AND_INT_BINARY("vslideup")
EPI_ANY_AND_INT_BINARY("vslidedown")
EPI_ANY_AND_INT_BINARY("vslide1up")
EPI_ANY_AND_INT_BINARY("vslide1down")
EPI_ANY_AND_INT_BINARY("vrgather")
EPI_ANY_AND_MASK_BINARY("vcompress")

EPI_INT_BINARY("vdot")
EPI_INT_BINARY("vdotu")
EPI_FP_BINARY("vfdot")

EPI_INT_UNARY("vbroadcast")
EPI_FP_UNARY("vbroadcast")


def emit_test(builtin_name, prototype):
    if builtin_name not in template_dict:
        # print "Skipping {}".format(builtin_name)
        return

    template = template_dict[builtin_name]

    (return_type, argument_types) = parse_type(prototype)
    print template().render(builtin_name, return_type, argument_types)


if __name__ == "__main__":
    for (builtin_name, prototype) in preprocess_builtins():
        emit_test(builtin_name, prototype)
