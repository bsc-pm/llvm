#!/usr/bin/env python2

from preprocess_builtins import *
from builtin_parser import *
from type_render import *

import string

class NopTemplate(object):
    def render(self, intrinsic_name, return_type, argument_types):
        return ""

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
                    store_stmt = "__builtin_epi_vstore_{}xi16({}, result, gvl);".format(result_type.vector_length, store_var)
                elif result_type.long:
                    store_decl = "long";
                    store_stmt = "__builtin_epi_vstore_{}xi64({}, result, gvl);".format(result_type.vector_length, store_var)
                else:
                    store_decl = "int";
                    store_stmt = "__builtin_epi_vstore_{}xi32({}, result, gvl);".format(result_type.vector_length, store_var)
            elif result_type.basic_type == TypeBuilder.CHAR:
                store_decl = "signed char"
                store_stmt = "__builtin_epi_vstore_{}xi8({}, result, gvl);".format(result_type.vector_length, store_var)
            elif result_type.basic_type == TypeBuilder.BOOL:
                if result_type.vector_length == 1:
                    store_decl = "unsigned long"
                    store_stmt = "__builtin_epi_vstore_1xi1({}, result);".format(store_var)
                elif result_type.vector_length == 2:
                    store_decl = "unsigned int"
                    store_stmt = "__builtin_epi_vstore_2xi1({}, result);".format(store_var)
                elif result_type.vector_length == 4:
                    store_decl = "unsigned short"
                    store_stmt = "__builtin_epi_vstore_4xi1({}, result);".format(store_var)
                elif result_type.vector_length == 8:
                    store_decl = "unsigned char"
                    store_stmt = "__builtin_epi_vstore_8xi1({}, result);".format(store_var)
                else:
                    raise Exception("Unsupported vector length for masks")

            elif result_type.basic_type == TypeBuilder.FLOAT:
                store_decl = "float"
                store_stmt = "__builtin_epi_vstore_{}xf32({}, result, gvl);".format(result_type.vector_length, store_var)
            elif result_type.basic_type == TypeBuilder.DOUBLE:
                store_decl += "double"
                store_stmt = "__builtin_epi_vstore_{}xf64({}, result, gvl);".format(result_type.vector_length, store_var)
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
void test_${intrinsic}(unsigned long gvl)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  result = __builtin_epi_${intrinsic}(lhs, gvl);
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
void test_${intrinsic}(unsigned long gvl)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_mask_type} mask;
  result = __builtin_epi_${intrinsic}(lhs, mask, gvl);
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
        subs["c_mask_type"] = TypeRender(argument_types[1]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(UnaryMaskTemplate.TEMPLATE).substitute(subs)

class BinaryTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(unsigned long gvl)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  result = __builtin_epi_${intrinsic}(lhs, rhs, gvl);
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
void test_${intrinsic}(unsigned long gvl)
{
  ${c_result_type} result;
  ${c_merge_type} merge;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  ${c_mask_type} mask;
  result = __builtin_epi_${intrinsic}(merge, lhs, rhs, mask, gvl);
  ${store_stmt}
}
"""

    def __init__(self):
        super(BinaryMaskTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_merge_type"] = TypeRender(argument_types[0]).render()
        subs["c_lhs_type"] = TypeRender(argument_types[1]).render()
        subs["c_rhs_type"] = TypeRender(argument_types[2]).render()
        subs["c_mask_type"] = TypeRender(argument_types[3]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(BinaryMaskTemplate.TEMPLATE).substitute(subs)

class TernaryTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}(unsigned long gvl)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  ${c_acc_type} acc;
  result = __builtin_epi_${intrinsic}(lhs, rhs, acc, gvl);
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
void test_${intrinsic}(unsigned long gvl)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  ${c_rhs_type} rhs;
  ${c_acc_type} acc;
  ${c_mask_type} mask;
  result = __builtin_epi_${intrinsic}(lhs, rhs, acc, mask, gvl);
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
        subs["c_mask_type"] = TypeRender(argument_types[3]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(TernaryMaskTemplate.TEMPLATE).substitute(subs)

class LoadStoreTemplate(TestTemplate):
    TEMPLATE = """
void test_${load_intrinsic}_${store_intrinsic}(${c_address_type} addr, unsigned long gvl)
{
  ${c_result_type} result;
  result = __builtin_epi_${load_intrinsic}(addr, gvl);
  __builtin_epi_${store_intrinsic}(addr, result, gvl);
}
"""

    def __init__(self, load_name, store_name):
        super(LoadStoreTemplate, self).__init__()
        self.load_name = load_name
        self.store_name = store_name

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["load_intrinsic"] = self.load_name
        subs["store_intrinsic"] = self.store_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_address_type"] = TypeRender(argument_types[0]).render()

        return string.Template(LoadStoreTemplate.TEMPLATE).substitute(subs)

class LoadStoreMaskTemplate(TestTemplate):
    TEMPLATE = """
void test_${load_intrinsic}_${store_intrinsic}(${c_address_type} addr)
{
  ${c_result_type} result;
  result = __builtin_epi_${load_intrinsic}(addr);
  __builtin_epi_${store_intrinsic}(addr, result);
}
"""

    def __init__(self, load_name, store_name):
        super(LoadStoreMaskTemplate, self).__init__()
        self.load_name = load_name
        self.store_name = store_name

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["load_intrinsic"] = self.load_name
        subs["store_intrinsic"] = self.store_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_address_type"] = TypeRender(argument_types[0]).render()
        return string.Template(LoadStoreMaskTemplate.TEMPLATE).substitute(subs)

class LoadStoreStridedTemplate(TestTemplate):
    TEMPLATE = """
void test_${load_intrinsic}_${store_intrinsic}(${c_address_type} addr, signed long stride, unsigned long gvl)
{
  ${c_result_type} result;
  result = __builtin_epi_${load_intrinsic}(addr, stride, gvl);
  __builtin_epi_${store_intrinsic}(addr, result, stride, gvl);
}
"""

    def __init__(self, load_name, store_name):
        super(LoadStoreStridedTemplate, self).__init__()
        self.load_name = load_name
        self.store_name = store_name

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["load_intrinsic"] = self.load_name
        subs["store_intrinsic"] = self.store_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_address_type"] = TypeRender(argument_types[0]).render()

        return string.Template(LoadStoreStridedTemplate.TEMPLATE).substitute(subs)

class LoadStoreIndexedTemplate(TestTemplate):
    TEMPLATE = """
void test_${load_intrinsic}_${store_intrinsic}(${c_address_type} addr, unsigned long gvl)
{
  ${c_result_type} result;
  ${c_index_type} index;
  result = __builtin_epi_${load_intrinsic}(addr, index, gvl);
  __builtin_epi_${store_intrinsic}(addr, result, index, gvl);
}
"""

    def __init__(self, load_name, store_name):
        super(LoadStoreIndexedTemplate, self).__init__()
        self.load_name = load_name
        self.store_name = store_name

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["load_intrinsic"] = self.load_name
        subs["store_intrinsic"] = self.store_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_address_type"] = TypeRender(argument_types[0]).render()
        subs["c_index_type"] = TypeRender(argument_types[1]).render()

        return string.Template(LoadStoreIndexedTemplate.TEMPLATE).substitute(subs)

class ExtractTemplate(TestTemplate):
    TEMPLATE = """
${c_result_type} test_${intrinsic}_(unsigned long idx)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  result = __builtin_epi_${intrinsic}(lhs, idx);
  return result;
}
"""

    def __init__(self):
        super(ExtractTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()

        return string.Template(ExtractTemplate.TEMPLATE).substitute(subs)

class SetFirstTemplate(TestTemplate):
    TEMPLATE = """
${store_decl}
void test_${intrinsic}_(${c_element_type} value, unsigned long gvl)
{
  ${c_result_type} result;
  result = __builtin_epi_${intrinsic}(value, gvl);
  ${store_stmt}
}
"""

    def __init__(self):
        super(SetFirstTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_element_type"] = TypeRender(argument_types[1]).render()
        (subs["store_decl"], subs["store_stmt"]) = self.store_code(return_type)

        return string.Template(SetFirstTemplate.TEMPLATE).substitute(subs)

class GetFirstTemplate(TestTemplate):
    TEMPLATE = """
${c_result_type} test_${intrinsic}_(void)
{
  ${c_result_type} result;
  ${c_lhs_type} lhs;
  result = __builtin_epi_${intrinsic}(lhs);
  return result;
}
"""

    def __init__(self):
        super(GetFirstTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        subs = {}
        subs["intrinsic"] = builtin_name
        # We use the load intrinsic
        subs["c_result_type"] = TypeRender(return_type).render()
        subs["c_lhs_type"] = TypeRender(argument_types[0]).render()

        return string.Template(GetFirstTemplate.TEMPLATE).substitute(subs)

class SetVectorLengthTemplate(TestTemplate):
    TEMPLATE = """
unsigned long test_vsetvl${sew}${lmul}(unsigned long rvl)
{
    unsigned long gvl = __builtin_epi_vsetvl(rvl, ${sew}, ${lmul});
    return gvl;
}
"""

    def __init__(self):
        super(SetVectorLengthTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        result = ""
        for lmul in [1, 2, 4, 8]:
            for sew in [64, 32, 16, 8]:
                subs = {}
                subs["sew"] = "__epi_e{}".format(sew)
                subs["lmul"] = "__epi_m{}".format(lmul)
                result += string.Template(SetVectorLengthTemplate.TEMPLATE).substitute(subs)
        return result

class SetMaxVectorLengthTemplate(TestTemplate):
    TEMPLATE = """
unsigned long test_vsetvlmax${sew}${lmul}()
{
    unsigned long vlmax = __builtin_epi_vsetvlmax(${sew}, ${lmul});
    return vlmax;
}
"""

    def __init__(self):
        super(SetMaxVectorLengthTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        result = ""
        for lmul in [1, 2, 4, 8]:
            for sew in [64, 32, 16, 8]:
                subs = {}
                subs["sew"] = "__epi_e{}".format(sew)
                subs["lmul"] = "__epi_m{}".format(lmul)
                result += string.Template(SetMaxVectorLengthTemplate.TEMPLATE).substitute(subs)
        return result

class ReadVectorLengthTemplate(TestTemplate):
    TEMPLATE = """
unsigned long test_vreadvl()
{
    return __builtin_epi_vreadvl();
}
"""
    def __init__(self):
        super(ReadVectorLengthTemplate, self).__init__()

    def render(self, intrinsic_name, return_type, argument_types):
        return ReadVectorLengthTemplate.TEMPLATE

template_dict = {}

def EPI_FP_UNARY(string_name):
    global template_dict
    template_dict[string_name + "_2xf32"] = UnaryTemplate
    template_dict[string_name + "_1xf64"] = UnaryTemplate
    template_dict[string_name + "_2xf32"] = UnaryTemplate
    template_dict[string_name + "_1xf64"] = UnaryTemplate
    template_dict[string_name + "_2xf32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_1xf64_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_2xf32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_1xf64_mask"] = UnaryMaskTemplate

def EPI_FP_TO_INT_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_2xi32_2xf32"] = UnaryTemplate
    template_dict[string_name + "_1xi64_1xf64"] = UnaryTemplate
    template_dict[string_name + "_2xi32_2xf32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_1xi64_1xf64_mask"] = UnaryMaskTemplate

def EPI_INT_TO_FP_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_2xf32_2xi32"] = UnaryTemplate
    template_dict[string_name + "_1xf64_1xi64"] = UnaryTemplate
    template_dict[string_name + "_2xf32_2xi32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_1xf64_1xi64_mask"] = UnaryMaskTemplate

def EPI_FP_TO_FP_WIDEN_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_2xf64_2xf32"] = UnaryTemplate
    template_dict[string_name + "_2xf64_2xf32_mask"] = UnaryMaskTemplate

def EPI_FP_TO_FP_NARROW_CONVERSION(string_name):
    global template_dict
    template_dict[string_name + "_2xf32_2xf64"] = UnaryTemplate
    template_dict[string_name + "_2xf32_2xf64_mask"] = UnaryMaskTemplate

def EPI_MASK_TO_SCALAR_INT_UNARY(string_name):
    global template_dict
    template_dict[string_name + "_1xi1"] = UnaryTemplate
    template_dict[string_name + "_2xi1"] = UnaryTemplate
    template_dict[string_name + "_4xi1"] = UnaryTemplate
    template_dict[string_name + "_8xi1"] = UnaryTemplate
    template_dict[string_name + "_1xi1_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_2xi1_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_4xi1_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_8xi1_mask"] = UnaryMaskTemplate

def EPI_INT_TO_MASK_UNARY(string_name):
    global template_dict
    template_dict[string_name + "_8xi8"] = UnaryTemplate
    template_dict[string_name + "_4xi16"] = UnaryTemplate
    template_dict[string_name + "_2xi32"] = UnaryTemplate
    template_dict[string_name + "_1xi64"] = UnaryTemplate
    template_dict[string_name + "_8xi8_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_4xi16_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_2xi32_mask"] = UnaryMaskTemplate
    template_dict[string_name + "_1xi64_mask"] = UnaryMaskTemplate

EPI_MASK_UNARY = EPI_MASK_TO_SCALAR_INT_UNARY
EPI_INT_UNARY = EPI_INT_TO_MASK_UNARY

def EPI_INT_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_8xi8"] = BinaryTemplate
    template_dict[string_name + "_4xi16"] = BinaryTemplate
    template_dict[string_name + "_2xi32"] = BinaryTemplate
    template_dict[string_name + "_1xi64"] = BinaryTemplate
    template_dict[string_name + "_8xi8_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_4xi16_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_2xi32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_1xi64_mask"] = BinaryMaskTemplate
EPI_INT_RELATIONAL = EPI_INT_BINARY

def EPI_FP_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_2xf32"] = BinaryTemplate
    template_dict[string_name + "_1xf64"] = BinaryTemplate
    template_dict[string_name + "_2xf32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_1xf64_mask"] = BinaryMaskTemplate
EPI_FP_RELATIONAL = EPI_FP_BINARY

def EPI_MASK_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_1xi1"] = BinaryTemplate
    template_dict[string_name + "_2xi1"] = BinaryTemplate
    template_dict[string_name + "_4xi1"] = BinaryTemplate
    template_dict[string_name + "_8xi1"] = BinaryTemplate

def EPI_ANY_AND_INT_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_8xi8"] = BinaryTemplate
    template_dict[string_name + "_4xi16"] = BinaryTemplate
    template_dict[string_name + "_2xi32"] = BinaryTemplate
    template_dict[string_name + "_1xi64"] = BinaryTemplate

    template_dict[string_name + "_2xf32"] = BinaryTemplate
    template_dict[string_name + "_1xf64"] = BinaryTemplate

    template_dict[string_name + "_8xi8_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_4xi16_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_2xi32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_1xi64_mask"] = BinaryMaskTemplate

    template_dict[string_name + "_2xf32_mask"] = BinaryMaskTemplate
    template_dict[string_name + "_1xf64_mask"] = BinaryMaskTemplate

def EPI_ANY_AND_MASK_BINARY(string_name):
    global template_dict
    template_dict[string_name + "_8xi8"] = BinaryTemplate
    template_dict[string_name + "_4xi16"] = BinaryTemplate
    template_dict[string_name + "_2xi32"] = BinaryTemplate
    template_dict[string_name + "_1xi64"] = BinaryTemplate

    template_dict[string_name + "_2xf32"] = BinaryTemplate
    template_dict[string_name + "_1xf64"] = BinaryTemplate

def EPI_FP_TERNARY(string_name):
    global template_dict
    template_dict[string_name + "_2xf32"] = TernaryTemplate
    template_dict[string_name + "_1xf64"] = TernaryTemplate
    template_dict[string_name + "_2xf32_mask"] = TernaryMaskTemplate
    template_dict[string_name + "_1xf64_mask"] = TernaryMaskTemplate

def add_load_store(load_name, store_name, suffix):
    global template_dict
    # This is a bit convoluted but we need to create the template in
    # template_dict doing a call with zero arguments, so capture them using
    # this idiom
    def create_load_store(load_name, store_name):
        return lambda : LoadStoreTemplate(load_name, store_name)
    template_dict[load_name + suffix] = \
            create_load_store(load_name + suffix, store_name + suffix)
    # We're testing them inside load name
    template_dict[store_name + suffix] = NopTemplate

def EPI_LOAD_STORE_INT(load_name, store_name):
    add_load_store(load_name, store_name, "_1xi64")
    add_load_store(load_name, store_name, "_2xi32")
    add_load_store(load_name, store_name, "_4xi16")
    add_load_store(load_name, store_name, "_8xi8")

def EPI_LOAD_STORE_FP(load_name, store_name):
    add_load_store(load_name, store_name, "_2xf64")
    add_load_store(load_name, store_name, "_1xf64")
    add_load_store(load_name, store_name, "_2xf32")

def EPI_LOAD_STORE_MASK(load_name, store_name):
    def add_load_store_mask(load_name, store_name, suffix):
        global template_dict
        # This is a bit convoluted but we need to create the template in
        # template_dict doing a call with zero arguments, so capture them using
        # this idiom
        def create_load_store(load_name, store_name):
            return lambda : LoadStoreMaskTemplate(load_name, store_name)
        template_dict[load_name + suffix] = \
                create_load_store(load_name + suffix, store_name + suffix)
        # We're testing them inside load name
        template_dict[store_name + suffix] = NopTemplate
    add_load_store_mask(load_name, store_name, "_1xi1")
    add_load_store_mask(load_name, store_name, "_2xi1")
    add_load_store_mask(load_name, store_name, "_4xi1")
    add_load_store_mask(load_name, store_name, "_8xi1")

def add_load_store_strided(load_name, store_name, suffix):
    global template_dict
    # This is a bit convoluted but we need to create the template in
    # template_dict doing a call with zero arguments, so capture them using
    # this idiom
    def create_load_store(load_name, store_name):
        return lambda : LoadStoreStridedTemplate(load_name, store_name)
    template_dict[load_name + suffix] = \
            create_load_store(load_name + suffix, store_name + suffix)
    # We're testing them inside load name
    template_dict[store_name + suffix] = NopTemplate

def EPI_LOAD_STORE_STRIDED_INT(load_name, store_name):
    add_load_store_strided(load_name, store_name, "_1xi64")
    add_load_store_strided(load_name, store_name, "_2xi32")
    add_load_store_strided(load_name, store_name, "_4xi16")
    add_load_store_strided(load_name, store_name, "_8xi8")

def EPI_LOAD_STORE_STRIDED_FP(load_name, store_name):
    add_load_store_strided(load_name, store_name, "_1xf64")
    add_load_store_strided(load_name, store_name, "_2xf32")

def add_load_store_indexed(load_name, store_name, suffix):
    global template_dict
    # This is a bit convoluted but we need to create the template in
    # template_dict doing a call with zero arguments, so capture them using
    # this idiom
    def create_load_store(load_name, store_name):
        return lambda : LoadStoreIndexedTemplate(load_name, store_name)
    template_dict[load_name + suffix] = \
            create_load_store(load_name + suffix, store_name + suffix)
    # We're testing them inside load name
    template_dict[store_name + suffix] = NopTemplate

def EPI_LOAD_STORE_INDEXED_INT(load_name, store_name):
    add_load_store_indexed(load_name, store_name, "_1xi64")
    add_load_store_indexed(load_name, store_name, "_2xi32")
    add_load_store_indexed(load_name, store_name, "_4xi16")
    add_load_store_indexed(load_name, store_name, "_8xi8")

def EPI_LOAD_STORE_INDEXED_FP(load_name, store_name):
    add_load_store_indexed(load_name, store_name, "_1xf64")
    add_load_store_indexed(load_name, store_name, "_2xf32")

def EPI_EXTRACT(string_name):
    global template_dict
    template_dict[string_name + "_1xi64"] = ExtractTemplate
    template_dict[string_name + "_2xi32"] = ExtractTemplate
    template_dict[string_name + "_4xi16"] = ExtractTemplate
    template_dict[string_name + "_8xi8"] = ExtractTemplate

def EPI_SET_FIRST(string_name):
    global template_dict
    template_dict[string_name + "_1xi64"] = SetFirstTemplate
    template_dict[string_name + "_2xi32"] = SetFirstTemplate
    template_dict[string_name + "_4xi16"] = SetFirstTemplate
    template_dict[string_name + "_8xi8"] = SetFirstTemplate
    template_dict[string_name + "_1xf64"] = SetFirstTemplate
    template_dict[string_name + "_2xf32"] = SetFirstTemplate

def EPI_GET_FIRST(string_name):
    global template_dict
    template_dict[string_name + "_1xf64"] = GetFirstTemplate
    template_dict[string_name + "_2xf32"] = GetFirstTemplate

################################################################################
################################################################################
################################################################################

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
EPI_FP_RELATIONAL("vfle")
EPI_FP_RELATIONAL("vfgt")
EPI_FP_RELATIONAL("vfge")
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

EPI_LOAD_STORE_INT("vload", "vstore")
EPI_LOAD_STORE_INT("vload_unsigned", "vstore_unsigned")
EPI_LOAD_STORE_FP("vload", "vstore")

EPI_LOAD_STORE_STRIDED_INT("vload_strided", "vstore_strided")
EPI_LOAD_STORE_STRIDED_INT("vload_strided_unsigned", "vstore_strided_unsigned")
EPI_LOAD_STORE_STRIDED_FP("vload_strided", "vstore_strided")

EPI_LOAD_STORE_INDEXED_INT("vload_indexed", "vstore_indexed")
EPI_LOAD_STORE_INDEXED_INT("vload_indexed_unsigned", "vstore_indexed_unsigned")
EPI_LOAD_STORE_INDEXED_FP("vload_indexed", "vstore_indexed")

EPI_LOAD_STORE_MASK("vload", "vstore")

EPI_EXTRACT("vextract")
EPI_SET_FIRST("vsetfirst")
EPI_GET_FIRST("vgetfirst")

template_dict["vsetvl"] = SetVectorLengthTemplate
template_dict["vsetvlmax"] = SetMaxVectorLengthTemplate
template_dict["vreadvl"] = ReadVectorLengthTemplate

################################################################################
################################################################################
################################################################################

tested = []
untested = []
test_output = []

def emit_test(builtin_name, prototype):
    if builtin_name not in template_dict:
        global untested
        untested.append(builtin_name)
        return

    tested.append(builtin_name)

    template = template_dict[builtin_name]

    (return_type, argument_types) = parse_type(prototype)
    global test_output
    s = template().render(builtin_name, return_type, argument_types)
    if s:
        test_output.append(s)

if __name__ == "__main__":
    print r"""// RUN: %clang_cc1 -triple riscv64-unknown-linux-gnu -emit-llvm -O2 -o - \
// RUN:       -target-feature +m -target-feature +a -target-feature +f \
// RUN:       -target-feature +d -target-feature +c \
// RUN:       -target-feature +epi -target-abi lp64d %s \
// RUN:       | FileCheck --check-prefix=CHECK-O2 %s
"""
    for (builtin_name, prototype) in preprocess_builtins():
        emit_test(builtin_name, prototype)

    for u in untested:
        print "// Warning: '{}' not tested in this file".format(u)
    print ""
    for tpl in template_dict:
        if (tpl not in tested) and (tpl not in untested):
            print "// Warning: '{}' registered as a test but no intrinsic exists".format(tpl)

    for t in test_output:
        print t

