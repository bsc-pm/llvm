from type_builder import *

class TypeRender:
    def __init__(self, type_builder):
        self.type_builder = type_builder

    def render(self):
        if self.type_builder.scalable_vector and not self.type_builder.vector:
            raise Exception("Scalable but not a vector")
        if self.type_builder.basic_type < 0:
            raise Exception("Missing basic type")
        if self.type_builder.signed and self.type_builder.basic_type not in [TypeBuilder.INT, TypeBuilder.CHAR]:
            raise Exception("TypeBuilder can't be signed")
        if self.type_builder.unsigned and self.type_builder.basic_type not in [TypeBuilder.INT, TypeBuilder.CHAR]:
            raise Exception("TypeBuilder can't be signed")
        if self.type_builder.short and self.type_builder.basic_type not in [TypeBuilder.INT]:
            raise Exception("TypeBuilder can't be long")
        if self.type_builder.long and self.type_builder.basic_type not in [TypeBuilder.INT, TypeBuilder.DOUBLE]:
            raise Exception("TypeBuilder can't be long")
        if self.type_builder.vector and self.type_builder.basic_type not in [TypeBuilder.INT, \
                TypeBuilder.CHAR, TypeBuilder.DOUBLE, TypeBuilder.FLOAT, \
                TypeBuilder.BOOL]:
            raise Exception("Scalable but not a vector")

        rendered = ""

        if self.type_builder.must_be_constant:
            rendered += "/* constant */ "

        if self.type_builder.scalable_vector:
            if self.type_builder.basic_type == TypeBuilder.INT:
                if self.type_builder.short:
                    rendered += "__epi_{}xi16".format(self.type_builder.vector_length)
                elif self.type_builder.long:
                    rendered += "__epi_{}xi64".format(self.type_builder.vector_length)
                else:
                    rendered += "__epi_{}xi32".format(self.type_builder.vector_length)
            elif self.type_builder.basic_type == TypeBuilder.CHAR:
                rendered += "__epi_{}xi8".format(self.type_builder.vector_length)
            elif self.type_builder.basic_type == TypeBuilder.BOOL:
                rendered += "__epi_{}xi1".format(self.type_builder.vector_length)
            elif self.type_builder.basic_type == TypeBuilder.FLOAT:
                rendered += "__epi_{}xf32".format(self.type_builder.vector_length)
            elif self.type_builder.basic_type == TypeBuilder.DOUBLE:
                rendered += "__epi_{}xf64".format(self.type_builder.vector_length)
            else:
                assert False, "Unreachable"
        else:
            if self.type_builder.basic_type in [TypeBuilder.INT, TypeBuilder.CHAR]:
                if self.type_builder.unsigned:
                    rendered += "unsigned "
                elif (self.type_builder.basic_type == TypeBuilder.CHAR and self.type_builder.signed) \
                        or self.type_builder.basic_type == TypeBuilder.INT:
                    rendered += "signed "
                else:
                    assert self.type_builder.basic_type == TypeBuilder.CHAR, "only char may be left without signedness"

                if self.type_builder.short:
                    rendered += "short "
                elif self.type_builder.long:
                    if self.type_builder.long == 1:
                        rendered += "long "
                    elif self.type_builder.long == 2:
                        rendered += "long long "
                    else:
                        assert False, "unreachable"

                if self.type_builder.basic_type == TypeBuilder.CHAR:
                    rendered += "char"
                elif self.type_builder.basic_type == TypeBuilder.INT:
                    rendered += "int"
                else:
                    assert False, "unreachable"
            elif self.type_builder.basic_type == TypeBuilder.BOOL:
                rendered += "_Bool"
            elif self.type_builder.basic_type == TypeBuilder.FLOAT:
                rendered += "float"
            elif self.type_builder.basic_type == TypeBuilder.DOUBLE:
                rendered += "double"
            elif self.type_builder.basic_type == TypeBuilder.VOID:
                rendered += "void"
            else:
                assert False, "unreachable"

        if self.type_builder.pointer:
            rendered += "* "

        return rendered

    def value_type_render(self):
        raise Exception("not implemented")

    def llvm_type_render(self):
        raise Exception("not implemented")

    def builtin_type(self):
        raise Exception("not implemented")
