class TypeBuilder:
    VOID = 0
    BOOL = 1
    CHAR = 2
    INT = 3
    FLOAT = 4
    DOUBLE = 5

    MIN_TYPE = VOID
    MAX_TYPE = DOUBLE

    def __init__(self):
        self.short = False
        self.long = 0
        self.unsigned = False
        self.signed = False
        self.basic_type = -1
        self.scalable_vector = False
        self.vector = False
        self.vector_length = -1
        self.pointer = False
        self.must_be_constant = False

    def set_short(self):
        if self.long:
            raise Exception("TypeBuilder is already long")
        if self.short:
            raise Exception("TypeBuilder is already short")
        self.short = True

    def set_long(self):
        if self.short:
            raise Exception("TypeBuilder is already short")
        if self.long == 2:
            raise Exception("TypeBuilder would be too long")
        self.long = self.long + 1

    def set_unsigned(self):
        if self.signed:
            raise Exception("TypeBuilder is already signed")
        self.unsigned = True

    def set_signed(self):
        if self.unsigned:
            raise Exception("TypeBuilder is already unsigned")
        self.signed = True

    def set_scalable_vector(self):
        if self.scalable_vector:
            raise Exception("Already a scalable vector")
        self.scalable_vector = True

    def set_vector(self):
        if self.vector:
            raise Exception("Already a vector")
        self.vector = True

    def set_vector_length(self, length):
        if self.vector_length >= 0:
            raise Exception("Already have vector length")
        if length < 0:
            raise Exception("Invalid length")
        self.vector_length = length

    def set_basic_type(self, basic_type):
        if basic_type < TypeBuilder.MIN_TYPE or basic_type > TypeBuilder.MAX_TYPE:
            raise Exception("Invalid basic type")
        if self.basic_type >= 0 and self.basic_type != basic_type:
            raise Exception("Overwriting basic type")
        self.basic_type = basic_type

    def has_basic_type(self):
        return self.basic_type >= 0

    def set_pointer(self):
        if self.pointer:
            raise Exception("Only one pointer supported")
        self.pointer = True

    def set_must_be_constant(self):
        if self.must_be_constant:
            raise Exception("Already constant")
        self.must_be_constant = True

