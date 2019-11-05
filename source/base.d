module inmet.base;

import core.stdc.stdint;
import std.complex;
import std.string: format;

alias idx = int;

alias cpx64 = Complex!float;
alias cpx128 = Complex!double;

enum DType : int {
    Unknown = 0,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Complex64, Complex128
}

enum Order: int {
    RowMajor, ColMajor
}

Order fromStr(T)(in T str) @safe pure
{
    switch(str) {
        case "RowMajor":
            return Order.RowMajor;
        case "C":
            return Order.RowMajor;
        case "ColMajor":
            return Order.ColMajor;
        case "Fortran":
            return Order.ColMajor;
        default:
            throw new Exception(format!"Unknown order type '%s'!"(str));
    }
}

struct TypeInfo {
    bool is_complex;
    int size;
    DType id;
    string name;
    
    this(bool is_cpx, int size, DType id, string name) @safe @nogc pure nothrow
    {
        this.is_complex = is_cpx;
        this.size = size;
        this.id = id;
        this.name = name;
    }
    
    bool opEquals(in ref const(TypeInfo) other) immutable
    @safe @nogc pure nothrow
    {
        return this.id == other.id;
    }
    
    
    bool opEquals(in ref const(TypeInfo) other) const
    @safe @nogc pure nothrow
    {
        return this.id == other.id;
    }
    
    static immutable(TypeInfo*) get(T)(in T str) @safe pure
    {
        switch(str) {
            case "Unknown":
                throw new Exception("Unknown type");
            
            case "Int8":
                return &Types[DType.Int8];
            case "Int16":
                return &Types[DType.Int16];
            case "Int32":
                return &Types[DType.Int32];
            case "Int64":
                return &Types[DType.Int64];
            
            case "UInt8":
                return &Types[DType.UInt8];
            case "UInt16":
                return &Types[DType.UInt16];
            case "UInt32":
                return &Types[DType.UInt32];
            case "UInt64":
                return &Types[DType.UInt64];
            
            case "Float32":
                return &Types[DType.Float32];
            case "Float64":
                return &Types[DType.Float64];
            
            case "Complex64":
                return &Types[DType.Complex64];
            case "Complex128":
                return &Types[DType.Complex128];
            
            default:
                throw new Exception("Unknown type");
        }
    }
} 


static immutable(TypeInfo[DType.max + 1]) Types = [
    DType.Unknown: TypeInfo(false, 0, DType.Unknown, "Unknown"),
    
    DType.Int8:  TypeInfo(false, int8_t.sizeof, DType.Int8, "Int8"),
    DType.Int16: TypeInfo(false, int16_t.sizeof, DType.Int16, "Int16"),
    DType.Int32: TypeInfo(false, int32_t.sizeof, DType.Int32, "Int32"),
    DType.Int64: TypeInfo(false, int64_t.sizeof, DType.Int64, "Int64"),
    
    DType.UInt8:  TypeInfo(false, uint8_t.sizeof, DType.UInt8, "UInt8"),
    DType.UInt16: TypeInfo(false, uint16_t.sizeof, DType.UInt16, "UInt16"),
    DType.UInt32: TypeInfo(false, uint32_t.sizeof, DType.UInt32, "UInt32"),
    DType.UInt64: TypeInfo(false, uint64_t.sizeof, DType.UInt64, "UInt64"),
    
    DType.Float32: TypeInfo(false, float.sizeof, DType.Float32, "Float32"),
    DType.Float64: TypeInfo(false, double.sizeof, DType.Float64, "Float64"),
    
    DType.Complex64:  TypeInfo(false, cpx64.sizeof, DType.Complex64, "Complex64"),
    DType.Complex128: TypeInfo(false, cpx128.sizeof, DType.Complex128, "Complex128"),
];


