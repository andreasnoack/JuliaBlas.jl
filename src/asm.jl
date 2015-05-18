function gen_arg(args, arg::Expr)
    if arg.head === :ref
        string(arg.args[2], "(", gen_arg(args, arg.args[1]), ")")
    elseif arg.head === :macrocall
        string(".", string(arg.args[1])[2:end])
    else
        error("?!! $arg")
    end
end
function gen_arg(args, arg::Symbol)
    idx = findfirst(args, arg)
    idx == 0 && return string("%", arg)
    "\$$(idx-1)"
end

# TODO add more of those
typ_to_llvm{T}(::Type{Ptr{T}}) = string(typ_to_llvm(T), "*")
typ_to_llvm(::Type{Float64}) = "double"
typ_to_llvm(::Type{Int32}) = "i32"
typ_to_llvm(::Type{Int64}) = "i64"
const DEBUG_ASM = false
function gen_asm(args, xs)
    io = IOBuffer()
    println(io, "call void asm \"")
    argnames = Symbol[]
    typs = []
    for a in args
        isa(a,Expr) && a.head === :(::) || error("invalid arg sig $a")
        typ = eval(a.args[2])
        push!(argnames,a.args[1])
        push!(typs,typ)
    end
    for ex in xs
        isa(ex, LineNumberNode) && continue
        isa(ex, Expr) && ex.head === :line && continue

        if isa(ex,Expr)
            if ex.head === :call
                op = string(ex.args[1])
                opargs = join(map(a -> gen_arg(argnames, a), ex.args[2:end]), ", ")
                println(io, op, " ", opargs)
            elseif ex.head === :macrocall
                println(io, ".", string(ex.args[1])[2:end], ":")
            else
                dump(ex)
                error("unknown expr $ex")
            end
        else
            error("??? $(typeof(ex))")
        end
    end
    llvmtypes = map(typ_to_llvm, typs)
    for i = 1:length(llvmtypes)
        llvmtypes[i] = string(llvmtypes[i], " %", i-1)
    end
    constr = map(_ -> "r", llvmtypes)
    println(io, "\",\"", join(constr, ","), "\"(", join(llvmtypes, ", "), ")")
    println(io, "ret void")
    seekstart(io)
    asm = readall(io)
    DEBUG_ASM && println(asm)
    Expr(:call, TopNode(:llvmcall), asm, Void, Tuple{typs...}, args...)
end

macro asm(f)
    @assert f.head === :function
    sig = f.args[1]
    @assert sig.head === :call
    body = f.args[2]
    @assert body.head === :block
    body.args = Any[gen_asm(sig.args[2:end], body.args)]
    esc(f)
end



# examples


# z[1:4] <- x[1:4] + y[1:4]
@asm function add_vec2(x::Ptr{Float64},y::Ptr{Float64},z::Ptr{Float64})
    movapd(x[0], xmm0)
    movapd(y[0], xmm1)
    addpd(xmm0, xmm1)
    movapd(xmm1, z[0])
    
    movapd(x[16], xmm0)
    movapd(y[16], xmm1)
    addpd(xmm0, xmm1)
    movapd(xmm1, z[16])
end

x = [1.0,2.0,3.0,4.0]
y = [4.0,3.0,2.0,1.0]
z = similar(x)
@show x y z
add_vec2(pointer(x),pointer(y),pointer(z))
@show z
println("===")


# z[1:2] <- x[1:2]*n (with a loop)
@asm function add_loop_vec2(x::Ptr{Float64},n::Int,z::Ptr{Float64})
    movq(n, rcx)
    movapd(x[0], xmm0)
    xorpd(xmm1,xmm1)
    @loop
    addpd(xmm0,xmm1)
    dec(rcx)
    jnz(@loop)
    movapd(xmm1, z[0])
end

x = [1.0,2.0]
n = 10
z = similar(x)
@show x n z
add_loop_vec2(pointer(x),n,pointer(z))
@show z
println("===")
