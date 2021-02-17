module Storage

import Base
using ConstructionBase

using Setfield
using Strided

# To be able to coalesce the memory read on GPU
# We need to align all reads in blocks of WarpSize
#
# On GPU we will make sure that the first dimension of
# the storage is a multiple of 32 to have everything align
# nicely
const MIN_GPU_THREADS = 32

# Type stricly used for user to declare their data structure
abstract type SizedArray{T, L} end

getStrides(t) = cumprod(t)

# Extrat the information of a single field in the user structure
function getFieldInfo(T, s::Symbol)
    tpe = fieldtype(T, s)
    shape = (1, )
    primitive = true
    if !isprimitivetype(tpe)
        tpe, shape = tpe.parameters
        if !(shape isa Tuple)
            shape = (shape, )
        end
        primitive = false
    end
    (size=sizeof(tpe), shape=shape, primitive=primitive)
end

function alignAddress(addr::Int, target::Int)
    if mod(addr, target) != 0 # The offset is not aligned!!!!
        # We add the minimum offset so that the new address is aligned
        addr += target - mod(addr, target)
    end
    @assert mod(addr, target) == 0 # It should be aligned now
    addr
end

# Perform reflection on the type T to extract all the information
# We need to determine the data layout later
function getTypeInfo(T)
    fnames = fieldnames(T)

    # We assume here that the base address will be aligned on 256 bits
    # This is the case for cuMalloc but I don't know what Julia will give
    # us on the CPU
    previousOffset = 0

    infos = map(x -> getFieldInfo(T, x), fnames)
    fullInfos = []
    for info in infos
        # We want to make sure that data alignment is correct
        # I really hope I'm not screwing this up since this
        # will fail silently on the GPU
        # We want to make sure that the address is aligned
        aligned = alignAddress(previousOffset, info.size)
        push!(fullInfos, merge((offset=aligned,), info))
        previousOffset = aligned += info.size * prod(info.shape)
    end
    return (typesInfo=NamedTuple{fnames}(fullInfos),
            totalSize=alignAddress(previousOffset, 32))
end

totalSize(T) = sum(x.size * prod(x.shape) for x in getTypeInfo(T))

# T: Type (The struct)
# I: Parsed info about the type (performance optimization)
# D: Number of dimensions
# L: Layout (:cpu or :gpu)
# O: Type of the the reference
struct StoreElement{T, I, D, L, O}
    _target::O # Pointer to the Store this Element belongs to
    _cursor::NTuple{D, Int} # The index of the element selected
end

# T: Type (The struct)
# I: Parsed info about the type (performance optimization)
# D: Number of dimensions
# L: Layout (:cpu or :gpu)
# O: Type of the the reference
# F: The number of filled dimensions
struct StorePartialElement{T, I, D, L, O, F}
    _target::O # Pointer to the Store this Element belongs to
    _cursor::NTuple{F, Int} # The index of the element selected
end

# T: Type (The struct)
# I: Parsed info about the type (performance optimization)
# D: Number of dimensions
# L: Layout (:cpu or :gpu)
struct Store{T, I, D, L} <: AbstractArray{StoreElement, D}
    size::NTuple{D, Int}
    storage::AbstractArray{UInt8, 1}
    fastStorage::Ptr{UInt8}
end

# Create a view on the particular element that we are looking at
function Base.getindex(store::Store{T, I, D, L}, II...) where {T, I, D, L}
    n = II[begin:D]
    StoreElement{T, I, D, L, Store{T, I, D, L}}(store, II[begin:D])
end

# Create a view on the particular element that we are looking at
function Base.getindex(store::Store{T, I, D, L}, i::Int) where {T, I, D, L}
    newCursor = (i, )
    if length(newCursor) == D
        return StoreElement{T, I, D, L, Store{T, I, D, L}}(store, newCursor)
    else
        return StorePartialElement{T, I, D, L, Store{T, I, D, L}, length(newCursor)}(store, newCursor)
    end
end

# Create a view on the particular element that we are looking at
function Base.getindex(partialView::StorePartialElement{T, I, D, L, Store{T, I, D, L}, F}, i::Int) where {T, I, D, L, F}
    newCursor = ((getfield(partialView, :_cursor))..., i)
    store = getfield(partialView, :_target)
    if length(newCursor) == D
        return StoreElement{T, I, D, L, Store{T, I, D, L}}(store, newCursor)
    else
        return StorePartialElement{T, I, D, L, Store{T, I, D, L}, length(newCursor)}(store, newCursor)
    end
end

# Create a view on the particular element that we are looking at
function Base.setindex!(partialView::StorePartialElement{T, I, D, L, Store{T, I, D, L}, F},
                        source::StoreElement{T, I, D, L, Store{T, I, D, L}},
                        i::Int) where {T, I, D, L, F}
    dest = partialView[i]
    copy(dest, source)
end


function Base.show(io::IO, store::Store{T, I, D, L}) where {T, I, D, L}
    print(io, "Store{$T}($(store.size))")
end


# Copy a state from another
@generated function copy(dest::StoreElement{T, I, D, L, Store{T, I, D, L}},
                source::StoreElement{T, I, D, L, Store{T, I, D, L}}) where {T, I, D, L}
    ks = I.typesInfo
    instructions = []
    for f in fieldnames(T)
        push!(instructions, :(setproperty!(dest, $(QuoteNode(f)), getproperty(source, $(QuoteNode(f))))))
    end
    result = Expr(:block, instructions..., :dest)
    return result
end

# Copy a state from another
@generated function initFromFields(dest::StoreElement{T, I, D, L, Store{T, I, D, L}}, data...) where {T, I, D, L}
    ks = I.typesInfo
    instructions = []
    for (i, f) in enumerate(fieldnames(T))
        push!(instructions, :(setproperty!(dest, $(QuoteNode(f)), data[$(i)])))
    end
    result = Expr(:block, instructions..., :dest)
    return result
end

# Create a view on the particular element that we are looking at
function Base.setindex!(store::Store{T, I, D, L}, source::StoreElement{T, I, D, L, Store{T, I, D, L}}, II...) where {T, I, D, L}
    dest = StoreElement{T, I, D, L, Store{T, I, D, L}}(store, II)
    copy(dest, source)
end

# Implement the size method of the interface AbstractArray
function Base.size(store::Store{T, I, D, L}) where {T, I, D, L}
    store.size
end

# Pretty print a StoreElement (also avoids infinite
# loop in the REPL when printing one
function Base.show(io::IO, lookup::StoreElement{T, I, D, L, O}) where {T, I, D, L, O}
    target = getfield(lookup, :_target)
    cursor = getfield(lookup, :_cursor)
    print(io, "StoreElement{$T, $L}($(target.size), $(cursor))")
end

@inline function fastGetType(T, s::Symbol)
    tpe = fieldtype(T, s)
    if isprimitivetype(tpe)
        tpe
    else
        fieldtype(T, s).parameters[1]
    end
end

# This one is pretty tricky...
@inbounds @inline function getUnderlyingStorage(lookup::StoreElement{T, I, D, L, O}, s::Symbol) where {T, I, D, L, O}
    tpe=fastGetType(T, s)
    storeStrides = I.storeStrides
    target = getfield(lookup, :_target)
    cursor = getfield(lookup, :_cursor)
    typeInfo = I.typesInfo[s]

    # Get the offset in words of size the underlying type
    fullOffset = div(sum(storeStrides .* (cursor .- 1)) + typeInfo.offset, typeInfo.size)

    correctPtr = Base.unsafe_convert(Ptr{tpe}, target.fastStorage)
    Strided.UnsafeStridedView(correctPtr,
                              typeInfo.shape,
                              typeInfo.cpuStrides,
                              fullOffset)
end

@inbounds @Base.pure function Base.getproperty(lookup::StoreElement{T, I, D, L, O}, s::Symbol) where {T, I, D, L, O}
    if hasfield(T, s)
        view = getUnderlyingStorage(lookup, s)
        typeInfo = I.typesInfo[s]
        if typeInfo.primitive
            return view[1]
        end
        view
    else
        getfield(lookup, s)
    end
end

@inline function Base.setproperty!(lookup::StoreElement{T, I, D, L, O}, s::Symbol, value) where {T, I, D, L, O}
    if hasfield(T, s)
        view = getUnderlyingStorage(lookup, s)
        typeInfo = I.typesInfo[s]
        if typeInfo.primitive
            view[1] = value
        else
            view .= value
        end
    else
        setfield!(lookup, s, v)
    end
end

function appendStrides(storeShape, I, T)
    storeStrides = getStrides((I.totalSize, storeShape...))[1:end .!= end]
    typesInfo = I.typesInfo
    newTypesInfo = []
    for tname in keys(typesInfo)
        tinfo = typesInfo[tname]
        p = cumprod((1, tinfo.shape...))[1:end .!= end]
        push!(newTypesInfo, merge(tinfo, (cpuStrides=p,)))
    end
    (
     totalSize=I.totalSize,
     typesInfo=NamedTuple{keys(typesInfo)}(newTypesInfo),
     storeStrides=storeStrides
    )
end

# Allocate storage on the CPU for the given type and size
function allocateCPU(T, size::NTuple{D, Int}) where {D}
    I = getTypeInfo(T)
    memory = zeros(UInt8, I.totalSize * prod(size))
    memoryPointer = Base.unsafe_convert(Ptr{UInt8}, memory)
    I = appendStrides(size, I, T)
    Store{T, I, D, :cpu}(size, memory, memoryPointer)
end

export SizedArray, Store, allocateCPU
end
