module TSP

import Random
import Base
using ..Storage

# Util function to generate random instances
function generateRandomTSPData(numCities, dims=2)
    Random.rand(numCities, dims)
end

# The state of a solution to the problem
struct TSPSolution{N}
    order::SizedArray{Int32, N} # I doubt people will have more than 2**31 cities
    v::Int
end

# The struct containing the problem data
# This is constant and should NEVER be changed during optimization
# TODO We should probably enforce that
struct TSPData{ND}
    coords::SizedArray{Float32, ND}
end

function generateProblemForData(problemData::AbstractArray{T, 2}) where {T}
    numCities, dims = size(problemData)

    # Make julia create a compiled version of the code for these specific problem sizes
    # Because they are passed as types and not values it tells julia to compile
    # a different version for every pair of these values
    # It makes the loss calculation stupidly fast for example
    function superFastGen(::Val{dims}, ::Val{numCities}) where {dims, numCities}

        randomInit(state) = state.order .= Random.randperm(numCities)

        # We assume proper indices !!!!
        @Base.pure @inline function distance(a, b, order, coords)
            result = 0.0
            for j in 1:dims
                @inbounds result += (coords[order[a], j] - coords[order[b], j]) ^2
            end
            result
        end

        @Base.pure @inline checkBounds(n) = mod1(n, numCities)

        @inline @Base.pure function loss(state, problemData)
            order = state.order
            coords = problemData.coords
            result = distance(1, numCities, order, coords) # The first is the link from the end to start
            for i in 2:numCities
                result += distance(i , i - 1, order, coords)
            end
            result
        end

        ### Swap2
        @inline function swap2Propose(currentState, problemData, dir)
                a, b = dir[1], dir[2]
                if b < a
                    a, b = b, a
                end
                @inline @Base.pure d(a, b) = distance(checkBounds(a), checkBounds(b),
                                                      currentState.order, problemData.coords)
                if (a == b) return 0.0 end #swapping the same cities
                return (
                        -(d(a - 1, a) + d(a, a + 1) + d(b - 1, b) + d(b, b + 1)) # old distances
                        +(d(a - 1, b) + d(b, a + 1) + d(b - 1, a) + d(a, b + 1)) # new distances
                       )
        end
        @inline function swap2Apply(currentState, problemData, dir)
            @inbounds a, b = dir[1], dir[2]
            if b < a; a, b = b, a end
            if (a == b) return end
            order = currentState.order
            @inbounds order[a], order[b] = order[b], order[a]
        end
        swap2 = (propose = swap2Propose, apply = swap2Apply)


        @inline function opt2Propose(currentState, problemData, dir)
                @inbounds a, b = dir[1], dir[2]
                if b < a; a, b = b, a end
                @inline @Base.pure d(a, b) = distance(checkBounds(a), checkBounds(b),
                                                      currentState.order, problemData.coords)
                if (a == b) || (a == 1 && b == numCities) return 0.0 end
                (d(a - 1, b) + d(a, b + 1) - d(a - 1, a) - d(b, b + 1))
        end
        @inline function opt2Apply(currentState, problemData, dir)
            @inbounds a, b = dir[1], dir[2]
            if b < a; a, b = b, a end
            if (a == b) || (a == 1 && b == numCities) return end
            order = currentState.order
            for i in 0:div((b - a + 1), 2) - 1
                @inbounds order[a + i], order[b - i] = order[b - i], order[a + i]
            end
        end

        opt2 = (propose = opt2Propose, apply = opt2Apply)

        (
         # Problem Data
         dataType=TSPData{(numCities, dims)}, # How to represent the data
         data=(problemData, ), # What is the actual data we are solving for
         # Solution
         solType=TSPSolution{numCities}, # How to represent a solution
         init=randomInit, # How to init the solution
         # Loss
         loss=loss,
         # Exploration
         neighborSpace=CartesianIndices((numCities, numCities)), # What space to explore
         opt2...
        )
    end
    superFastGen(Val(dims), Val(numCities))
end


export generateRandomTSPData, generateProblemForData

end
