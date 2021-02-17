include("../storage.jl")
include("../optimizer.jl")
a = CartesianIndices((5, ))


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
        @Base.pure @inline @inbounds function distance(a, b, order, coords)
            result = 0.0
            for j in 1:dims
                result += (coords[order[a], j] - coords[order[b], j]) ^2
            end
            result
        end

        checkBounds(n) = mod1(n, numCities)

        @Base.pure function loss(state, problemData)
            order = state.order
            coords = problemData.coords
            result = distance(1, numCities, order, coords) # The first is the link from the end to start
            for i in 2:numCities
                result += distance(i , i - 1, order, coords)
            end
            result
        end

        function simpleNeighborhood(currentState, problemData, a, b)
            order = currentState.order
            coords = problemData.coords
            d(a, b) = distance(checkBounds(a), checkBounds(b), order, coords)

            if b < a
                a, b = b, a
            end

            # swapping the same cities
            if (a == b) return 0.0 end #swapping the same cities

            loss_diff = 0.0
            loss_diff -= d(a - 1, a) + d(a, a + 1) + d(b - 1, b) + d(b, b + 1)
            order[a], order[b] = order[b], order[a]
            loss_diff += d(a - 1, a) + d(a, a + 1) + d(b - 1, b) + d(b, b + 1)
            loss_diff
        end

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
         neighbor=simpleNeighborhood, # How to explore
         withDelta=true # Whether our neighbor function computes the delta loss
        )
    end
    superFastGen(Val(dims), Val(numCities))
end


export generateRandomTSPData, generateProblemForData

end

import .TSP

import .Storage
import .Optim
import Random

# Code generation
data = TSP.generateRandomTSPData(1000, 2)
problem = Optim.completeImplementation(TSP.generateProblemForData(data), false)
optimizer = Optim.ExhaustiveLocalSearch(problem)
# println(keys(optimizer))

#Allocation
stateStorage = Storage.allocateCPU(problem.solType, (optimizer.solutionStates, ))
dataStorage = Storage.allocateCPU(problem.dataType, (1, ))[1]
optimStorage = Storage.allocateCPU(optimizer.stateType, (1,))

# Initialization
Storage.initFromFields(dataStorage, problem.data...)
problem.init(stateStorage[1])
optimizer.init(optimStorage[1])

nothing
