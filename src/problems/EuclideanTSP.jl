include("../storage.jl")


module TSP

import Random
using ..Storage

# Util function to generate random instances
function generateRandomTSPData(numCities, dims=2)
    Random.rand(numCities, dims)
end

# The state of a solution to the problem
struct TSPSolution{N}
    order::SizedArray{Int32, N} # I doubt people will have more than 2**31 cities
end

function generateProblemForData(problemData::AbstractArray{T, 2}) where {T}
    numCities, dims = size(problemData)

    randomInit(state) = state.order .= Random.randperm(numCities)

    # We assume proper indices !!!!
    @inline @inbounds function distance(a, b, order, problemData)
        result = 0.0
        a = order[a]
        b = order[b]
        for j in 1:dims
            result += (problemData[a, j] - problemData[b, j]) ^2
        end
        return result
    end

    function loss(state, problemData)
        order = state.order
        # The first is the link from the end to start
        result = distance(1, numCities, order, problemData)
        for i in 2:numCities
            result += distance(i , i - 1, order, problemData)
        end
        result
    end

    (
     solType=TSPSolution{numCities},
     loss=loss,
     init=randomInit
    )
end


export generateRandomTSPData, generateProblemForData

end

import .TSP

import .Storage

data = TSP.generateRandomTSPData(1000, 2)
problem = TSP.generateProblemForData(data)
stateStorage = Storage.allocateCPU(problem.solType, (1, ))
problem.init(stateStorage[1])
loss = problem.loss(stateStorage[1], data)
