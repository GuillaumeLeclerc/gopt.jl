module Optim

import Random
import ..Storage

function completeImplementation(problem, debug=false)
    problemDict = Dict(pairs(problem))

    if hasproperty(problem, :withDelta) && problem.withDelta

        @inline function neighborLoss(previousLoss, currentState, problemData, d...)
            delta = problem.neighbor(currentState, problemData, d...)
            newLoss = previousLoss + delta
            return newLoss
        end

        function checkedNeighborLoss(previousLoss, currentState, problemData, d...)
            result = neighborLoss(previousLoss, currentState, problemData, d...)
            @assert isapprox(result, problem.loss(currentState, problemData))
            return result
        end

        if debug
            problemDict[:neighborLoss] = checkedNeighborLoss
        else
            problemDict[:neighborLoss] = neighborLoss
        end
    else
        function slowNeighborLoss(previousLoss, currentState, problemData, d...)
            problem.neighbor(currentState, problemData, d...)
            return problem.loss(currentState, problemData)
        end

        problemDict[:neighborLoss] = slowNeighborLoss
    end

    (;problemDict...)
end

function RandomLocalSearch(;problem...)

    function optimize(state, stateStorage, problem, data, iterations=1000000)
        loss = problem.loss(stateStorage[1], data)
        stateStorage[2] = stateStorage[1]
        first = stateStorage[1]
        other = stateStorage[2]
        a = 0
        for i in 1:iterations
            direction = Tuple(Random.rand(problem.neighborSpace))
            newLoss = problem.neighborLoss(loss, other, data, direction...)
            if newLoss > loss
                stateStorage[2] = stateStorage[1]
            else
                loss = newLoss
                stateStorage[1] = stateStorage[2]
            end
        end
        problem.loss(stateStorage[1], data)
    end

    (
     state=nothing, # The type of state required by this optimizer
     stateInit=nothing, # How to init the state of this optimizer
     solutionStates=2, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

struct ExhaustiveLocalSearchState
    currentIteration::Int
    hasImproved::Bool
end

function ExhaustiveLocalSearch(;problemDesc...)
    function init(state, stateStorage)
        state.currentIteration = 1
        state.hasImproved = false
    end

    println(keys(problemDesc))

    function optimize(state, stateStorage, data, iterations=1000000)
        println("Hello 1")
        first = stateStorage[1]
        println("Hello 2")
        other = stateStorage[2]
        println("Hello 3")
        println(typeof(problemDesc[1]))

        loss = problemDesc.loss(first, data)
        println("Hello 4")
        println("Start ", loss)


        Storage.copy(other, first)

        idx = state.currentIteration
        hasImproved = state.hasImproved

        for i in 1:iterations
            direction = problemDesc.neighborSpace[idx]
            newLoss = problemDesc.neighborLoss(loss, other, data, direction...)
            if newLoss > loss
                Storage.copy(other, first)
            else
                loss = newLoss
                state.hasImproved = true
                Storage.copy(first, other)
            end

            idx += 1
            if idx > length(problemDesc.neighborSpace)
                idx = 1
            end
        end

        state.currentIteration = idx
        state.hasImproved = hasImproved

        r = problemDesc.loss(stateStorage[1], data)
        println("end ", r)
    end

    (
     stateType=ExhaustiveLocalSearchState, # The type of state required by this optimizer
     stateInit=init, # How to init the state of this optimizer
     solutionStates=2, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

end
