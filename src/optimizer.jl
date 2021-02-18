module Optim

import Random
import ..Storage

function RandomLocalSearch(problem)

    function optimize(state, stateStorage, data, iterations, rng)
        loss = problem.loss(stateStorage[1], data)
        # stateStorage[2] = stateStorage[1]
        first = stateStorage[1]
        other = stateStorage[2]
        a = 0
        for i in 1:iterations
            direction = Tuple(Random.rand(rng, problem.neighborSpace))
            newLoss = problem.neighborLoss(loss, other, data, direction)
            if newLoss > loss
                # stateStorage[2] = stateStorage[1]
            else
                loss = newLoss
                # stateStorage[1] = stateStorage[2]
            end
        end
        problem.loss(stateStorage[1], data)
    end

    (
     stateType=nothing, # No need to allocate memory for this optimizer
     init=nothing, # No need to initialize this optimizer
     solutionStates=2, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

struct ExhaustiveLocalSearchState
    currentIteration::Int
    hasImproved::Bool
end

function ExhaustiveLocalSearch(problem)

    function init(state)
        state.currentIteration = 1
        state.hasImproved = false
    end

    function optimize(state, stateStorage, data, iterations=1000000)
        first = stateStorage[1]
        other = stateStorage[2]

        loss = problem.loss(first, data)

        Storage.copy(other, first)

        idx = state.currentIteration
        hasImproved = state.hasImproved

        for i in 1:iterations
            direction = problem.neighborSpace[idx]
            newLoss = problem.neighborLoss(loss, other, data, direction)
            if newLoss > loss
                Storage.copy(other, first)
            else
                loss = newLoss
                state.hasImproved = true
                Storage.copy(first, other)
            end

            idx += 1
            if idx > length(problem.neighborSpace)
                idx = 1
                if !hasImproved
                    # break
                end
            end
        end

        state.currentIteration = idx
        state.hasImproved = hasImproved

        r = problem.loss(stateStorage[1], data)
    end

    (
     stateType=ExhaustiveLocalSearchState, # The type of state required by this optimizer
     init=init, # How to init the state of this optimizer
     solutionStates=2, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

end
