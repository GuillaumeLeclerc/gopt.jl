module Optim

import Random
import ..Storage

function RandomLocalSearch(problem)

    function optimize(state, stateStorage, data, iterations, rng)
        solution = stateStorage[1]
        a = 0
        for i in 1:iterations
            direction = Tuple(Random.rand(rng, problem.neighborSpace))
            delta = problem.propose(solution, data, direction)
            if delta < 0
                problem.apply(solution, data, direction)
            end
        end
        problem.loss(stateStorage[1], data)
    end

    (
     stateType=nothing, # No need to allocate memory for this optimizer
     init=nothing, # No need to initialize this optimizer
     solutionStates=1, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

struct ExhaustiveLocalSearchState
    currentIteration::Int
    hasImproved::Bool
end

function ExhaustiveLocalSearch(problem)

    function init(state)
        state.currentIteration = 17
        state.hasImproved = false
    end

    function optimize(state, stateStorage, data, iterations, rng)
        solution = stateStorage[1]
        idx = state.currentIteration
        hasImproved = state.hasImproved

        for i in 1:iterations
            direction = problem.neighborSpace[idx]
            delta = problem.propose(solution, data, direction)
            if delta < 0
                state.hasImproved = true
                problem.apply(solution, data, direction)
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
     solutionStates=1, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

end
