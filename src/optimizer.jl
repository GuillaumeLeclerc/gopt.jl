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

struct SimulatedAnnealing 
    T::Float64
end

function SimulatedAnnealing(problem)
    T_0 ::Float64 = 10^15

    function init(state)
        state.T = T_0
    end

    function optimize(state, stateStorage, data, iterations, rng)
        solution = stateStorage[1]
        alpha = 0.5

        for i in 1:iterations
            state.T /= i   ## Scheduler 1
            #state.T = T_0 * alpha^i  ## Scheduler 2
            direction = Tuple(Random.rand(rng, problem.neighborSpace))
            delta = problem.propose(solution, data, direction)
            if delta < 0 || rand() < exp(-delta/state.T)
                problem.apply(solution, data, direction)
            end
        end
        problem.loss(stateStorage[1], data)
    end

    (
     stateType=SimulatedAnnealing, # No need to allocate memory for this optimizer
     init=init, # No need to initialize this optimizer
     solutionStates=1, # How many solution states does this solver uses
     optimize=optimize, # What is doing the optimizer to solve the problem
    )

end

struct BasinHopping 
    T::Float64
    STEPSIZE::Float64
end

function BasinHopping(problem)
    T_0 ::Float64 = 10^15
    STEP_0 ::Float64 = 10

    function init(state)
        state.T = T_0
        state.STEPSIZE = STEP_0 
    end

    function optimize(state, stateStorage, data, iterations, rng)
        solution = stateStorage[1]
        target_accept_rate = 0.05
        naccept = 0
        factor = 0.9
        take_step_every = 5*10^6
        update_stepsize_every = 5*10^6

        for i in 1:iterations
            state.T /= i  
            direction = Tuple(Random.rand(rng, problem.neighborSpace))
            delta = problem.propose(solution, data, direction)
            if delta < 0 || rand() < exp(-delta/state.T)
                naccept+=1    
                problem.apply(solution, data, direction)
            end
            if i % take_step_every == 0
                for _ in 1:round(state.STEPSIZE)
                    direction = Tuple(Random.rand(rng, problem.neighborSpace))
                    problem.apply(solution, data, direction)
                end
            end
            if i % update_stepsize_every == 0
                if naccept/i > target_accept_rate
                    state.STEPSIZE /= factor 
                else
                    state.STEPSIZE *= factor
                end
            end
        end
        problem.loss(stateStorage[1], data)
    end

    (
     stateType=BasinHopping, # No need to allocate memory for this optimizer
     init=init, # No need to initialize this optimizer
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
