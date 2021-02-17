include("./storage.jl")
include("./optimizer.jl")
include("./problems/EuclideanTSP.jl")

module Ma

using Printf
import Base
import ProgressMeter
using ..Storage

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


mutable struct Manager
    problem
    optimizer
    storage::Dict{Symbol, Any}
    debug::Bool

    function Manager()
        new(missing, missing, Dict{Symbol, Any}(), false)
    end
end


function Base.setproperty!(m::Manager, s::Symbol, value)
    if s == :problem
        filled = completeImplementation(value, getfield(m, :debug))
        setfield!(m, :problem, filled)
    elseif s == :optimizer
        setfield!(m, :optimizer, value)
    else
        setfield!(m, s)
    end
end

function allocateStorage!(m::Manager)
    m.storage[:stateStorage] = Storage.allocateCPU(m.problem.solType,
                                                   (1, m.optimizer.solutionStates, ))
    m.storage[:dataStorage] = Storage.allocateCPU(m.problem.dataType, (1, ))[1]
    if !isnothing(m.optimizer.stateType)
        m.storage[:optimStorage] = Storage.allocateCPU(m.optimizer.stateType, (1,))
    end
end

function init!(m::Manager)
    Storage.initFromFields(m.storage[:dataStorage], m.problem.data...)
    m.problem.init(m.storage[:stateStorage][1, 1])
    if !isnothing(m.optimizer.stateType) && !isnothing(m.optimizer.init)
        m.optimizer.init(m.storage[:optimStorage][1])
    end
end

function run!(m::Manager, maxIter=missing)
    if ismissing(maxIter)
        p = ProgressMeter.ProgressUnknown()
    else
        p = ProgressMeter.Progress(maxIter)
    end

    blocksIter = 500000
    iterationsDone = 0
    startTime = time()
    while true
        optimStorage = nothing

        if haskey(m.storage, :optimStorage)
            optimStorage = m.storage[:optimStorage][1]
        end

        m.optimizer.optimize(optimStorage, m.storage[:stateStorage][1],
                             m.storage[:dataStorage], blocksIter)
        iterationsDone += blocksIter
        ProgressMeter.next!(p, step=blocksIter)
        currentLoss = m.problem.loss(m.storage[:stateStorage][1][1], m.storage[:dataStorage])
        throughput = (p.counter - p.start) /  (time() - p.tfirst)
        p.desc = @sprintf("Optimizing (l=%.2f, t=%i/s) ", currentLoss, throughput)
        if iterationsDone >= maxIter
            ProgressMeter.finish!(p)
            break
        end
    end
end

export Manager

end

import .Ma
import .TSP

import .Storage
import .Optim
import Random

# Code generation

data = TSP.generateRandomTSPData(1000, 2)
manager = Ma.Manager()
manager.problem = TSP.generateProblemForData(data)
manager.optimizer = Optim.RandomLocalSearch(manager.problem)
Ma.allocateStorage!(manager)
Ma.init!(manager)
Ma.run!(manager, 100000000)

#  # Initialization
#  
#  nothing
