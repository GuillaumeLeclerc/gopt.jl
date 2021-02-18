include("./storage.jl")
include("./optimizer.jl")
include("./problems/EuclideanTSP.jl")

module Ma

using Printf
import Base
import ProgressMeter
using ..Storage
using RandomNumbers

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

function runBlock(code, maxIter, maxTime, momentum=0.9, updateInterval=1)
    if maxIter isa Number || maxTime isa Number
        p = ProgressMeter.Progress(1000)
    else
        p = ProgressMeter.ProgressUnknown()
    end

    startTime = time()
    estimatedThroughput = 1.0
    iterationsDone = 0
    lastProgress = 0
    currentProgress = 0.0
    while true
        if (maxTime isa Number) && time() - startTime > maxTime
            break
        elseif (maxIter isa Number ) && iterationsDone >= maxIter
            break
        end

        iterations = floor(Int, updateInterval * estimatedThroughput)

        if maxIter isa Number
            iterationsLeft = maxIter - iterationsDone
            iterations = min(iterations, iterationsLeft)
        end

        timeTaken = @elapsed infos = code(iterations)

        iterationsDone += iterations
        lastThroughput = iterations / timeTaken
        estimatedThroughput = (momentum * estimatedThroughput) + (1 - momentum) * lastThroughput

        ## Reporting
        progresses = Vector{Float64}()

        if maxIter isa Number
            push!(progresses, iterationsDone / maxIter)
        end

        if maxTime isa Number
            push!(progresses, (time() - startTime) / maxTime)
        end

        if length(progresses) > 0
            currentProgress = maximum(progresses) * 1000
            if currentProgress > lastProgress
                improvement = floor(Int, currentProgress - lastProgress)
                lastProgress = currentProgress
                ProgressMeter.next!(p, step=improvement)
            end
        else
            ProgressMeter.next!(p, step=iterations)
        end


        infos = Dict(pairs(infos))
        infos[:t] = @sprintf("%i/s", (iterationsDone) / (time() - startTime))

        strings = map(collect(infos)) do (key, value)
            "$key=$value"
        end
        p.desc = "Optimizing ($(join(strings, ", "))) "
    end

    ProgressMeter.finish!(p)

    return (iterationsDone, time() - startTime)
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

function run!(m::Manager, maxIter=nothing, maxTime=nothing)
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Star()

    optimStorage = nothing

    if haskey(m.storage, :optimStorage)
        optimStorage = m.storage[:optimStorage][1]
    end

    runBlock(maxIter, maxTime) do blocksIter
        m.optimizer.optimize(optimStorage, m.storage[:stateStorage][1],
                             m.storage[:dataStorage], blocksIter, rng_xor)
        return (
                loss=@sprintf("%.2f", m.problem.loss(m.storage[:stateStorage][1][1], m.storage[:dataStorage])),
               )
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
Ma.run!(manager, 1000000000000, 10)

#  # Initialization
#  
#  nothing
