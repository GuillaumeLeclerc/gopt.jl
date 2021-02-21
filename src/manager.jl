#Framework
include("./storage.jl")
include("./optimizer.jl")
include("./shuffler.jl")

#Problem specific
include("./problems/EuclideanTSP.jl")

module Ma

using Printf
import Base
import ProgressMeter
using ..Storage
using RandomNumbers


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

        iterations = max(1, floor(Int, updateInterval * estimatedThroughput))

        if maxIter isa Number
            iterationsLeft = maxIter - iterationsDone
            iterations = min(iterations, iterationsLeft)
        end


        before = time()
        infos = code(iterations)
        timeTaken = time() - before


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
    println()

    return (iterationsDone, time() - startTime)
end


mutable struct Manager
    problem
    optimizer
    shuffler
    storage::Dict{Symbol, Any}
    debug::Bool

    function Manager()
        new(missing, missing, missing, Dict{Symbol, Any}(), false)
    end
end


function allocateStorage!(m::Manager)
    m.storage[:stateStorage] = Storage.allocateCPU(m.problem.solType,
                                                   (m.shuffler.popSize,
                                                    m.optimizer.solutionStates, ))
    m.storage[:dataStorage] = Storage.allocateCPU(m.problem.dataType, (1, ))[1]
    if !isnothing(m.optimizer.stateType)
        m.storage[:optimStorage] = Storage.allocateCPU(m.optimizer.stateType,
                                                       (m.optimizer.solutionStates,))
    end
end

function init!(m::Manager)
    Storage.initFromFields(m.storage[:dataStorage], m.problem.data...)

    # We only init the best state,
    for i in 1:m.shuffler.popSize
        m.problem.init(m.storage[:stateStorage][i, 1], m.storage[:dataStorage])
        if !isnothing(m.optimizer.stateType) && !isnothing(m.optimizer.init)
            m.optimizer.init(m.storage[:optimStorage][i])
        end
    end

end

function run!(m::Manager, maxIter=nothing, maxTime=nothing)

    optimStorage = nothing


    runBlock(maxIter, maxTime) do blocksIter
        losses = zeros(Float32, m.shuffler.popSize)
        Base.Threads.@threads for i in 1:m.shuffler.popSize
            rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Star()
            if haskey(m.storage, :optimStorage)
                optimStorage = m.storage[:optimStorage][i]
            end
            state = m.storage[:stateStorage][i]
            m.optimizer.optimize(optimStorage, state,
                                 m.storage[:dataStorage], blocksIter, rng_xor)
            loss = m.problem.loss(state[1], m.storage[:dataStorage])
            losses[i] = loss
        end
        return (
                loss=@sprintf("%.3f - %3f", minimum(losses), maximum(losses)),
               )
    end
end

function bestSolution(m::Manager)
    losses = zeros(Float32, m.shuffler.popSize)
    Base.Threads.@threads for i in 1:m.shuffler.popSize
        state = m.storage[:stateStorage][i]
        loss = m.problem.loss(state[1], m.storage[:dataStorage])
        losses[i] = loss
    end
    bestIx = argmin(losses)
    return (losses[bestIx], m.storage[:stateStorage][bestIx][1])
end

export Manager

end


import .Ma
import .TSP

import .Storage
import .Optim
import .Shufflers

# Code generation

# data = TSP.readTourFile("./pla85900.tsp")
data = TSP.generateRandomTSPData(10000)
manager = Ma.Manager()
manager.problem = TSP.generateProblemForData(data)
manager.optimizer = Optim.RandomLocalSearch(manager.problem)
manager.shuffler = Shufflers.Independent(16)
Ma.allocateStorage!(manager)
Ma.init!(manager)
Ma.run!(manager, 10000000000000, 6)
loss, solution = Ma.bestSolution(manager)
print("LOSS: ", loss)
