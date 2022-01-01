module Environments

using CSV
using DataFrames
using Random
using Setfield
using Distributions
using Plots

export HexEnv, visualize, setTiles, setTerminalStates, reset, matchRow, step, getNextAction, STEP_COST, visualizeEpsLabel, OUT_OF_BOUNDS_PENALTY, visualizeOptimalPolicy

STEP_COST = -1
OUT_OF_BOUNDS_PENALTY = -1
MOVE_PROB = 0.7
HEX_ACTIONS_NUM = 6
ANNOTATION_FONT_SIZE = 7
MARKER_SIZE = 30
FIGURE_DPI = 300
FIGURE_LOW_DPI = 100
ACTIONS = [-1 0; 0 1; 1 1; 1 0; 1 -1; 0 -1]

fig = nothing

struct HexEnv
    tiles::Matrix{Int64}
    currentState::Vector{Int64}
    vizTiles::Matrix{Float64}
    HexEnv() = new()
    HexEnv(tiles::Matrix{Int64}, currentState::Vector{Int64}, vizTiles::Matrix{Float64}) = new(tiles, currentState, vizTiles)
end;

function HexEnv(tiles::Matrix{Int64}, currentState::Vector{Int64})
    vizTiles = convert(Matrix{Float64}, tiles)
    for row in eachrow(vizTiles)
        if (row[2] % 2 == 1)
            row[1] = row[1] - 0.5
        end
        row[2] = row[2] + 0.5
    end
    return HexEnv(tiles, currentState, vizTiles)
end

function HexEnv(tiles::Matrix)
    return HexEnv(tiles, Int64[])
end

# visualize the environment with paths
function visualize(env::HexEnv, path::Any = nothing, directions::Any = nothing, directionAlpha::Any = nothing; 
                    show::Bool = true, lowDpi::Bool = false, optimalPolicy::Bool = false, filename = nothing)

    width = findmax(env.vizTiles[:, 1])[1] + 0.5
    height = findmax(env.vizTiles[:, 2])[1] + 1
    
    if (lowDpi)
        fDpi = FIGURE_LOW_DPI
    else
        fDpi = FIGURE_DPI
    end

    negTmn = getNegRewardTerminals(env, path);
    posTmn = getPosRewardTerminals(env, path);
    normalTiles = getNormalTiles(env, path);

    global fig

    if (path !== nothing)
        if optimalPolicy
            fig = scatter(path[:, 1], path[:, 2], m = (0, :h, MARKER_SIZE), bg = :gray98, color = :dodgerblue, ticks=false, 
                showaxis=false, xlims=(0, width), aspect_ratio = :equal, lab = nothing, dpi = fDpi, size = (width*60, height*60),
                markeralpha = directionAlpha);
            annotate!(fig, path[:, 1], path[:, 2], directions, :royalblue4);

            scatter!(fig, posTmn[:, 1], posTmn[:, 2], color = :darkseagreen1, m = (0, :h, MARKER_SIZE), lab = nothing);
            scatter!(fig, negTmn[:, 1], negTmn[:, 2], color = :lightsalmon, m = (0, :h, MARKER_SIZE), lab = nothing);

            annotate!(fig, posTmn[:, 1], posTmn[:, 2], [text(ann, :mediumseagreen, ANNOTATION_FONT_SIZE) for ann in posTmn[:, 3]]);
            annotate!(fig, negTmn[:, 1], negTmn[:, 2], [text(ann, :snow, ANNOTATION_FONT_SIZE) for ann in negTmn[:, 3]]);
        else
            fig = scatter(path[:, 1], path[:, 2], m = (0, :h, MARKER_SIZE), bg = :gray98, color = :wheat1, ticks=false, 
                showaxis=false, xlims=(0, width), aspect_ratio = :equal, lab = nothing, dpi = fDpi, size = (width*60, height*60));
            annotate!(fig, path[:, 1], path[:, 2], directions, :wheat3);
            scatter!(fig, posTmn[:, 1], posTmn[:, 2], color = :darkseagreen1, m = (0, :h, MARKER_SIZE), lab = nothing);
            scatter!(fig, negTmn[:, 1], negTmn[:, 2], color = :lightsalmon, m = (0, :h, MARKER_SIZE), lab = nothing);
            scatter!(fig, normalTiles[:, 1], normalTiles[:, 2], color = :azure, m = (0, :h, MARKER_SIZE), lab = nothing);
            
            annotate!(fig, normalTiles[:, 1], normalTiles[:, 2], [text(ann, :turquoise3, ANNOTATION_FONT_SIZE) for ann in normalTiles[:, 3]]);
            annotate!(fig, posTmn[:, 1], posTmn[:, 2], [text(ann, :mediumseagreen, ANNOTATION_FONT_SIZE) for ann in posTmn[:, 3]]);
            annotate!(fig, negTmn[:, 1], negTmn[:, 2], [text(ann, :snow, ANNOTATION_FONT_SIZE) for ann in negTmn[:, 3]]);
        end
        
        
    else
        fig = scatter(negTmn[:, 1], negTmn[:, 2], m = (0, :h, MARKER_SIZE), color = :lightsalmon, bg = :gray98, ticks=false, 
            showaxis=false, xlims=(0, width), aspect_ratio = :equal, lab = nothing, dpi = fDpi, size = (width*60, height*60));

        scatter!(fig, posTmn[:, 1], posTmn[:, 2], color = :darkseagreen1, m = (0, :h, MARKER_SIZE), lab = nothing);
        scatter!(fig, normalTiles[:, 1], normalTiles[:, 2], color = :azure, m = (0, :h, MARKER_SIZE), lab = nothing);
        annotate!(fig, normalTiles[:, 1], normalTiles[:, 2], [text(ann, :turquoise3, ANNOTATION_FONT_SIZE) for ann in normalTiles[:, 3]]);
        annotate!(fig, posTmn[:, 1], posTmn[:, 2], [text(ann, :mediumseagreen, ANNOTATION_FONT_SIZE) for ann in posTmn[:, 3]]);
        annotate!(fig, negTmn[:, 1], negTmn[:, 2], [text(ann, :snow, ANNOTATION_FONT_SIZE) for ann in negTmn[:, 3]]);
    end

    if show
        display(fig);
    end

    if (filename !== nothing)
        print("\nSaving current figure");
        savefig(fig, filename);
    end
end;

function getNormalTiles(env::HexEnv, path::Any = nothing)
    n = size(env.vizTiles)[1]

    if (path !== nothing)
        vizTiles = Matrix{Float64}(undef, 0, 4)
        for i in 1:n
            if (env.vizTiles[i, 4] == 0 && matchRow(path, env.vizTiles[i, 1:2]) == 0)
                vizTiles = vcat(vizTiles, env.vizTiles[i, :]')
            end
        end
    else
        vizTiles = env.vizTiles[env.vizTiles[:, 4] .== 0, :]
    end
    
    return vizTiles
end

function getNegRewardTerminals(env::HexEnv, path::Any = nothing)

    n = size(env.vizTiles)[1]

    if (path !== nothing)
        vizTiles = Matrix{Float64}(undef, 0, 4)
        for i in 1:n
            if (env.vizTiles[i, 4] == 1 && matchRow(path, env.vizTiles[i, 1:2]) == 0)
                vizTiles = vcat(vizTiles, env.vizTiles[i, :]')
            end
        end
    else
        vizTiles = env.vizTiles[env.vizTiles[:, 4] .== 1, :]
    end
    
    return vizTiles[vizTiles[:, 3] .< 0, :]
end

function getPosRewardTerminals(env::HexEnv, path::Any = nothing)
    n = size(env.vizTiles)[1]

    if (path !== nothing)
        vizTiles = Matrix{Float64}(undef, 0, 4)
        for i in 1:n
            if (env.vizTiles[i, 4] == 1 && matchRow(path, env.vizTiles[i, 1:2]) == 0)
                vizTiles = vcat(vizTiles, env.vizTiles[i, :]')
            end
        end
    else
        vizTiles = env.vizTiles[env.vizTiles[:, 4] .== 1, :]
    end

    return vizTiles[vizTiles[:, 3] .>= 0, :]
end

# randomly choose a tile from the list of tiles to start from
function reset(env::HexEnv)
    normalTiles = env.tiles[env.tiles[:, 4] .== 0, :]
    env = @set env.currentState = normalTiles[rand(1:size(normalTiles)[1]), 1:2]
end;

# check if state is in the list of tiles
function matchRow(matrix, row)
    idx = 0
    for i in 1:size(matrix)[1]
        if (matrix[i, :] == row)
            idx = i
            break
        end
    end
    return idx
end;

# transfer from one state to another
function step(env::HexEnv, action::Int64)
    nextState = env.currentState + ACTIONS[action, :]
    if (action != 1 && action != 4)
        nextState[1] -= abs((nextState[2] + 1) % 2)
    end

    # check if the next state is in the list of tiles (available in the environment)
    position = matchRow(env.tiles[:, 1:2], nextState)

    if (position != 0)

        reward = STEP_COST

        reward += env.tiles[position, 3]

        if (env.tiles[position, 4] == 1)
            return env, nextState, reward, true
        end

        return env, nextState, reward, false

    else
        return env, env.currentState, OUT_OF_BOUNDS_PENALTY, false
    end
end;

# get next action with stochastic effect
function getNextActionWithProb(action::Int64, prob::Float64)

    neighborsProb = (1 - prob)
    randNum = rand(Uniform(0., 1.))

    # |=====|=====|===================|
    #    A     B            C

    # if the random number in A
    if (randNum < neighborsProb / 2)
        action = (action - 1) % HEX_ACTIONS_NUM
        if (action == 0)
            action = HEX_ACTIONS_NUM
        end
        return action;
    # if the random number in B
    elseif (randNum < neighborsProb)
        action = (action + 1) % HEX_ACTIONS_NUM
        if (action == 0)
            action = HEX_ACTIONS_NUM
        end
        return action;
    # if the random number in C
    else
        return action;
    end

end;

# get next action
function getNextAction(currentState::Vector{Int64}, epsilon, qTable; prob::Bool=true)
    if rand(Uniform(0., 1.)) > epsilon
        if prob
            return getNextActionWithProb(findmax(qTable[currentState[1], currentState[2], :])[2], MOVE_PROB)
        else
            return findmax(qTable[currentState[1], currentState[2], :])[2]
        end
    else
        if prob
            return getNextActionWithProb(rand(1:HEX_ACTIONS_NUM), MOVE_PROB)
        else
            return rand(1:HEX_ACTIONS_NUM)
        end
    end
end;

# get arrow corresponding to action
function getArrow(action::Int64)
    if (action == 1)
        return "←"
    elseif (action == 2)
        return "↖"
    elseif (action == 3)
        return "↗"
    elseif (action == 4)
        return "→"
    elseif (action == 5)
        return "↘"
    elseif (action == 6)
        return "↙"
    else
        return "?"
    end
end;

# find the best path from start to terminal
function visualize(env::HexEnv, startState::Vector{Int64}, qTable::Array{Float64, 3}; filename::Any = nothing, show::Bool = true, prob::Bool = true, render::Bool = true)

    env = @set env.currentState = startState
    
    maxMove = 2 * size(env.tiles)[1];
    move = 0;

    totalReward = 0;

    terminal = false;
    stepTiles = Matrix{Float64}(undef, 0,2);
    stepDirections = String[];

    while (!terminal)
        
        row = convert(Vector{Float64}, env.currentState)

        if (row[2] % 2 == 1)
            row[1] = row[1] - 0.5
        end
        row[2] = row[2] + 0.5

        nextAction = getNextAction(env.currentState, 0., qTable, prob=prob)
        env, nextState, reward, terminal = step(env, nextAction)
        totalReward += reward

        position = matchRow(stepTiles, row)
        if (position == 0)
            stepTiles = vcat(stepTiles, row')
            push!(stepDirections, getArrow(nextAction))
        else
            stepDirections[position] = getArrow(nextAction)
        end

        env = @set env.currentState = nextState

        move += 1
        if (move > maxMove)
            print("\nCan't find a path")
            return
        end
    end

    if (!render)
        return totalReward;
    end

    visualize(env, stepTiles, stepDirections, show = show)

    if (terminal)
        print("\nReward: $totalReward \n")
    end

    if filename !== nothing
        print("\nSaving current figure");
        savefig(fig, filename);
    end
end;

function visualizeEpsLabel(episode::Int)
    annotate!(fig, 0, 0.5, text("Episode: $episode", ANNOTATION_FONT_SIZE, :left));
end

# visualize blank map
function visualize(tile::Vector{Int64}; show = true)
    row = convert(Vector{Float64}, tile)
    if (row[2] % 2 == 1)
        row[1] = row[1] - 0.5
    end
    row[2] = row[2] + 0.5
    scatter!([row[1]], [row[2]], color = :wheat1, m = (0, [:h], MARKER_SIZE), lab = nothing)
    if show
        display(fig);
    end
end;

# visualize the optimal policy for the given map
function visualizeOptimalPolicy(env::HexEnv, qTable::Array{Float64,3}; show = true, filename = nothing)
    normalTiles = env.tiles[env.tiles[:, 4] .== 0, :]

    tiles = Matrix{Float64}(undef, 0,2);
    tileDirections = String[];
    directionAlpha = Float64[];

    minQ = findmin(qTable)[1];
    qTable = if (minQ < 0) qTable .+ abs(minQ) else qTable end;

    maxQ = findmax(qTable)[1];

    for i in 1:size(normalTiles)[1]
        row = convert(Vector{Float64}, normalTiles[i, 1:2])
        if (row[2] % 2 == 1)
            row[1] = row[1] - 0.5
        end
        row[2] = row[2] + 0.5

        maxAction = findmax(qTable[normalTiles[i, 1], normalTiles[i, 2], :])
        
        push!(directionAlpha, maxAction[1] / maxQ)
        tiles = vcat(tiles, row')
        tileDirections = push!(tileDirections, getArrow(maxAction[2]))
    end

    visualize(env, tiles, tileDirections, directionAlpha, show = show, optimalPolicy = true)

    if filename !== nothing
        print("\nSaving current figure");
        savefig(fig, filename);
    end
end;

end;