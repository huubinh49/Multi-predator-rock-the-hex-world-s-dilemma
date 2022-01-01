module Algos

include("./Environments.jl");
using .Environments
using Setfield
using Plots
using Distributions

export QLearning, Sarsa, train, evaluate

DEFAULT_EPSILON = 0.8
DEFAULT_DISCOUNT = 0.9
DEFAULT_LEARNING_RATE = 0.1
ACTIONS = ("left", "upper-left", "upper-right", "right", "lower-right", "lower-left")
ACTIONS_NUM = length(ACTIONS)

struct QLearning
    qTable::Array{Float64, 3}
    learningRate::Float64
    discountFactor::Float64
    epsilon::Float64
    env::HexEnv
end;

function QLearning(tiles::Matrix{Int64}; learningRate::Float64 = DEFAULT_LEARNING_RATE, 
                    discountFactor::Float64 = DEFAULT_DISCOUNT, epsilon::Float64 = DEFAULT_EPSILON)

    qTableSize = (findmax(tiles[:, 1])[1], findmax(tiles[:, 2])[1]);
    qTable = rand(Uniform(-1., 0.), tuple(vcat(collect(qTableSize), [ACTIONS_NUM])...));
    # qTable = zeros(tuple(vcat(collect(qTableSize), [ACTIONS_NUM])...));
    env = HexEnv(tiles);
    return QLearning(qTable, learningRate, discountFactor, epsilon, env);
end;

function train(agent::QLearning, episodes::Int64; visualizeEps = nothing, gifFile = nothing, prob::Bool = true, samplingRate::Float64 = 0.001)

    print("\nStep cost: $(Environments.STEP_COST) \n")
    print("Out of bounds penalty: $(Environments.OUT_OF_BOUNDS_PENALTY) \n")

    EPSILON_DECAY_START_EP = 1
    EPSILON_DECAY_END_EP = episodes // 2;
    EPSILON_DECAY = agent.epsilon / EPSILON_DECAY_END_EP;
    SAMPLING_EP = trunc(Int, 1 / samplingRate);

    # accumulated reward
    y = [];
    accumulatedReward = 0;

    anim = Environments.Plots.Animation()

    if visualizeEps === nothing
        visualizeEps = episodes + 1
    end

    for episode in 1:episodes

        if (episode % visualizeEps == 0)
            print("Visualizing episode $episode/$episodes\r");
            flush(stdout);
            if (gifFile !== nothing)
                visualize(agent.env, show = false);
                Environments.Plots.frame(anim);
            else
                visualize(agent.env, lowDpi = true);
                sleep(0.5);
            end;
            visualizeEpsLabel(episode);
        end;

        terminal = false
        agent = @set agent.env = Environments.reset(agent.env)
        
        while (!terminal)
            
            nextAction = getNextAction(agent.env.currentState, agent.epsilon, agent.qTable, prob = prob)
            
            env, nextState, reward, terminal = Environments.step(agent.env, nextAction)
            accumulatedReward += reward;
            
            currentQValue = agent.qTable[env.currentState[1], env.currentState[2], nextAction]
            
            newQValue = (1 - agent.learningRate) * currentQValue + agent.learningRate * 
                (reward + agent.discountFactor * findmax(agent.qTable[nextState[1], nextState[2], :])[1])

            agent.qTable[env.currentState[1], env.currentState[2], nextAction] = newQValue

            if (episode % visualizeEps == 0)
                if (gifFile !== nothing)
                    visualize(agent.env.currentState, show = false);
                    Environments.Plots.frame(anim);
                else
                    visualize(agent.env.currentState);
                    sleep(0.1);
                end
            end
            
            env = @set env.currentState = nextState

            agent = @set agent.env = env
    
        end

        if (episode <= EPSILON_DECAY_END_EP && episode >= EPSILON_DECAY_START_EP)
            agent = @set agent.epsilon = agent.epsilon - EPSILON_DECAY
        end

        if (episode % visualizeEps == 0)
            if (gifFile === nothing)
                sleep(1);
            end;
        end;

        if (visualizeEps > episodes && episode % SAMPLING_EP == 0)
            push!(y, accumulatedReward / episode);
            print("Trained $episode/$episodes episodes\r");
            flush(stdout);
        end
        
    end;

    print("\n");

    if (gifFile !== nothing)
        print("Writing GIF\n");
        Environments.Plots.gif(anim, gifFile, fps = 10);
    else
        if (visualizeEps <= episodes)
            print("Press any key to continue...\n");
            readline();
        end
    end;
        
    return y;
end;

struct Sarsa
    qTable::Array{Float64, 3}
    learningRate::Float64
    discountFactor::Float64
    epsilon::Float64
    env::HexEnv
end;

function Sarsa(tiles::Matrix{Int64}; learningRate::Float64 = DEFAULT_LEARNING_RATE, 
                    discountFactor::Float64 = DEFAULT_DISCOUNT, epsilon::Float64 = DEFAULT_EPSILON)

    qTableSize = (findmax(tiles[:, 1])[1], findmax(tiles[:, 2])[1]);
    qTable = rand(Uniform(-1., 0.), tuple(vcat(collect(qTableSize), [ACTIONS_NUM])...));
    env = HexEnv(tiles);
    return Sarsa(qTable, learningRate, discountFactor, epsilon, env);
end;

function train(agent::Sarsa, episodes::Int64; visualizeEps = nothing, gifFile = nothing, prob::Bool = true, samplingRate::Float64 = 0.001)

    print("Step cost: $(Environments.STEP_COST) \n")
    print("Out of bounds penalty: $(Environments.OUT_OF_BOUNDS_PENALTY) \n")

    EPSILON_DECAY_START_EP = 1
    EPSILON_DECAY_END_EP = episodes // 2;
    EPSILON_DECAY = agent.epsilon / EPSILON_DECAY_END_EP;
    SAMPLING_EP = trunc(Int, 1 / samplingRate);

    # accumulated reward
    y = [];
    accumulatedReward = 0;

    anim = Environments.Plots.Animation()

    if visualizeEps === nothing
        visualizeEps = episodes + 1
    end

    for episode in 1:episodes

        if (episode % visualizeEps == 0)
            print("Visualizing episode $episode/$episodes\r");
            flush(stdout);
            if (gifFile !== nothing)
                visualize(agent.env, show = false);
                Environments.Plots.frame(anim);
            else
                visualize(agent.env, lowDpi = true);
                sleep(0.5);
            end;
            visualizeEpsLabel(episode);
        end;

        terminal = false
        agent = @set agent.env = Environments.reset(agent.env)
        nextAction = getNextAction(agent.env.currentState, agent.epsilon, agent.qTable, prob = prob)
        
        while (!terminal)

            if (episode % visualizeEps == 0)
                if (gifFile !== nothing)
                    visualize(agent.env.currentState, show = false);
                    Environments.Plots.frame(anim);
                else
                    visualize(agent.env.currentState);
                    sleep(0.1);
                end
            end
            
            env, nextState, reward, terminal = Environments.step(agent.env, nextAction)
            accumulatedReward += reward;

            currentQValue = agent.qTable[env.currentState[1], env.currentState[2], nextAction]

            nextNextAction = getNextAction(nextState, agent.epsilon, agent.qTable, prob = prob)
            
            newQValue = (1 - agent.learningRate) * currentQValue + agent.learningRate * 
                (reward + agent.discountFactor * agent.qTable[nextState[1], nextState[2], nextNextAction])

            agent.qTable[env.currentState[1], env.currentState[2], nextAction] = newQValue
            
            env = @set env.currentState = nextState

            agent = @set agent.env = env

            nextAction = nextNextAction
    
        end

        if (episode <= EPSILON_DECAY_END_EP && episode >= EPSILON_DECAY_START_EP)
            agent = @set agent.epsilon = agent.epsilon - EPSILON_DECAY
        end

        if (episode % visualizeEps == 0)
            if (gifFile === nothing)
                sleep(1);
            end;
        end;

        if (visualizeEps > episodes && episode % SAMPLING_EP == 0)
            push!(y, accumulatedReward / episode);
            print("Trained $episode/$episodes episodes\r");
            flush(stdout);
        end
        
    end;

    print("\n");

    if (gifFile !== nothing)
        print("Writing GIF\n");
        Environments.Plots.gif(anim, gifFile, fps = 10);
    else
        if (visualizeEps <= episodes)
            print("Press any key to continue...\n");
            readline();
        end
    end;
    return y;
end;

function evaluate(agent, times = 100000; samplingRate = 0.001)
    result = [];
    SAMPLING_TIME = trunc(Int, 1 / samplingRate);
    accumulatedReward = 0;
    print("\n");
    for time in 1:times
        agent = @set agent.env = Environments.reset(agent.env);
        accumulatedReward += Environments.visualize(agent.env, agent.env.currentState, agent.qTable, show = false, render = false);
        if (time % SAMPLING_TIME == 0)
            push!(result, accumulatedReward / time);
            print("Evaluated $time/$times times\r");
            flush(stdout);
        end
    end
    return result;
end;

end;