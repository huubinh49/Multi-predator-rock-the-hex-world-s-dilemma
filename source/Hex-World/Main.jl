include("./Algos.jl")
using .Algos
using CSV
using DataFrames
using Plots

MAP_NAME = "map2"
TRAINING_EPS = 300000
EVALUATING_EPS = 100000
SAMPLING_RATE = 0.001

df = DataFrame(CSV.File("maps/$(MAP_NAME).csv"));
tiles = Matrix(df);
agent = Sarsa(tiles, learningRate = 0.1);
y1 = train(agent, TRAINING_EPS, samplingRate = SAMPLING_RATE);
Algos.visualizeOptimalPolicy(agent.env, agent.qTable, filename = "assets/sarsa_$(MAP_NAME)_optimal.png");

yt1 = Algos.evaluate(agent, EVALUATING_EPS, samplingRate = SAMPLING_RATE);

tiles = Matrix(df);
agent = QLearning(tiles, learningRate = 0.1);
y2 = train(agent, TRAINING_EPS, samplingRate = SAMPLING_RATE);
Algos.visualizeOptimalPolicy(agent.env, agent.qTable, filename = "assets/QL_$(MAP_NAME)_optimal.png");

yt2 = Algos.evaluate(agent, EVALUATING_EPS, samplingRate = SAMPLING_RATE);

fig = plot(collect(1:length(y1)) .* trunc(Int, 1 / SAMPLING_RATE), hcat(y1, y2), xlabel = "Episodes", ylabel = "Average Reward per Episode", 
            label = ["SARSA" "Q-Learning"], lw = 2, dpi = 300, xlabelfontsize = 8, ylabelfontsize = 8, 
            legend=:right, title = "Training", titlefontsize = 10);

savefig(fig, "assets/evaluation.png");

fig = plot(collect(1:length(yt1)) .* trunc(Int, 1 / SAMPLING_RATE), hcat(yt1, yt2), xlabel = "Episodes", ylabel = "Average Reward per Episode", 
            label = ["SARSA" "Q-Learning"], lw = 2, dpi = 300, xlabelfontsize = 10, ylabelfontsize = 10, 
            legend=:right, title = "Testing", titlefontsize = 10);

savefig(fig, "assets/evaluation_trained.png");

