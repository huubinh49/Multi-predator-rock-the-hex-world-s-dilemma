include("./RPSAgents.jl")
import Pkg
Pkg.add("StatsBase")
Pkg.add("Distributions")
using StatsBase, Distributions
using .RPSAgents

function evaluate(agent1, agent2, steps)
  # Hàm đánh giá 2 agent đầu vào bằng cách cho chúng đấu với nhau
  score = 0 # Số lượt thắng của agent1
  tie = 0 # Số lượt hòa của agent1
  agent1_last_step, agent2_last_step = 0, 0
  for i in 0:steps-1
    agent1_observation = Dict(
      "step" => i,
      "lastOpponentAction" => agent2_last_step
    )
    agent2_observation = Dict(
      "step" => i,
      "lastOpponentAction" => agent1_last_step
    )
    # 2 agent lần lượt ra quyết định
    agent1_action = agent1.run(agent1_observation)
    agent2_action = agent2.run(agent2_observation)
    if(agent1_action == agent2_action)
      tie += 1
    else
      counter_agent1_action = (agent1_action + 1) % 3
      if(agent2_action == counter_agent1_action)
        score += 0
      else
        score += 1
      end
    end
    agent1_last_step = agent1_action
    agent2_last_step = agent2_action
  end
  return score, steps - score - tie
end

function compare()
  # Thực nghiệm so sánh ThompsonSamplingAgent với các agent khác
  # Danh sách các agent để so sánh với agent của chúng ta
  agents = Dict(
    "random_agent" => RandomAgent(),
    "only_scissor_agent" => OnlyScissorAgent(),
    "only_rock_agent" => OnlyRockAgent(),
    "only_paper_agent" => OnlyPaperAgent(),
    "copy_agent" => CopyAgent(),
    "freq_counting_agent" => FreqCountingAgent(),
    "counter_last_action_agent" => CounterLastActionAgent(),
    "markov_agent" => MarkovAgent(),
    "thompson_sampling_agent" = ThompsonSamplingAgent()
  )
  for (name, agent) in agents
    ourScores = []
    oppoScores = []
    usedAgents = Dict(
    "random_agent" => [],
    "only_scissor_agent" => [],
    "only_rock_agent" => [],
    "only_paper_agent" => [],
    "copy_agent" => [],
    "freq_counting_agent" => [],
    "counter_last_action_agent" => []
    )
    for i in (1:5)
      our_agent = ThompsonSamplingAgent()
      score1, score2 = evaluate(our_agent, agent, 1000)
      append!(ourScores, score1)
      append!(oppoScores, score2)
      for (agentName, agentCount) in our_agent.agent_count
        append!(usedAgents[agentName], agentCount)
      end
    end
    println("\n ----------------- \n")
    println(name)
    println("Our score: ", ourScores)
    println("Oppo score: ", oppoScores)
    println("Used agents: ")
    for (agentName, agentCounts) in usedAgents
      println("Mean used ", agentName, ": ", agentCounts)
    end
  end
end
compare()

