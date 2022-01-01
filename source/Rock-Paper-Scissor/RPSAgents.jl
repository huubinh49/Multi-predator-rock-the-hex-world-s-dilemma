module RPSAgents
using Random
import Pkg
Pkg.add("StatsBase")
Pkg.add("Distributions")
using StatsBase, Distributions
export RandomAgent, OnlyScissorAgent, OnlyRockAgent,OnlyPaperAgent,CopyAgent, FreqCountingAgent, CounterLastActionAgent, ThompsonSamplingAgent, MarkovAgent

function RandomAgent()
  # Agent chơi bằng cách random sự lựa chọn
  function run(observation)
    return rand((0, 1, 2))
  end
  () -> (run)
end

function CopyAgent()
  # Agent chơi bằng cách copy hành động của đối thủ ở lượt trước đó
  function run(observation)
    if(observation["step"] > 0)
      return observation["lastOpponentAction"]
    end
    return rand((0, 1, 2))
  end
  () -> (run)
end

function CounterLastActionAgent()
  # Agent chơi bằng cách khắc chế hành động của đối thủ ở lượt trước đó
  function run(observation)
    return (observation["lastOpponentAction"] + 1) % 3
  end
  () -> (run)
end

function OnlyRockAgent()
  function run(observation)
    return 1
  end
  () -> (run)
end

function OnlyScissorAgent()
  function run(observation)
    return 0
  end
  () -> (run)
end

function OnlyPaperAgent()
  function run(observation)
    return 2
  end
  () -> (run)
end

function FreqCountingAgent()
  # Agent chơi bằng cách khắc chế lựa chọn mà đối thủ chọn nhiều nhất
  freqs = [1, 1, 1]
  
  function run(observation)
    if(observation["step"] == 0)
      return rand((0, 1, 2))
    end
    pred_action = findall(action->action == maximum(freqs), freqs)[1] - 1 
    return (pred_action + 1) % 3
  end
  () -> (run)
end

function MarkovAgent()
  # Agent chơi bằng cách khắc chế pattern mà đối thủ chọn nhiều nhất
  table = Dict() 
  action_seq = []
  
  function run(observation)
    k = 2
    # In early steps we can choose randomly option
    if(observation["step"] <= 2*k +1)
      action = rand((0,1,2))
      if(observation["step"] > 0)
        append!(action_seq, [observation["lastOpponentAction"], action])
      else
        append!(action_seq, action)
      end
      return action
    end
    key = join([i for i in action_seq[1: length(action_seq)-1]], "")
    if(haskey(table, key))
      table[key][observation["lastOpponentAction"]+1] += 1
    else
      table[key] = [1, 1, 1]
    end

    # Update action sequence
    deleteat!(action_seq, [1, 2])
    
    if(observation["step"] < 20)
      weights = table[key] ./ sum(table[key])
      pred_action = sample([0, 1, 2], Weights(weights))
    else  
    pred_action = findall(action->action == maximum(table[key]), table[key])[1] - 1 
    end
    counter_action = (pred_action + 1)%3
    append!(action_seq, [observation["lastOpponentAction"], counter_action])
    return counter_action
  end
  () -> (run)
end

function ThompsonSamplingAgent()
  decay_rate = 1.05
  reward = 3
  # Lưu số lượt thắng thua của từng agent
  bandit_state = Dict()
  # Lưu những thông tin về những lượt chơi
  history = []
  # Chứa nhưng agent đại diện cho từng chiến thuật chơi
  agents = Dict(
    "random_agent" => RandomAgent(),
    "only_scissor_agent" => OnlyScissorAgent(),
    "only_rock_agent" => OnlyRockAgent(),
    "only_paper_agent" => OnlyPaperAgent(),
    "copy_agent" => CopyAgent(),
    "freq_counting_agent" => FreqCountingAgent(),
    "counter_last_action_agent" => CounterLastActionAgent()
  )
  # Đếm số lần được lựa chọn của mỗi agent 
  agent_count = Dict(
    "random_agent" => 0,
    "only_scissor_agent" => 0,
    "only_rock_agent" => 0,
    "only_paper_agent" => 0,
    "copy_agent" => 0,
    "freq_counting_agent" => 0,
    "counter_last_action_agent" => 0
  )
  
  # Lưu những thông tin của lượt hiện tai
  function store_current_state(history, observation, agent_name)
    agent_count[agent_name] += 1;
    state = Dict("step" => observation["step"],"agent" => agent_name,"lastOpponentAction" => -1)
    append!(history, [state])
    return history
  end
  # Cập nhật lại history với lượt phản hồi của đối thủ
  function update_history(history, observation)
    history[length(history)]["lastOpponentAction"] = observation["lastOpponentAction"]
    return history
  end
  
  function run(observation)
    # Mặc định sẽ chọn chiến thuật random
    agent_name = "random_agent"
    if(observation["step"] < 2)
      agent_name = "random_agent"
    else
      history = update_history(history,observation)
      last_observation = history[length(history)-1]
      gt_action = history[length(history)]["lastOpponentAction"]
      best_prob = -1
      best_agent = "random_agent"
      # Chạy các agent để xem chúng có thắng lượt vừa rồi không và tính xác suất thắng của chúng ở lượt này theo phân phối Beta
      for (name, agent) in agents
        pred_action = agent.run(last_observation)
        if(haskey(bandit_state, name))
          bandit_state[name][1] = (bandit_state[name][1] .- 1) ./ decay_rate .+ 1
          bandit_state[name][2] = (bandit_state[name][2] .- 1) ./ decay_rate .+ 1
        else
          bandit_state[name] = [1.0, 1.0]
        end
        if(pred_action == gt_action)
          bandit_state[name][1] = bandit_state[name][1] .+ reward/2
          bandit_state[name][2] = bandit_state[name][2] .+ reward/2
        else
          counter_agent_action = (pred_action + 1) % 3
          if(gt_action == counter_agent_action)
            bandit_state[name][2] = bandit_state[name][2] .+ reward
          else
            bandit_state[name][1] = bandit_state[name][1] .+ reward
          end
        end
        prob = rand(Beta(bandit_state[name][1], bandit_state[name][2]), 1)[1]
        if(prob > best_prob)
          best_agent = name
          best_prob = prob
        end
      end
      agent_name = best_agent
    end
    # Dùng agent đã lựa chọn để ra quyết định
    step = agents[agent_name].run(observation)
    # Lưu thông tin về lượt hiện tại
    history = store_current_state(history, observation, agent_name)
    return step
  end
  () -> (run, agent_count)
end
end
