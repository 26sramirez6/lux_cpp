/*
 * actor.hpp
 *
 *  Created on: May 10, 2021
 *      Author: saul.ramirez
 */

#ifndef ACTOR_HPP_
#define ACTOR_HPP_

#include <chrono>
#include <cmath>
#include <random>
#include <type_traits>

#include "board.hpp"
#include "feature_builder.hpp"
#include "model_config.hpp"
#include "replay_buffer.hpp"

template <typename Mask, typename RandomEngine>
static inline void
override_actions_by_choice(std::true_type, RandomEngine &random_engine_,
                           const int _remove_amount,
                           const Mask &_remove_action_mask,
                           Eigen::Ref<Eigen::ArrayXi> new_random_actions_,
                           Eigen::Ref<Eigen::ArrayXi> actions_) {
  choice(random_engine_, static_cast<int>(ShipAction::Count) - 1,
         new_random_actions_);
  actions_ = _remove_action_mask.select(new_random_actions_, actions_);
}

template <typename Mask, typename RandomEngine>
static inline void
override_actions_by_choice(std::false_type, RandomEngine &random_engine_,
                           const int _remove_amount,
                           const Mask &_remove_action_mask,
                           Eigen::Ref<Eigen::ArrayXi> new_random_actions_,
                           Eigen::Ref<Eigen::ArrayXi> actions_) {
  actions_ = _remove_action_mask.select(static_cast<int>(ShipyardAction::NONE),
                                        actions_);
}

template <bool is_ship, typename RandomEngine>
static inline void
limit_actions_by_choice(RandomEngine &random_engine_, const int _max_allowed,
                        const int _limiting_action,
                        Eigen::Ref<Eigen::ArrayXi> cumsum_,
                        Eigen::Ref<Eigen::ArrayXi> new_random_actions_,
                        Eigen::Ref<Eigen::ArrayXi> actions_) {

  auto limiting_action_mask = actions_ == _limiting_action;
  const int remove_amount = limiting_action_mask.count() - _max_allowed;
  if (remove_amount > 0) {
    cumsum(limiting_action_mask, cumsum_);

    auto exceeded_limit_mask = cumsum_ > _max_allowed;
    auto remove_action_mask = limiting_action_mask && exceeded_limit_mask;

    override_actions_by_choice(
        std::integral_constant<bool, is_ship>{}, random_engine_, remove_amount,
        remove_action_mask, new_random_actions_, actions_);
  }
}

template <typename Mask>
static inline void override_actions_by_q(std::true_type,
                                         const Mask &_remove_action_mask,
                                         const torch::Tensor &_q_values,
                                         Eigen::Ref<Eigen::ArrayXi> actions_) {
  torch::Tensor top2 = std::get<1>(torch::topk(_q_values, 2));
  auto a = top2.accessor<int64_t, 2>();
  for (int i = 0; i < _remove_action_mask.size(); ++i) {
    if (_remove_action_mask(i)) {
      actions_(i) = a[i][1];
    }
  }
}

template <typename Mask>
static inline void override_actions_by_q(std::false_type,
                                         const Mask &_remove_action_mask,
                                         const torch::Tensor &_q_values,
                                         Eigen::Ref<Eigen::ArrayXi> actions_) {
  actions_ = _remove_action_mask.select(static_cast<int>(ShipyardAction::NONE),
                                        actions_);
}

template <bool is_ship>
static inline void limit_actions_by_q(const int _max_allowed,
                                      const int _limiting_action,
                                      Eigen::Ref<Eigen::ArrayXi> cumsum_,
                                      Eigen::Ref<Eigen::ArrayXi> actions_,
                                      const torch::Tensor &_q_values) {

  auto limiting_action_mask = actions_ == _limiting_action;
  const int remove_amount = limiting_action_mask.count() - _max_allowed;
  if (remove_amount > 0) {
    cumsum(limiting_action_mask, cumsum_);

    auto exceeded_limit_mask = cumsum_ > _max_allowed;
    auto remove_action_mask = limiting_action_mask && exceeded_limit_mask;

    override_actions_by_q(std::integral_constant<bool, is_ship>{},
                          remove_action_mask, _q_values, actions_);
  }
}

template <std::size_t ActorId, torch::DeviceType DeviceType,
          typename RewardEngine, typename ReplayBuf, 
					typename ModelConfig>
class MultiStepPawnManager {
public:
	using BatchStateFeatures = BatchStateFeature<DeviceType, BoardConfig, ModelConfig>;
	using FinalBatch = DynamicBatch<DeviceType, BoardConfig, ModelConfig>;
  MultiStepPawnManager(const std::size_t _multi_step_n, const float _gamma,
                       const std::size_t _batch_size)
      : m_n(_multi_step_n), m_gamma(_gamma), m_batch_size(_batch_size),
        m_retained_id_indices(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt64)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_destroyed_id_indices(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt64)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_multi_step_pawn_ids(), m_multi_step_actions(),
        m_multi_step_features(), m_multi_step_rewards(_multi_step_n),
        m_latest_pawn_count(0), m_nth_rewards_prior_cursor(0), 
				m_nth_ids_prior_size(0), m_final_batch()

  {}

  inline const FinalBatch& getFinalBatch() const { return m_final_batch; }

	inline std::size_t getLatestPawnCount() const { 	return m_latest_pawn_count; }

  inline std::size_t getQueueSize() const { return m_multi_step_actions.size(); }

	inline const BatchStateFeatures& getLatestStateFeatures() const { 
		return m_multi_step_features.back(); 
	}
	
	inline void printRewardState() const {
		std::cout << "reward cursor: " << m_nth_rewards_prior_cursor << std::endl;
		int i = 0;
		for (auto& map : m_multi_step_rewards) {
			std::cout << i << ": {";
			for (auto& kv : map) {
				std::cout << kv.first << ": " << kv.second << ", ";
			}
			std::cout << "}" << std::endl;
		}
	}	
	
  template <typename FeatureBuilder, typename Board>
  void updatePawnStates(const Board &_current_board,
                        RewardEngine &reward_engine_,
                        const torch::Tensor &_latest_actions) {
    // TODO: make generic for any pawn
    const auto &pawn_map = _current_board.getShipMap();
		m_latest_pawn_count = pawn_map.size(); // don't rely on multi_step_pawn_ids
		pushLatestFeatureState<FeatureBuilder>(_current_board);

		if (_latest_actions.size(0) > 0) {
			pushLatestPawns(pawn_map);
			pushLatestRewards(_current_board, reward_engine_);
			pushLatestActions(_latest_actions);
		}
    if (_current_board.m_step > m_n && m_multi_step_pawn_ids.size() == m_n) {
      std::vector<int> nth_retained_ids_prior;
      const auto &nth_ids_prior = m_multi_step_pawn_ids.front();
      nth_retained_ids_prior.reserve(nth_ids_prior.size());

      m_nth_ids_prior_size = nth_ids_prior.size();
			if (!m_nth_ids_prior_size) {
				return;
			}

      auto retained_accessor = m_retained_id_indices.accessor<int64_t, 1>();
      auto destroyed_accessor = m_destroyed_id_indices.accessor<int64_t, 1>();
      int retained_count = 0;
      int destroyed_count = 0;
			const auto& latest_ids = m_multi_step_pawn_ids.back();
      for (int i = 0; i < nth_ids_prior.size(); ++i) {
        const int id_prior = nth_ids_prior[i];
        if (id_prior == latest_ids[retained_count]) {
          retained_accessor[retained_count++] = i;
          nth_retained_ids_prior.push_back(id_prior);
        } else {
          destroyed_accessor[destroyed_count++] = i;
        }
      }
			m_nth_ids_prior_size = nth_ids_prior.size();
		
      updateFinalBatchFeatureState<FeatureBuilder>(nth_ids_prior.size(),
                                                   retained_count);
      updateFinalBatchNonTerminal(retained_count);
      updateFinalBatchActions(nth_ids_prior.size());
      updateFinalBatchRewards(_current_board, nth_retained_ids_prior,
                              reward_engine_);

      m_multi_step_pawn_ids.pop();
    }
  }

	inline bool pushTransitions(ReplayBuf& replay_buf_) const {
		if (m_multi_step_pawn_ids.size() == m_n-1) {
			replay_buf_.push(m_final_batch, m_nth_ids_prior_size);
			return true;
		}

		return false;
	}

	void inline resetState() {
		m_retained_id_indices.zero_();
		m_destroyed_id_indices.zero_();

		std::queue<std::vector<int>>().swap(m_multi_step_pawn_ids);
		std::queue<torch::Tensor>().swap(m_multi_step_actions);
		std::queue<BatchStateFeatures>().swap(m_multi_step_features);
		m_latest_pawn_count = 0;
		m_nth_rewards_prior_cursor = 0;
		m_nth_ids_prior_size = 0;
	}

private:
	template<typename PawnMap>
	void inline pushLatestPawns(const PawnMap& _pawn_map) {
    std::vector<int> latest_ids;
    latest_ids.reserve(_pawn_map.size());
    std::transform(_pawn_map.begin(), _pawn_map.end(),
                   std::back_inserter(latest_ids),
                   [](const auto &pair) { return pair.first; });

    m_multi_step_pawn_ids.push(std::move(latest_ids));
	}	

  void inline pushLatestActions(const torch::Tensor &_latest_actions) {
		std::cout << "pushing action: " << _latest_actions.index({0}).item().to<float>() << std::endl;
    m_multi_step_actions.push(_latest_actions.clone());
  }

  template <typename FeatureBuilder, typename Board>
  void inline pushLatestFeatureState(const Board &_current_board) {
    BatchStateFeatures feature_state(m_batch_size);
    FeatureBuilder::template setStateFeatures<BoardConfig>(_current_board,
                                                           feature_state);
    m_multi_step_features.push(std::move(feature_state));
  }

  template <typename Board>
  void inline pushLatestRewards(const Board &_current_board,
                                RewardEngine &reward_engine_) {
		std::unordered_map<int, float> reward_map;
    reward_engine_.template computeRewards<ActorId>(_current_board,
                                                    reward_map);
		std::cout << "pushing rewards: ";
		for(auto& kv : reward_map) {
			std::cout << kv.first << ", " << kv.second << std::endl; 
		}
		std::cout << std::endl;

    m_multi_step_rewards[m_nth_rewards_prior_cursor] = std::move(reward_map);
    m_nth_rewards_prior_cursor = (m_nth_rewards_prior_cursor + 1) % m_n;
  }

  void inline updateFinalBatchNonTerminal(const std::size_t _up_to_retained) {
    auto up_to_retained = torch::indexing::Slice(0, _up_to_retained, 1);
    m_final_batch.m_is_non_terminal.index_put_({torch::indexing::None}, false);
    m_final_batch.m_is_non_terminal.index_put_(
        {m_retained_id_indices.index({up_to_retained})}, true);
  }

  template <typename FeatureBuilder>
  void inline updateFinalBatchFeatureState(
      const std::size_t _nth_pawn_count_prior,
      const std::size_t _retained_pawn_count) {
    auto up_to_retained = torch::indexing::Slice(0, _retained_pawn_count, 1);
    const auto up_to_prior_pawn_count =
        torch::indexing::Slice(0, _nth_pawn_count_prior, 1);
    const auto &nth_features_prior = m_multi_step_features.front();

    m_final_batch.m_state.m_geometric.index_put_(
        {up_to_prior_pawn_count},
        nth_features_prior.m_geometric.index({up_to_prior_pawn_count}));

    m_final_batch.m_next_state.m_geometric.index_put_(
        {m_retained_id_indices.index({up_to_retained})},
        m_multi_step_features.back().m_geometric.index({up_to_retained}));

    // pop at end to avoid invalidating the reference
		// only pop when queue is 1 size greater
		// i.e. S1 -> S5 with m_n==4 will require a queue holding 5 states
		if (m_multi_step_features.size() == m_n + 1) {
    	m_multi_step_features.pop();
		}
  }

  void inline updateFinalBatchActions(const std::size_t _nth_pawn_count_prior) {
    // the latest actions have been pushed by this point
    const auto &nth_actions_prior = m_multi_step_actions.front();
    const auto up_to_prior_pawn_count =
        torch::indexing::Slice(0, _nth_pawn_count_prior, 1);
    // nth_actions_prior won't be on gpu, so must use to(Device) before
    // index_put
	
    m_final_batch.m_action.index_put_(
        {up_to_prior_pawn_count},
        nth_actions_prior.index({up_to_prior_pawn_count}).to(DeviceType));
    m_multi_step_actions.pop();
  }

  template <typename Board>
  void updateFinalBatchRewards(const Board &_current_board,
                               const std::vector<int> &_nth_retained_ids_prior,
                               RewardEngine &reward_engine_) {
		// copy ctor called here
    auto cumulative_rewards_map = m_multi_step_rewards[m_nth_rewards_prior_cursor];
		
		std::cout << "cum rewards before: ";
		for (auto& kv : cumulative_rewards_map) {
			std::cout << kv.first << ": " << kv.second <<", ";
		}		
		std::cout << std::endl;


		for (auto id : _nth_retained_ids_prior) {
			if (cumulative_rewards_map.find(id) == cumulative_rewards_map.end()) { // doesn't contain
				cumulative_rewards_map.insert({id, 0});
			}
		}		

    for (int i = 1; i < m_n; ++i) {
      const int index = (i + m_nth_rewards_prior_cursor) % m_n;
      const auto &next_reward_map = m_multi_step_rewards[index];
      const float gamma = std::pow(m_gamma, i);
      for (auto &kv : cumulative_rewards_map) {
        if (next_reward_map.find(kv.first) !=
            next_reward_map.end()) { // contains
          kv.second += gamma*next_reward_map.at(kv.first);
        }
      }
    }	

		std::cout << "cum rewards after: ";
		for (auto& kv : cumulative_rewards_map) {
			std::cout << kv.first << ": " << kv.second <<", ";
		}		
		std::cout << std::endl;

    auto retained_accessor = m_retained_id_indices.accessor<int64_t, 1>();

		std::cout << "nth retained ids prior: ";
		for (auto i : _nth_retained_ids_prior) std::cout << i << ",";
		std::cout << std::endl;
    for (int i = 0; i < _nth_retained_ids_prior.size(); ++i) {
      m_final_batch.m_reward.index_put_(
          {i}, cumulative_rewards_map.at(_nth_retained_ids_prior[i]));
    }
  }

private:
  const std::size_t m_n;
  const float m_gamma;
  const std::size_t m_batch_size;


  torch::Tensor m_retained_id_indices;
  torch::Tensor m_destroyed_id_indices;

  std::queue<std::vector<int>> m_multi_step_pawn_ids;
  std::queue<torch::Tensor> m_multi_step_actions;
  std::queue<BatchStateFeatures> m_multi_step_features;
  std::vector<std::unordered_map<int, float>> m_multi_step_rewards;

	std::size_t m_latest_pawn_count;
  std::size_t m_nth_rewards_prior_cursor;
	std::size_t m_nth_ids_prior_size;

  FinalBatch m_final_batch;
};

template <std::size_t ActorId, torch::DeviceType DeviceType,
          typename BoardConfig, typename ShipModelConfig,
          typename ShipyardModelConfig, typename ShipFeatureBuilder,
          typename ShipyardFeatureBuilder, typename ShipRewardEngine,
          typename ShipyardRewardEngine, typename ShipReplayBuffer,
          typename ShipyardReplayBuffer, typename RandomEngine>
class Actor {
public:
  Actor(const float _epsilon_decay, const float _epsilon_start,
        const float _epsilon_end, const std::size_t _atom_count,
        const float _v_min, const float _v_max, const std::size_t _multi_step_n,
        const float _gamma, const std::size_t _batch_size)
      : m_epsilon_decay(_epsilon_decay), m_epsilon_start(_epsilon_start),
        m_epsilon_end(_epsilon_end), m_episodes(0), m_prior_ship_count(0),
        m_prior_shipyard_count(0), m_multi_step_n(_multi_step_n),
        m_final_ship_batch(), m_final_shipyard_batch(),
        m_best_ship_actions(BoardConfig::size * BoardConfig::size),
        m_best_shipyard_actions(BoardConfig::size * BoardConfig::size),
        m_cumsum(BoardConfig::size * BoardConfig::size),
        m_new_random_actions(BoardConfig::size * BoardConfig::size),
        m_ship_action_recorder(
            Eigen::ArrayXf::Ones(ShipModelConfig::output_size)),
        m_shipyard_action_recorder(
            Eigen::ArrayXf::Ones(ShipyardModelConfig::output_size)),
        m_prior_ship_actions(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt32)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_prior_shipyard_actions(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt32)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_retained_ship_indices(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt64)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_retained_shipyard_indices(torch::zeros(
            {BoardConfig::size * BoardConfig::size}, torch::dtype(torch::kInt64)
                                                         .requires_grad(false)
                                                         .device(torch::kCPU))),
        m_support(torch::linspace(_v_min, _v_max, _atom_count,
                                  torch::dtype(torch::kFloat32)
                                      .requires_grad(false)
                                      .device(DeviceType))),
        m_ship_pawn_manager(_multi_step_n, _gamma, _batch_size){};

  inline bool isEpsilonFrame(RandomEngine &random_engine_) const {
    return random_engine_.uniform() <
           m_epsilon_end + (m_epsilon_start - m_epsilon_end) *
                               std::exp(-m_episodes / m_epsilon_decay);
  }

  inline void updateRecorderForActionsTaken(
      const Eigen::Ref<const Eigen::ArrayXi> &_actions_taken,
      Eigen::Ref<Eigen::ArrayXf> action_recorder_) {
    for (int i = 0; i < _actions_taken.size(); ++i) {
      action_recorder_(_actions_taken(i)) += 1;
    }
  }

  template <typename Board, typename ShipDQN, typename ShipyardDQN>
  inline void processEpisode(const Board &_current_board,
                             ShipDQN &dynamic_ship_model_,
                             ShipyardDQN &dynamic_shipyard_model_,
                             ShipReplayBuffer &ship_replay_buffer_,
                             ShipyardReplayBuffer &shipyard_replay_buffer_,
                             ShipRewardEngine &ship_reward_engine_,
                             ShipyardRewardEngine &shipyard_reward_engine_,
                             RandomEngine &random_engine_) {

    std::cout << "ACTOR: processEpisode()" << std::endl;
    const std::size_t latest_ship_count = _current_board.getShipCount();
    const std::size_t latest_shipyard_count = _current_board.getShipyardCount();
    const bool any_ships = latest_ship_count > 0;
    const bool any_shipyards = latest_shipyard_count > 0;
    const bool any_obj = any_ships || any_shipyards;

    const auto &ship_map = _current_board.getShipMap();
    const auto &shipyard_map = _current_board.getShipyardMap();
    const auto &retained_ships =
        _current_board.template getRetainedShips<ActorId>();
    const auto &retained_shipyards =
        _current_board.template getRetainedShipyards<ActorId>();
    const int retained_ships_count = retained_ships.size();
    const int retained_shipyards_count = retained_shipyards.size();

		m_ship_pawn_manager.template updatePawnStates<ShipFeatureBuilder>(
			_current_board, 
			ship_reward_engine_,
			m_prior_ship_actions.index({torch::indexing::Slice(0, m_prior_ship_count, 1)}));

		m_ship_pawn_manager.pushTransitions(ship_replay_buffer_);

    if (any_obj && isEpsilonFrame(random_engine_)) {
      if (any_ships) {
        std::cout << "m_ship_action_recorder" << std::endl;
        std::cout << m_ship_action_recorder << std::endl;
        // TODO: Remove
        // m_ship_action_recorder(5) = 1e10;
        Eigen::ArrayXf probs = (m_ship_action_recorder.sum() -
                                m_ship_action_recorder.head(5) /*TODO Remove*/);

        probs /= probs.sum();

        std::cout << "probs:" << std::endl;
        std::cout << probs << std::endl;
        choice(random_engine_, probs,
               m_best_ship_actions.head(latest_ship_count));
        std::cout << "choice: " << m_best_ship_actions.head(latest_ship_count)
                  << std::endl;

        const int max_shipyards_allowed =
            _current_board.getShipyardCount() == 0 ? 1 : 0;
        limit_actions_by_choice<true>(
            random_engine_, max_shipyards_allowed,
            static_cast<int>(ShipAction::CONVERT),
            m_cumsum.head(latest_ship_count),
            m_new_random_actions.head(latest_ship_count),
            m_best_ship_actions.head(latest_ship_count));
      }

      if (any_shipyards) {
        Eigen::ArrayXf probs = (m_shipyard_action_recorder.maxCoeff() -
                                m_shipyard_action_recorder) /
                               m_shipyard_action_recorder.sum();
        choice(random_engine_, probs,
               m_best_shipyard_actions.head(latest_shipyard_count));

        const int max_ships_allowed =
            _current_board.getShipCount() == 0 ? 1 : 0;
        limit_actions_by_choice<false>(
            random_engine_, max_ships_allowed,
            static_cast<int>(ShipyardAction::SPAWN),
            m_cumsum.head(latest_shipyard_count),
            m_new_random_actions.head(latest_shipyard_count),
            m_best_shipyard_actions.head(latest_shipyard_count));
      }

    } else if (any_obj) {
      torch::NoGradGuard no_grad;
      if (any_ships) {
        const auto slice = torch::indexing::Slice(0, m_ship_pawn_manager.getLatestPawnCount(), 1);

				const auto& state_features = m_ship_pawn_manager.getLatestStateFeatures();

				torch::Tensor q_distribution = dynamic_ship_model_.forward(
            state_features.m_geometric.index({slice}));

        torch::Tensor q_projected_dist = q_distribution * m_support;

        torch::Tensor q_current = q_projected_dist.sum(2).cpu();

        torch::Tensor argmax = std::get<1>(q_current.max(1));

        tensor_to_eigen<int64_t>(argmax, m_best_ship_actions);
        const int max_shipyards_allowed =
            _current_board.getShipyardCount() == 0 ? 1 : 0;

        limit_actions_by_q<true>(
            max_shipyards_allowed, static_cast<int>(ShipAction::CONVERT),
            m_cumsum.head(latest_ship_count),
            m_best_ship_actions.head(latest_ship_count), q_current);
        std::cout << "ACTOR Q current: " << std::endl;
        std::cout << q_current << std::endl;
      }

      if (any_shipyards) {
        m_best_shipyard_actions.head(latest_shipyard_count) = 0;
      }
    }

    //m_final_shipyard_batch.m_state = m_tmp_shipyard_batch.m_next_state;

    eigen_to_tensor<int32_t>(m_best_ship_actions.head(m_prior_ship_count),
                             m_prior_ship_actions);
    eigen_to_tensor<int32_t>(
        m_best_shipyard_actions.head(m_prior_shipyard_count),
        m_prior_shipyard_actions);

    updateRecorderForActionsTaken(m_best_ship_actions.head(latest_ship_count),
                                  m_ship_action_recorder);
    updateRecorderForActionsTaken(
        m_best_shipyard_actions.head(latest_shipyard_count),
        m_shipyard_action_recorder);

    m_prior_ship_count = latest_ship_count;
    m_prior_shipyard_count = latest_shipyard_count;
    m_episodes++;
  }

  inline const Eigen::Ref<const Eigen::ArrayXi> getBestShipActions() const {
    return m_best_ship_actions.head(m_prior_ship_count);
  }
  inline const Eigen::Ref<const Eigen::ArrayXi> getBestShipyardActions() const {
    return m_best_shipyard_actions.head(m_prior_shipyard_count);
  }

  inline void resetState() {
		m_ship_pawn_manager.resetState();
    m_prior_ship_count = 0;
    m_prior_shipyard_count = 0;
    m_prior_ship_actions.zero_();
    m_prior_shipyard_actions.zero_();
    m_final_ship_batch.zero_();
    m_final_shipyard_batch.zero_();
    m_tmp_ship_batch.zero_();
    m_tmp_shipyard_batch.zero_();
  }

private:
  float m_epsilon_decay;
  float m_epsilon_start;
  float m_epsilon_end;
  long m_episodes;
  std::size_t m_prior_ship_count;
  std::size_t m_prior_shipyard_count;
  std::size_t m_multi_step_n;
  DynamicBatch<DeviceType, BoardConfig, ShipModelConfig> m_final_ship_batch;
  DynamicBatch<DeviceType, BoardConfig, ShipyardModelConfig>
      m_final_shipyard_batch;
  DynamicBatch<DeviceType, BoardConfig, ShipModelConfig> m_tmp_ship_batch;
  DynamicBatch<DeviceType, BoardConfig, ShipyardModelConfig>
      m_tmp_shipyard_batch;

  // TODO: size known at compile time
  Eigen::ArrayXi m_best_ship_actions;
  Eigen::ArrayXi m_best_shipyard_actions;
  Eigen::ArrayXi m_cumsum;
  Eigen::ArrayXi m_new_random_actions;
  Eigen::ArrayXf m_ship_action_recorder;
  Eigen::ArrayXf m_shipyard_action_recorder;

  torch::Tensor m_prior_ship_actions;
  torch::Tensor m_prior_shipyard_actions;
  torch::Tensor m_retained_ship_indices;
  torch::Tensor m_retained_shipyard_indices;

  torch::Tensor m_support;

  MultiStepPawnManager<ActorId, DeviceType, ShipRewardEngine, ShipReplayBuffer, ShipModelConfig>
      m_ship_pawn_manager;
};

#endif /* ACTOR_HPP_ */
