#pragma once

#include "board.hpp"
#include "board_config.hpp"
#include "data_objects.hpp"
#include "hyper_parameters.hpp"
#include <torch/torch.h>
#include <vector>

template <torch::DeviceType DeviceType, typename BoardConfig,
          typename ModelConfig>
class RewardEngine {
public:
  RewardEngine(const HyperParameters &_hp)
      : m_mine_beta(_hp.m_reward_mine_beta),
        m_deposit_beta(_hp.m_reward_deposit_beta),
        m_distance_beta(_hp.m_reward_distance_beta),
        m_discovery_beta(_hp.m_reward_discovery_beta),
        m_mine_time_beta(_hp.m_reward_mine_time_beta *
                         BoardConfig::episode_steps),
        m_deposit_time_beta(_hp.m_reward_deposit_time_beta *
                            BoardConfig::episode_steps),
        m_distance_time_beta(_hp.m_reward_distance_time_beta *
                             BoardConfig::episode_steps),
        m_discovery_time_beta(_hp.m_reward_discovery_time_beta *
                              BoardConfig::episode_steps),
        m_reward_step(1. / BoardConfig::episode_steps),
        m_mine_min_clip(_hp.m_reward_mine_min_clip),
        m_mine_max_clip(_hp.m_reward_mine_max_clip),
        m_deposit_min_clip(_hp.m_reward_deposit_min_clip),
        m_deposit_max_clip(_hp.m_reward_deposit_max_clip),
        m_distance_min_clip(_hp.m_reward_distance_min_clip),
        m_distance_max_clip(_hp.m_reward_distance_max_clip),
        m_discovery_min_clip(_hp.m_reward_discovery_min_clip),
        m_discovery_max_clip(_hp.m_reward_discovery_max_clip),
        m_mine_weights(initializeMineWeights(_hp.m_reward_mine_beta,
                                             _hp.m_reward_mine_time_beta)),
        m_deposit_weights(initializeDepositWeights(
            _hp.m_reward_deposit_beta, _hp.m_reward_deposit_time_beta)),
        m_distance_weights(initializeDistanceWeights(
            _hp.m_reward_distance_beta, _hp.m_reward_distance_time_beta)),
        m_discovery_weights(initializeDiscoveryWeights(
            _hp.m_reward_discovery_beta, _hp.m_reward_discovery_time_beta)),
        m_rewards_on_cpu(torch::zeros({BoardConfig::size * BoardConfig::size},
                                      torch::dtype(torch::kFloat32)
                                          .requires_grad(false)
                                          .device(torch::kCPU))) {}

  template <std::size_t ActorId, typename Board>
  inline void computeRewards(const Board &_current_board,
                             std::unordered_map<int, float> &reward_map_) {
    const float max_halite_cell =
        _current_board.getMaxHaliteCell().template to<float>();
    const float max_halite_mineable =
        max_halite_cell * BoardConfig::collect_rate;

    const unsigned step = _current_board.getStep();
    const auto &retained_ships =
        _current_board.template getRetainedShips<ActorId>();
    const unsigned ship_count = retained_ships.size();

    int i = 0;
    for (const auto &kv : retained_ships) {
      const float mine_reward = clip(
          m_mine_weights[step] * (kv.second->delta_cargo / max_halite_mineable),
          m_mine_min_clip, m_mine_max_clip);

      const float deposit_reward = clip(
          m_deposit_weights[step] * (kv.second->delta_halite / max_halite_cell),
          m_deposit_min_clip, m_deposit_max_clip);

      const float distance_reward = (kv.second->cargo > 0 && kv.second->closer)
                                        ? m_distance_weights[step]
                                        : 0;

      const float discovery_reward = clip(
          static_cast<int>(kv.second->action != ShipAction::NONE) *
              m_discovery_weights[step] *
              (_current_board.getHaliteAtPoint(kv.second->x, kv.second->y) /
               max_halite_cell),
          m_discovery_min_clip, m_discovery_max_clip);
      reward_map_.insert({kv.first, mine_reward + deposit_reward +
                                        distance_reward + discovery_reward});

    }
  }

  template <unsigned ActorId, typename Board, typename BatchType>
  inline void computeRewards(const Board &_current_board, BatchType &batch_) {
    batch_.m_reward.zero_();
    const float max_halite_cell =
        _current_board.getMaxHaliteCell().template to<float>();
    const float max_halite_mineable =
        max_halite_cell * BoardConfig::collect_rate;

    const unsigned step = _current_board.getStep();
    const auto &retained_ships =
        _current_board.template getRetainedShips<ActorId>();
    const unsigned ship_count = retained_ships.size();

    int i = 0;
    auto a = m_rewards_on_cpu.template accessor<float, 1>();
    for (const auto &kv : retained_ships) {
      const float mine_reward = clip(
          m_mine_weights[step] * (kv.second->delta_cargo / max_halite_mineable),
          m_mine_min_clip, m_mine_max_clip);

      const float deposit_reward = clip(
          m_deposit_weights[step] * (kv.second->delta_halite / max_halite_cell),
          m_deposit_min_clip, m_deposit_max_clip);

      const float distance_reward = (kv.second->cargo > 0 && kv.second->closer)
                                        ? m_distance_weights[step]
                                        : 0;

      const float discovery_reward = clip(
          static_cast<int>(kv.second->action != ShipAction::NONE) *
              m_discovery_weights[step] *
              (_current_board.getHaliteAtPoint(kv.second->x, kv.second->y) /
               max_halite_cell),
          m_discovery_min_clip, m_discovery_max_clip);

      a[i++] =
          mine_reward + deposit_reward + distance_reward + discovery_reward;
    }

    const auto slice = torch::indexing::Slice(0, ship_count, 1);
    batch_.m_reward.index_put_(
        {slice}, m_rewards_on_cpu.index({slice}).to(
                     DeviceType, /* nonblocking */ false, /*copy*/ false));

    m_rewards_on_cpu.zero_();
    // std::cout << "reward from batch after zero: " <<
    // batch_.m_reward.index({0}) << std::endl;
  }

  static inline std::vector<float>
  initializeMineWeights(const float _mine_beta, const float _mine_time_beta) {
    std::vector<float> weights(BoardConfig::episode_steps);
    for (int i = 0; i < BoardConfig::episode_steps; ++i) {
      weights[i] = _mine_beta *
                   (std::exp(-static_cast<float>(i) /
                             (_mine_time_beta * BoardConfig::episode_steps)));
    }

    return weights;
  }

  static inline std::vector<float>
  initializeDepositWeights(const float _deposit_beta,
                           const float _deposit_time_beta) {
    std::vector<float> weights(BoardConfig::episode_steps);
    for (int i = 0; i < BoardConfig::episode_steps; ++i) {
      weights[i] =
          _deposit_beta *
          (1 - std::exp(-static_cast<float>(i) /
                        (_deposit_time_beta * BoardConfig::episode_steps)));
    }

    return weights;
  }

  static inline std::vector<float>
  initializeDistanceWeights(const float _distance_beta,
                            const float _distance_time_beta) {
    std::vector<float> weights(BoardConfig::episode_steps);
    for (int i = 0; i < BoardConfig::episode_steps; ++i) {
      weights[i] =
          _distance_beta *
          (1 - std::exp(-static_cast<float>(i) /
                        (_distance_time_beta * BoardConfig::episode_steps)));
    }

    return weights;
  }

  static inline std::vector<float>
  initializeDiscoveryWeights(const float _discovery_beta,
                             const float _discovery_time_beta) {
    std::vector<float> weights(BoardConfig::episode_steps);
    for (int i = 0; i < BoardConfig::episode_steps; ++i) {
      weights[i] =
          _discovery_beta *
          (std::exp(-static_cast<float>(i) /
                    (_discovery_time_beta * BoardConfig::episode_steps)));
    }
    return weights;
  }

private:
  float m_mine_beta;
  float m_deposit_beta;
  float m_distance_beta;
  float m_discovery_beta;
  float m_mine_time_beta;
  float m_deposit_time_beta;
  float m_distance_time_beta;
  float m_discovery_time_beta;
  float m_reward_step;
  float m_mine_min_clip;
  float m_mine_max_clip;
  float m_deposit_min_clip;
  float m_deposit_max_clip;
  float m_distance_min_clip;
  float m_distance_max_clip;
  float m_discovery_min_clip;
  float m_discovery_max_clip;
  std::vector<float> m_mine_weights;
  std::vector<float> m_deposit_weights;
  std::vector<float> m_distance_weights;
  std::vector<float> m_discovery_weights;
  torch::Tensor m_rewards_on_cpu;
};
