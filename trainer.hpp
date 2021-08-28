/*
 * Trainer.hpp
 *
 *  Created on: May 15, 2021
 *      Author: saul.ramirez
 */

#ifndef TRAINER_HPP_
#define TRAINER_HPP_
#include "actor.hpp"
#include "board_config.hpp"
#include "board_store.hpp"
#include "dqn.hpp"
#include "feature_builder.hpp"
#include "hyper_parameters.hpp"
#include "math_util.hpp"
#include "model_config.hpp"
#include "model_trainer.hpp"
#include "random_engine.hpp"
#include "replay_buffer.hpp"
#include "reward_engine.hpp"
#include "template_util.hpp"
#include <tuple>

template <typename Board> static inline void pregame_process(Board &board_) {
  Eigen::ArrayXi start_ship_actions(1);
  start_ship_actions << 5;
  Eigen::ArrayXi start_shipyard_actions(1);
  start_shipyard_actions << 0;
  board_.setActions(start_ship_actions, start_shipyard_actions);
  board_.step();
  start_ship_actions(0) = 0;
  start_shipyard_actions(0) = 1;
  board_.setActions(start_ship_actions, start_shipyard_actions);
  board_.step();
}

template <unsigned Iterations, unsigned Chunksize, unsigned ActorCount,
          torch::DeviceType DeviceType, typename RandomEngine>
struct Trainer {
  using ShipBatch = DynamicBatch<DeviceType, BoardConfig, ShipModelConfig>;
  using ShipyardBatch =
      DynamicBatch<DeviceType, BoardConfig, ShipyardModelConfig>;
  using ShipExample = SingleExample<DeviceType, BoardConfig, ShipModelConfig>;
  using ShipyardExample =
      SingleExample<DeviceType, BoardConfig, ShipyardModelConfig>;
  using ShipReplayBuffer = ReplayBuffer<DeviceType, ShipBatch, ShipExample>;
  using ShipyardReplayBuffer =
      ReplayBuffer<DeviceType, ShipyardBatch, ShipyardExample>;
  using ShipRewardEngine =
      RewardEngine<DeviceType, BoardConfig, ShipModelConfig>;
  using ShipyardRewardEngine =
      RewardEngine<DeviceType, BoardConfig, ShipyardModelConfig>;

  template <std::size_t ActorId>
  using ActorType =
      Actor<ActorId, DeviceType, BoardConfig, ShipModelConfig,
            ShipyardModelConfig, ShipFeatureBuilder, ShipyardFeatureBuilder,
            ShipRewardEngine, ShipyardRewardEngine, ShipReplayBuffer,
            ShipyardReplayBuffer, RandomEngine>;

  using Actors = std::tuple<ActorType<0>>;
  static_assert(ActorCount == 1, "Unsupported");
  static float train(const HyperParameters &_hyper_parameters,
                     RandomEngine &random_engine_) {
    BigDQN ship_dqn(ShipModelConfig::channels, BoardConfig::size,
                    static_cast<uint64_t>(ShipAction::Count),
                    _hyper_parameters.m_nn_std_init,
                    _hyper_parameters.m_nn_atom_count,
                    _hyper_parameters.m_nn_v_min, _hyper_parameters.m_nn_v_max);
    ship_dqn.to(DeviceType);
    SmallDQN shipyard_dqn(ShipyardModelConfig::channels, BoardConfig::size,
                          static_cast<uint64_t>(ShipyardAction::Count));
    shipyard_dqn.to(DeviceType);

    ModelTrainer<decltype(ship_dqn), DeviceType,
                 static_cast<std::size_t>(ShipAction::Count)>
    ship_model_trainer(
        _hyper_parameters, ship_dqn, unsigned(ShipModelConfig::channels),
        unsigned(BoardConfig::size), static_cast<uint64_t>(ShipAction::Count),
        _hyper_parameters.m_nn_std_init, _hyper_parameters.m_nn_atom_count,
        _hyper_parameters.m_nn_v_min, _hyper_parameters.m_nn_v_max);

    ModelTrainer<decltype(shipyard_dqn), DeviceType,
                 static_cast<std::size_t>(ShipyardAction::Count)>
    shipyard_model_trainer(_hyper_parameters, shipyard_dqn,
                           unsigned(ShipyardModelConfig::channels),
                           unsigned(BoardConfig::size),
                           static_cast<uint64_t>(ShipyardAction::Count));

    ShipFeatureBuilder ship_feature_builder;
    ShipyardFeatureBuilder shipyard_feature_builder;

    ShipReplayBuffer ship_replay_buffer(
        _hyper_parameters.m_replay_capacity,
        _hyper_parameters.m_replay_batch_size, _hyper_parameters.m_replay_alpha,
        _hyper_parameters.m_replay_beta, _hyper_parameters.m_replay_beta_decay);

    ShipyardReplayBuffer shipyard_replay_buffer(
        _hyper_parameters.m_replay_capacity,
        _hyper_parameters.m_replay_batch_size, _hyper_parameters.m_replay_alpha,
        _hyper_parameters.m_replay_beta, _hyper_parameters.m_replay_beta_decay);

    ShipRewardEngine ship_reward_engine(_hyper_parameters);
    ShipyardRewardEngine shipyard_reward_engine(_hyper_parameters);

    BoardStore<RandomEngine, BoardConfig, Chunksize, ActorCount> board_store;

    auto actors = tuple_builder<Actors>::create(
        _hyper_parameters.m_actor_epsilon_decay,
        _hyper_parameters.m_actor_epsilon_start,
        _hyper_parameters.m_actor_epsilon_end,
        _hyper_parameters.m_nn_atom_count, _hyper_parameters.m_nn_v_min,
        _hyper_parameters.m_nn_v_max, _hyper_parameters.m_nn_step_size,
        _hyper_parameters.m_nn_gamma, _hyper_parameters.m_replay_batch_size);

    std::size_t frame = 0;
    // at::InferenceMode guard(false);
    // at::AutoNonVariableTypeMode guard(false);
    // at::AutoDispatchBelowADInplaceOrView guard(false);
    for (int i = 0; i < Iterations; ++i) {
      Board<BoardConfig, ActorCount> &board =
          board_store.getNextBoard(random_engine_);
      pregame_process(board);

      for (int episode = board.getStep(); episode < BoardConfig::episode_steps;
           ++episode, ++frame) {
        auto &actor0 = std::get<0>(actors);
        actor0.processEpisode(board, ship_dqn, shipyard_dqn, ship_replay_buffer,
                              shipyard_replay_buffer, ship_reward_engine,
                              shipyard_reward_engine, random_engine_);

        if (frame > _hyper_parameters.m_replay_capacity) {
          ship_model_trainer.train(frame, ship_replay_buffer, random_engine_);
          // shipyard_model_trainer.train(frame, shipyard_replay_buffer,
          // random_engine_);
        }

        board.setActions(actor0.getBestShipActions(),
                         actor0.getBestShipyardActions());
        std::cout << "episode " << episode << " completed" << std::endl;
        // board.printBoard();
        board.step();
        board.printBoard();
      }
      std::cout << "game " << i
                << " completed. Total halite: " << board.getPlayerHalite()
                << ", Total cargo: " << board.getTotalShipCargo() << std::endl;

      std::get<0>(actors).resetState();
    }
    return 0;
  }
};

#endif /* TRAINER_HPP_ */
