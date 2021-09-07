/*
 * hyper_parameters.hpp
 *
 *  Created on: May 15, 2021
 *      Author: saul.ramirez
 */

#ifndef HYPER_PARAMETERS_HPP_
#define HYPER_PARAMETERS_HPP_

struct HyperParameters {
  static constexpr std::size_t m_replay_capacity = 15000;
  static constexpr std::size_t m_replay_batch_size = 32;
  static constexpr float m_replay_alpha = 0.5;
  static constexpr float m_replay_beta = 0.3;
  static constexpr float m_replay_beta_decay = 50000;

  static constexpr float m_actor_epsilon_decay = 15000;
  static constexpr float m_actor_epsilon_start = 1;
  static constexpr float m_actor_epsilon_end = .01;

  static constexpr float m_nn_lr = .01;
  static constexpr std::size_t m_nn_step_size = 6;
  static constexpr float m_nn_momentum = .9;
  static constexpr float m_nn_weight_decay = 5e-4;
  static constexpr float m_nn_gamma = .99;
  static constexpr float m_nn_std_init = 0.1;
  static constexpr float m_nn_v_min = 0;
  static constexpr float m_nn_v_max = 40.;
  static constexpr std::size_t m_nn_atom_count = 75;

  static constexpr std::size_t m_nn_target_model_update = 5000;

  static constexpr float m_reward_mine_beta = 2.;
  static constexpr float m_reward_deposit_beta = 0.;   // 3.;
  static constexpr float m_reward_distance_beta = 0.;  //.05;
  static constexpr float m_reward_discovery_beta = 0.; // 2.;
  static constexpr float m_reward_mine_time_beta = .5;
  static constexpr float m_reward_deposit_time_beta = .5;
  static constexpr float m_reward_distance_time_beta = .5;
  static constexpr float m_reward_discovery_time_beta = .25;
  static constexpr float m_reward_mine_min_clip = 0.;
  static constexpr float m_reward_mine_max_clip = 5.;
  static constexpr float m_reward_deposit_min_clip = 0.;
  static constexpr float m_reward_deposit_max_clip = 3.;
  static constexpr float m_reward_distance_min_clip = 0.;
  static constexpr float m_reward_distance_max_clip = 1.;
  static constexpr float m_reward_discovery_min_clip = 0.;
  static constexpr float m_reward_discovery_max_clip = .25;
};

#endif /* HYPER_PARAMETERS_HPP_ */
