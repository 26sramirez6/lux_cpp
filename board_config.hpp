#pragma once

struct BoardConfig {
  static constexpr std::size_t size = 12;
  static constexpr std::size_t episode_steps = 361;
  static constexpr std::size_t actor_count = 1;
	static constexpr std::size_t worker_max_cargo = 100;
	static constexpr std::size_t cart_max_cargo = 2000;
};
