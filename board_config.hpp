#pragma once

struct BoardConfig {
  static constexpr std::size_t size = 12;
  static constexpr std::size_t episode_steps = 361;
  static constexpr std::size_t actor_count = 1;
	static constexpr std::size_t worker_max_cargo = 100;
	static constexpr std::size_t cart_max_cargo = 2000;
	static constexpr std::size_t max_wood_collect = 20;
	static constexpr std::size_t max_coal_collect = 5;
	static constexpr std::size_t max_uranium_collect = 2;
};
