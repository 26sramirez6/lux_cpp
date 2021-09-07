#pragma once

struct BoardConfig {
  static constexpr int size = 12;
  static constexpr int episode_steps = 361;
  static constexpr int actor_count = 1;
	static constexpr float worker_max_cargo = 100;
	static constexpr float cart_max_cargo = 2000;
	static constexpr float max_wood_collect = 20;
	static constexpr float wood_to_fuel = 1;
	static constexpr float max_coal_collect = 5;
	static constexpr float coal_to_fuel = 10;
	static constexpr float max_uranium_collect = 2;
	static constexpr float uranium_to_fuel = 40;
};
