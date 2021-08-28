/*
 * model_config.hpp
 *
 *  Created on: May 8, 2021
 *      Author: saul.ramirez
 */

#ifndef MODEL_CONFIG_HPP_
#define MODEL_CONFIG_HPP_

struct BaseModelConfig {};

struct WorkerModelConfig : public BaseModelConfig {
  constexpr static std::size_t ts_ftr_count = 2;
  constexpr static std::size_t channels = 3;
  constexpr static std::size_t output_size = 8;
};

struct CartModelConfig : public BaseModelConfig {
  constexpr static std::size_t ts_ftr_count = 2;
  constexpr static std::size_t channels = 2;
  constexpr static std::size_t output_size = 6;
};

struct CityTileModelConfig : public BaseModelConfig {
  constexpr static std::size_t ts_ftr_count = 2;
  constexpr static std::size_t channels = 2;
  constexpr static std::size_t output_size = 3;
};


#endif /* MODEL_CONFIG_HPP_ */
