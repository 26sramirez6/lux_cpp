#pragma once
#include "board.hpp"
#include <EigenRand/EigenRand>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

template <typename RandomEngine, typename Config, unsigned Chunksize,
          unsigned ActorCount>
class BoardStore {
  static constexpr unsigned half = (Config::size / 2) + 1;
  static constexpr unsigned fourth = half / 4;
  using HalfMatI = Eigen::Array<int, half * Chunksize, half>;
  using HalfMatF = Eigen::Array<float, half * Chunksize, half>;
  using FourthMatF = Eigen::Array<float, fourth * Chunksize, fourth>;

private:
  inline void generateBoards(RandomEngine &random_engine_) {
    auto &urng = random_engine_.getGenerator();
    Eigen::Rand::ExtremeValueGen<float> gumbel1_gen(0, 300);
    Eigen::Rand::ExtremeValueGen<float> gumbel2_gen(0, 500);
    Eigen::Rand::BinomialGen<int> binomial_gen(1, .5);
    HalfMatF gumbel1 =
        gumbel1_gen.generate<HalfMatF>(half * Chunksize, half, urng);
    FourthMatF gumbel2 =
        gumbel2_gen.generate<FourthMatF>(fourth * Chunksize, fourth, urng);
    HalfMatI binomial =
        binomial_gen.generate<HalfMatI>(half * Chunksize, half, urng);

    const auto quartile_add = gumbel1.max(0) * binomial.template cast<float>();

    for (int i = 0; i < Chunksize; ++i) {
      m_boards[i].populate(
          quartile_add.template middleRows<half>(i * half),
          (gumbel2.template middleRows<fourth>(i * fourth)).max(0));
    }
  }

public:
  inline Board<Config, ActorCount> &getNextBoard(RandomEngine &random_engine_) {
    if (m_current_index % Chunksize == 0) {
      delete[] m_boards;
      m_boards = new Board<Config, ActorCount>[Chunksize];
      generateBoards(random_engine_);
      m_current_index = 0;
    }
    auto &ret = m_boards[m_current_index];
    m_current_index++;
    return ret;
  }

private:
  Board<Config, ActorCount> *m_boards = nullptr;
  unsigned m_current_index = 0;
};
