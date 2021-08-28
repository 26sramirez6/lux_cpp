#pragma once

#include "math_util.hpp"
#include "ship.hpp"
#include "template_util.hpp"
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <map>
#include <sstream>
#include <torch/torch.h>
#include <unordered_set>

template <std::size_t ActorCount> struct StepReturn {
  StepReturn(const std::size_t _step) : m_step(_step){};
  const std::size_t m_step;
  tuple_n_t<ActorCount, std::vector<int>> m_retained_ships;
  tuple_n_t<ActorCount, std::vector<int>> m_converted_ships;
  tuple_n_t<ActorCount, std::vector<int>> m_collided_ships;
};

template <typename BoardConfig, std::size_t ActorCount> struct Board {
  static constexpr std::size_t half = (BoardConfig::size / 2) + 1;
  static constexpr std::size_t fourth = half / 4;
  static constexpr std::size_t p0_starting_index =
      BoardConfig::size * (BoardConfig::size / 2) + (BoardConfig::size / 2);
  using GridMat = Eigen::Array<float, BoardConfig::size, BoardConfig::size>;
  using QuartileMatF = Eigen::Array<float, half, half>;
  using CornerMatF = Eigen::Array<float, fourth, fourth>;
  using QuartileMatI = Eigen::Array<int, half, half>;
  using Positions = Eigen::Array<int, BoardConfig::size * BoardConfig::size, 2>;

  Board()
      : m_step(0), m_global_ship_id(0), m_global_shipyard_id(0),
        m_p0_halite(BoardConfig::starting_halite),
        m_halite_tensor(
            torch::zeros(BoardConfig::size * BoardConfig::size,
                         torch::dtype(torch::kFloat32).requires_grad(false))),
        m_halite_tensor_a(m_halite_tensor.data_ptr<float>(),
                          m_halite_tensor.sizes().data(),
                          m_halite_tensor.strides().data()),
        m_has_ship(BoardConfig::size * BoardConfig::size, 0),
        m_has_shipyard(BoardConfig::size * BoardConfig::size, 0), m_ship_map(),
        m_shipyard_map(),

        m_retained_ship_count(), m_retained_ships(), m_converted_ships(),
        m_collided_ships() {}

  Board(const Board<BoardConfig, ActorCount> &_other)
      : m_step(_other.m_step), m_global_ship_id(_other.m_global_ship_id),
        m_global_shipyard_id(_other.m_global_shipyard_id),
        m_p0_halite(_other.m_p0_halite),
        m_halite_tensor(_other.m_halite_tensor.detach().clone()),
        m_halite_tensor_a(m_halite_tensor.data_ptr<float>(),
                          m_halite_tensor.sizes().data(),
                          m_halite_tensor.strides().data()),
        m_has_ship(_other.m_has_ship), m_has_shipyard(_other.m_has_shipyard),
        m_ship_map(_other.m_ship_map), m_shipyard_map(_other.m_shipyard_map),

        m_retained_ship_count(), m_retained_ships(), m_converted_ships(),
        m_collided_ships(), m_retained_shipyards(), m_collided_shipyards() {}

  Board<BoardConfig, ActorCount> &
  operator=(const Board<BoardConfig, ActorCount> &_other) {
    if (this == &_other) {
      return *this;
    }

    m_step = _other.m_step;
    m_global_ship_id = _other.m_global_ship_id;
    m_global_shipyard_id = _other.m_global_shipyard_id;
    m_p0_halite = _other.m_p0_halite;
    m_halite_tensor = _other.m_halite_tensor.detach().clone();
    m_halite_tensor_a = m_halite_tensor.accessor<float, 1>();

    m_has_ship = _other.m_has_ship;
    m_has_shipyard = _other.m_has_shipyard;
    m_ship_map = _other.m_ship_map;
    m_shipyard_map = _other.m_shipyard_map;

    return *this;
  }

  inline void populate(const Eigen::Ref<const QuartileMatF> &_quartile_add,
                       const Eigen::Ref<const CornerMatF> &_corner_add) {

    QuartileMatF quartile = QuartileMatF::Zero();
    for (int i = 0; i < half; ++i) {
      quartile(rand() % (half - 1), rand() % (half - 1)) = i * i;
      quartile(rand_in_range(half / 2, half - 1),
               rand_in_range(half / 2, half - 1)) = i * i;
    }

    for (int c = 0; c < half; ++c) {
      for (int r = 0; r < half; ++r) {
        const float value = quartile(r, c);
        if (value == 0) {
          continue;
        }

        const int radius =
            std::min(static_cast<int>(std::round(std::sqrt(value / half))), 1);
        for (int c2 = c - radius + 1; c2 < c + radius; ++c2) {
          for (int r2 = r - radius + 1; r2 < r + radius; ++r2) {
            const float distance = std::sqrt(std::pow(std::abs(r2 - r), 2) +
                                             std::pow(std::abs(c2 - c), 2));
            quartile(r2, c2) +=
                std::pow(value / std::max(1.f, distance), distance);
          }
        }
      }
    }

    quartile += _quartile_add;
    quartile.template bottomRightCorner<fourth, fourth>() += _corner_add;
    const float quartile_sum = quartile.sum();
    const float multiplier = BoardConfig::starting_halite / quartile_sum / 4.;
    quartile *= multiplier;

    for (int c = 0; c < half; ++c) {
      for (int r = 0; r < half; ++r) {
        m_halite_tensor_a[BoardConfig::size * r + c] = quartile(r, c);
        m_halite_tensor_a[BoardConfig::size * r + (BoardConfig::size - c - 1)] =
            quartile(r, c);
        m_halite_tensor_a[BoardConfig::size * (BoardConfig::size - 1) -
                          (BoardConfig::size * r) + c] = quartile(r, c);
        m_halite_tensor_a[BoardConfig::size * (BoardConfig::size - 1) -
                          (BoardConfig::size * r) +
                          (BoardConfig::size - c - 1)] = quartile(r, c);
      }
    }

    Ship p0_start_ship{0, 0, 0, 0.f, 0.f, 0.f, false, ShipAction::NONE};

    indexToPoint(p0_starting_index, p0_start_ship.x, p0_start_ship.y);
    m_has_ship[p0_starting_index] = true;
    m_ship_map.emplace(p0_start_ship.id, p0_start_ship);
    ++m_global_ship_id;
  }

  static inline std::size_t pointToIndex(uint8_t x, uint8_t y) {
    return (BoardConfig::size - y - 1) * BoardConfig::size + x;
  }

  static inline void indexToPoint(const std::size_t _index, uint8_t &x_,
                                  uint8_t &y_) {
    auto dv =
        std::div(static_cast<int>(_index), static_cast<int>(BoardConfig::size));
    x_ = dv.rem;
    y_ = BoardConfig::size - dv.quot - 1;
  }

  inline float getPlayerHalite() const { return m_p0_halite; }
  inline bool pointHasShip(uint8_t x, uint8_t y) const {
    return m_has_ship[pointToIndex(x, y)];
  }
  inline bool pointHasShipyard(uint8_t x, uint8_t y) const {
    return m_has_shipyard[pointToIndex(x, y)];
  }
  inline bool indexHasShip(std::size_t index) const {
    return m_has_ship[index];
  }
  inline bool indexHasShipyard(std::size_t index) const {
    return m_has_shipyard[index];
  }
  inline torch::Tensor getHaliteTensor() const {
    return m_halite_tensor.detach().clone();
  }
  inline std::size_t getShipCount() const { return m_ship_map.size(); }
  inline std::size_t getShipyardCount() const { return m_shipyard_map.size(); }
  inline std::size_t getStep() const { return m_step; }
  inline const Ship &getShip(int key) const { return m_ship_map.at(key); }
  inline torch::Scalar getMaxHaliteCell() const {
    return m_halite_tensor.max().item();
  }
  inline const std::map<int, Ship> &getShipMap() const { return m_ship_map; }
  inline const std::map<int, Shipyard> &getShipyardMap() const {
    return m_shipyard_map;
  }
  inline float getHaliteAtPoint(const uint8_t x, const uint8_t y) const {
    return m_halite_tensor_a[pointToIndex(x, y)];
  }

  template <std::size_t ActorId> inline std::size_t getActorShipCount() const {
    return m_ship_map.size();
  }

  inline bool isTerminal() const {
    return m_step == BoardConfig::episode_steps - 1;
  }

  template <std::size_t ActorId>
  inline const std::map<int, const Ship *> &getRetainedShips() const {
    return std::get<ActorId>(m_retained_ships);
  }

  template <std::size_t ActorId>
  inline const std::map<int, const Ship *> &getCollidedShips() const {
    return std::get<ActorId>(m_collided_ships);
  }

  template <std::size_t ActorId>
  inline const std::map<int, const Ship *> &getConvertedShips() const {
    return std::get<ActorId>(m_converted_ships);
  }

  template <std::size_t ActorId>
  inline const std::map<int, const Shipyard *> &getRetainedShipyards() const {
    return std::get<ActorId>(m_retained_shipyards);
  }

  template <std::size_t ActorId>
  inline const std::map<int, const Shipyard *> &getCollidedShipyards() const {
    return std::get<ActorId>(m_collided_shipyards);
  }

  inline void getShipCargo(torch::Tensor &ship_cargo_) const {
    auto accessor = ship_cargo_.accessor<float, 1>();
    int i = 0;
    for (const auto &kv : m_ship_map) {
      accessor[i++] = kv.second.cargo;
    }
  }

  inline void getShipPositions(torch::Tensor &positions_) const {
    int i = 0;
    auto accessor = positions_.accessor<short, 2>();
    for (const auto &kv : m_ship_map) {
      accessor[i][0] = kv.second.x;
      accessor[i][1] = kv.second.y;
      i++;
    }
  }

  inline void getShipyardPositions(torch::Tensor &positions_) const {
    int i = 0;
    auto accessor = positions_.accessor<short, 2>();
    for (const auto &kv : m_shipyard_map) {
      accessor[i][0] = kv.second.x;
      accessor[i][1] = kv.second.y;
      i++;
    }
  }

  inline float getTotalShipCargo() const {
    float cargo = 0;
    for (const auto &kv : m_ship_map) {
      cargo += kv.second.cargo;
    }
    return cargo;
  }

  inline bool isCloserToShipyard(int x1, int y1, int x2, int y2) const {
    if (m_shipyard_map.empty()) {
      return false;
    }
    // std::cout << "(" << x1 << ", " << y1 << ") , (" << x2 << ", " << y2 <<
    // ")" <<  std::endl;
    int prior_min = std::numeric_limits<int>::max();
    for (const auto &kv : m_shipyard_map) {
      const auto &shipyard = kv.second;
      const int distance =
          manhattan<BoardConfig::size>(x1, y1, shipyard.x, shipyard.y);
      prior_min = std::min(prior_min, distance);
    }
    int current_min = std::numeric_limits<int>::max();
    for (const auto &kv : m_shipyard_map) {
      const auto &shipyard = kv.second;
      const int distance =
          manhattan<BoardConfig::size>(x2, y2, shipyard.x, shipyard.y);
      current_min = std::min(current_min, distance);
    }
    // std::cout << "prior_min: " << prior_min << std::endl;
    // std::cout << "current_min: " << current_min << std::endl;
    return current_min < prior_min;
  }

  inline void step() {
    std::unordered_set<int> just_added_shipyards;
    auto &converted_ships = std::get<0>(m_converted_ships);
    auto &retained_ships = std::get<0>(m_retained_ships);
    auto &retained_shipyards = std::get<0>(m_retained_shipyards);
    auto &collided_ships = std::get<0>(m_collided_ships);
    auto &collided_shipyards = std::get<0>(m_collided_shipyards);

    retained_ships.clear();
    retained_shipyards.clear();
    converted_ships.clear();
    collided_ships.clear();
    collided_shipyards.clear();
    for (auto &kv : m_ship_map) {
      auto &ship = kv.second;
      const std::size_t index = pointToIndex(ship.x, ship.y);
      ship.delta_cargo = 0.f;
      ship.delta_halite = 0.f;
      ship.closer = false;
      switch (ship.action) {
      case ShipAction::NONE: {
        const float delta =
            m_halite_tensor_a[index] * BoardConfig::collect_rate;
        m_halite_tensor_a[index] -= delta;
        ship.cargo += delta;
        ship.delta_cargo = delta;
      } break;
      case ShipAction::CONVERT:
        m_p0_halite -= BoardConfig::convert_cost;
        m_has_ship[index] = false;
        m_has_shipyard[index] = true;
        converted_ships.emplace(ship.id, &ship);
        m_shipyard_map.emplace(m_global_shipyard_id,
                               Shipyard{m_global_shipyard_id, ship.x, ship.y,
                                        ShipyardAction::NONE});
        just_added_shipyards.insert(m_global_shipyard_id);
        ++m_global_shipyard_id;
        break;
      case ShipAction::MOVE_NORTH: {
        const uint8_t update =
            (ship.y == BoardConfig::size - 1) ? 0 : ship.y + 1;
        ship.closer = isCloserToShipyard(ship.x, ship.y, ship.x, update);
        ship.y = update;
        m_has_ship[index] = false;
				auto new_index = pointToIndex(ship.x, ship.y);
				if (m_has_ship[new_index]) {
					collided_ships.emplace(ship.id, &ship);
				}
        m_has_ship[new_index] = true;
      } break;
      case ShipAction::MOVE_EAST: {
        const uint8_t update =
            (ship.x == BoardConfig::size - 1) ? 0 : ship.x + 1;
        ship.closer = isCloserToShipyard(ship.x, ship.y, update, ship.y);
        ship.x = update;
        m_has_ship[index] = false;
				auto new_index = pointToIndex(ship.x, ship.y);
				if (m_has_ship[new_index]) {
					collided_ships.emplace(ship.id, &ship);
				}
        m_has_ship[new_index] = true;
      } break;
      case ShipAction::MOVE_SOUTH: {
        const uint8_t update =
            (ship.y == 0) ? BoardConfig::size - 1 : ship.y - 1;
        ship.closer = isCloserToShipyard(ship.x, ship.y, ship.x, update);
        ship.y = update;
        m_has_ship[index] = false;
				auto new_index = pointToIndex(ship.x, ship.y);
				if (m_has_ship[new_index]) {
					collided_ships.emplace(ship.id, &ship);
				}
        m_has_ship[new_index] = true;
      } break;
      case ShipAction::MOVE_WEST: {
        const uint8_t update =
            (ship.x == 0) ? BoardConfig::size - 1 : ship.x - 1;
        ship.closer = isCloserToShipyard(ship.x, ship.y, update, ship.y);
        ship.x = update;
        m_has_ship[index] = false;
				auto new_index = pointToIndex(ship.x, ship.y);
				if (m_has_ship[new_index]) {
					collided_ships.emplace(ship.id, &ship);
				}
        m_has_ship[new_index] = true;
      } break;
      }

      // std::cout << "closer?: " << ship.closer << std::endl;
      if (m_shipyard_map.size() > 0) {
        const auto &shipyard = m_shipyard_map.at(0);
        if (ship.x == shipyard.x && ship.y == shipyard.y) {
          m_p0_halite += ship.cargo;
          ship.delta_cargo = 0;
          ship.cargo = 0;
        }
      }

      retained_ships.emplace(ship.id, &ship);
    }

    for (auto &kv : m_shipyard_map) {
      auto &shipyard = kv.second;
      const std::size_t index = pointToIndex(shipyard.x, shipyard.y);
      if (shipyard.action == ShipyardAction::SPAWN) {
        m_p0_halite -= BoardConfig::spawn_cost;
        m_has_ship[index] = true;
        m_ship_map.emplace(m_global_ship_id,
                           Ship{m_global_ship_id, shipyard.x, shipyard.y, 0.f,
                                0.f, 0.f, false, ShipAction::NONE});
        m_global_ship_id++;
      }
      shipyard.action = ShipyardAction::NONE;
      m_halite_tensor_a[index] = 0;
    }

    // TODO: handle collisions
    for (const auto &kv : converted_ships) {
      m_ship_map.erase(kv.second->id);
    }
		
		for (const auto &kv : collided_ships) {
			m_ship_map.erase(kv.second->id);
		}

    m_halite_tensor *= (1 + BoardConfig::regen_rate);
    m_halite_tensor.clamp_max_(BoardConfig::max_cell_halite);
    m_step++;
  }

  void printBoard() const {
    std::stringstream ss;
    ss << "H: " << getPlayerHalite() << ", C: " << getTotalShipCargo()
       << std::endl;

    for (int y = 0; y < BoardConfig::size; ++y) {
      for (int x = 0; x < BoardConfig::size; ++x) {
        const std::size_t index = pointToIndex(x, BoardConfig::size - y - 1);
        ss << "|";
        if (m_has_ship[index]) {
          ss << "a";
        } else {
          ss << " ";
        }

        const int normalized_halite = static_cast<int>(
            9 * m_halite_tensor_a[index] / BoardConfig::max_cell_halite);
        ss << normalized_halite;
        if (m_has_shipyard[index]) {
          ss << "A";
        } else {
          ss << " ";
        }
      }
      ss << "|\n";
    }
    std::cout << ss.str() << std::endl;
  }

  void setActions(const Eigen::Ref<const Eigen::ArrayXi> &_ship_actions,
                  const Eigen::Ref<const Eigen::ArrayXi> &_shipyard_actions) {
    assert(_ship_actions.size() == m_ship_map.size());
    assert(_shipyard_actions.size() == m_shipyard_map.size());
    // std::cout << "ship_actions" << std::endl;
    // std::cout << _ship_actions << std::endl;
    {
      int i = 0;
      for (auto &kv : m_ship_map) {
        kv.second.action = static_cast<ShipAction>(_ship_actions(i));
        ++i;
      }
    }
    {
      int i = 0;
      for (auto &kv : m_shipyard_map) {
        kv.second.action = static_cast<ShipyardAction>(_shipyard_actions(i));
        ++i;
      }
    }
  }

  std::size_t m_step;
  int m_global_ship_id;
  int m_global_shipyard_id;
  float m_p0_halite;
  torch::Tensor m_halite_tensor;
  torch::TensorAccessor<float, 1> m_halite_tensor_a;
  std::vector<bool> m_has_ship;
  std::vector<bool> m_has_shipyard;
  std::map<int, Ship> m_ship_map;
  std::map<int, Shipyard> m_shipyard_map;

  std::size_t m_retained_ship_count;
  tuple_n_t<ActorCount, std::map<int, Ship const *>> m_retained_ships;
  tuple_n_t<ActorCount, std::map<int, Ship const *>> m_converted_ships;
  tuple_n_t<ActorCount, std::map<int, Ship const *>> m_collided_ships;
  tuple_n_t<ActorCount, std::map<int, Shipyard const *>> m_retained_shipyards;
  tuple_n_t<ActorCount, std::map<int, Shipyard const *>> m_collided_shipyards;
};
