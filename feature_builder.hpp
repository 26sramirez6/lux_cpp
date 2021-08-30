#ifndef FEATURE_BUILDER_HPP
#define FEATURE_BUILDER_HPP

#include "board.hpp"
#include "board_config.hpp"
#include "math_util.hpp"
#include "model_config.hpp"
#include "lux/kit.hpp"
#include <torch/torch.h>

template<typename Units, int size>
static void emplace_resources(
	const Units &_units,
	const kit:GameMap &_game_map, 
	torch::Tensor& geometric_) {	
	auto accessor = geometric_.accessor<float, 4>();
	const auto resource_channels = torch::indexing::Slice({0,3,1});
	geometric_.index_put_({resource_channels}, -1);
	for (int y = 0; y < _game_map.height; y++) {
		for (int x = 0; x < _game_map.width; x++) {
			const Cell *cell = _game_map.getCell(x, y);
			if (cell->hasResource()) {
				int i = 0;
				for (const auto& unit : _units) { 
					const int shift_y = size / 2 - unit->pos.y;
					const int shift_x = size / 2 - unit->pos.x;

					switch (cell->resource.type) {
					case ResourceType::wood:
						accessor[i++][0][y + shift_y][x + shift_x] = cell->resource.amount; 
						break;
					case ResourceType::coal:
						accessor[i++][1][y + shift_y][x + shift_x] = cell->resource.amount;
						break;
					case ResourceType::uranium:
						accessor[i++][2][y + shift_y][x + shift_x] = cell->resource.amount; 
						break;
					}
				}
			}
		}
	}
	return resources;
}

template<typename UnitVector>
static torch::Tensor translate_positions(const UnitVector& _units) {
	auto positions = torch::zeros({_units.size() , 2}, torch::dtype(torch::kInt16).requires_grad(false));
	auto accessor = positions.accessor<short, 2>();
	for (int i = 0; i < _units.size(); i++) {
		const auto& unit_ptr = _units[i];
		accessor[i][0] = unit_ptr->pos.x;
		accessor[i][1] = unit_ptr->pos.y;
	}
	return positions;
}


struct VectorizedUnits {
	VectorizedUnits(const Player& _player, const int _reserve) {
		m_workers.reserve(_reserve);
		m_carts.reserve(_reserve);
		m_city_tiles.reserve(_reserve);
		for (int i = 0; i < _player.units.size(); i++) {	
			if (_player.units[i].isWorker()) {
				units.m_workers.push_back(&_player.units[i]);
			} else {
				units.m_carts.push_back(&_player.units[i]);
			}
		}		
		for (auto& kv : _player.cities) {
			const auto& ctiles = kv.second.citytiles;
			for (int i = 0; i < ctiles.size(); i++) {
				units.m_citytiles.push_back(&ctiles[i]);
			}
		} 
	}

	std::vector<Unit const * const> m_workers;
	std::vector<Unit const * const> m_carts;
	std::vector<CityTile const * const> m_city_tiles;
}

template <class Derived> struct FeatureBuilder {

  template <std::size_t size>
  static inline const torch::Tensor &getSpatialTensor() {
    static torch::Tensor spatial(construct_spatial(size));
    return spatial;
  }

  template <typename BoardConfig, typename StateFeatureType>
  static void setStateFeatures(const kit::Agent &_env, StateFeatureType &ftrs_) {
    Derived::template setStateFeaturesImpl<BoardConfig, StateFeatureType>(_game_map, ftrs_);
  }
};


struct WorkerFeatureBuilder : public FeatureBuilder<WorkerFeatureBuilder> {

  template <typename BoardConfig, typename StateFeatures>
  static void setStateFeaturesImpl(const kit::Agent &_env, StateFeatures &ftrs_) {
    torch::NoGradGuard no_grad;
    ftrs_.m_geometric.zero_();
    ftrs_.m_temporal.zero_();
    ftrs_.m_reward_ftrs.m_distances.zero_();
    const float remaining =
        static_cast<float>(BoardConfig::episode_steps - _env.turn) /
        static_cast<float>(BoardConfig::episode_steps);

    ftrs_.m_temporal.index_put_({torch::indexing::Slice(0,
      torch::indexing::None, 1), 0}, remaining);
		
		const GameMap &game_map = _env.map;
		const Player &player = _env.players[_env.id];
		const Player &opponent = _env.players[(_env.id + 1)%2];

		VectorizedUnits units(player, BoardConfig::size*BoardConfig::size);
		emplace_resources(game_map, units.m_workers, ftrs_.m_geometric);
		const int worker_count = units.m_workers.size();
		const int ctile_count = units.m_citytiles.size();

		const auto up_to_worker_count = torch::indexing::Slice(0, worker_count, 1);

    ftrs_.m_geometric.index_put_(
        {up_to_worker_count, 3}, remaining);

		min_max_norm(ftrs_.m_geometric.index({up_to_worker_count, 0}) , 1.f, true);
		if (player.researchedCoal()) {
			min_max_norm(ftrs_.m_geometric.index({up_to_worker_count, 1}) , 1.f, true);
		} else {
			ftrs_.m_geometric.index({up_to_worker_count, 1}).fill_(0.f);
		}

		if (player.researchedUranium()) { 
			min_max_norm(ftrs_.m_geometric.index({up_to_worker_count, 2}) , 1.f, true);
		} else {
			ftrs_.m_geometric.index({up_to_worker_count, 2}).fill_(0.f);
		}

    torch::Tensor worker_cargo(torch::zeros({worker_count}, torch::dtype(torch::kFloat32).requires_grad(false)));
		auto accessor = worker_cargo.accessor<float,1>();
		for (int i = 0; i < worker_count; ++i) {
			const float cargo = static_cast<float>(units.m_workers[i]->getCargoSpaceLeft()) / static_cast<float>(BoardConfig::worker_max_cargo);
			accessor[i] = cargo;
			ftrs_.m_geometric.index_put_({i, 4}, cargo); 
		}

    if (worker_count > 0 && ctile_count > 0) {
      torch::Tensor worker_positions(translate_positions(units.m_workers));
      torch::Tensor ctile_positions(translate_positions(units.m_citytiles));

      const torch::Tensor shift_worker_positions =
          static_cast<int>(BoardConfig::size / 2) - worker_positions;

      const torch::Tensor ending_ctile_positions =
          (torch::tile(ctile_positions, {worker_count, 1}) +
           torch::repeat_interleave(shift_worker_positions, ctile_count, 0)) %
          static_cast<int>(BoardConfig::size);

      const auto spatial_repeated =
          torch::tile(getSpatialTensor<BoardConfig::size>(),
                      {worker_count * ctile_count, 1});

      const auto ending_ctile_positions_interleaved =
          torch::repeat_interleave(ending_ctile_positions,
                                   BoardConfig::size * BoardConfig::size, 0);

      const auto distances = std::get<0>(
          manhattan<BoardConfig::size>(spatial_repeated,
                                       ending_ctile_positions_interleaved)
              .reshape({BoardConfig::size, BoardConfig::size, ctile_count,
                        worker_count})
              .min(2));


      const auto heats = 1 / (distances + 1);

      const auto flipped_heats = torch::flip(heats,
                                             {
                                                 0,
                                             })
                                     .transpose(2, 1)
                                     .transpose(1, 0);

      // +1 to ship_cargo to always have non-zero here
      ftrs_.m_geometric.index_put_(
          {up_to_worker_count, 5},
          (1 - remaining) * flipped_heats *
              ((worker_cargo).unsqueeze(1).unsqueeze(1)));
    }
  }
};

struct CityTileFeatureBuilder : public FeatureBuilder<CityTileFeatureBuilder> {
	template <typename BoardConfig, typename StateFeatures>
  static void setStateFeaturesImpl(const kit::Agent &_env, StateFeatures &ftrs_) {
    torch::NoGradGuard no_grad;
    ftrs_.m_geometric.zero_();
    ftrs_.m_temporal.zero_();
    ftrs_.m_reward_ftrs.m_distances.zero_();
}

struct CartFeatureBuilder : public FeatureBuilder<CartFeatureBuilder> {
	template <typename BoardConfig, typename StateFeatures>
  static void setStateFeaturesImpl(const kit::Agent &_env, StateFeatures &ftrs_) {
    torch::NoGradGuard no_grad;
    ftrs_.m_geometric.zero_();
    ftrs_.m_temporal.zero_();
    ftrs_.m_reward_ftrs.m_distances.zero_();
}

//struct ShipyardFeatureBuilder : public FeatureBuilder<ShipyardFeatureBuilder> {
//
//  template <typename BoardConfig, typename Board, typename StateFeatures>
//  static void setStateFeaturesImpl(const Board &_board, StateFeatures &ftrs_) {
//    torch::NoGradGuard no_grad;
//    ftrs_.m_geometric.zero_();
//    // ftrs_.m_temporal.zero_();
//
//    const float remaining =
//        static_cast<float>(BoardConfig::episode_steps - _board.m_step) /
//        static_cast<float>(BoardConfig::episode_steps);
//    // ftrs_.m_temporal.index_put_({torch::indexing::Slice(0,
//    // torch::indexing::None, 1), 0}, remaining);
//    ftrs_.m_geometric.index_put_(
//        {torch::indexing::Slice(0, torch::indexing::None, 1), 1}, remaining);
//
//    torch::Tensor halite_tensor = _board.getHaliteTensor().reshape(
//        {BoardConfig::size, BoardConfig::size});
//    const float max_halite_cell = halite_tensor.max().item().to<float>();
//    const float min_halite_cell = halite_tensor.min().item().to<float>();
//    const float diff_halite = max_halite_cell - min_halite_cell;
//    halite_tensor.sub_(min_halite_cell);
//    halite_tensor.div_(diff_halite);
//    halite_tensor.mul_(2);
//    halite_tensor.sub_(1);
//    const unsigned ship_count = _board.getShipCount();
//    const unsigned shipyard_count = _board.getShipyardCount();
//
//    if (shipyard_count > 0 && ship_count > 0) {
//      torch::Tensor ship_positions(
//          torch::zeros({ship_count, 2}, torch::dtype(torch::kInt16)));
//      torch::Tensor shipyard_positions(
//          torch::zeros({shipyard_count, 2}, torch::dtype(torch::kInt16)));
//      _board.getShipPositions(ship_positions);
//      _board.getShipyardPositions(shipyard_positions);
//
//      const auto shift_shipyard_positions =
//          static_cast<int>(BoardConfig::size / 2) - shipyard_positions;
//
//      const auto ending_ship_positions =
//          (torch::tile(ship_positions, {shipyard_count, 1}) +
//           torch::repeat_interleave(shift_shipyard_positions, ship_count, 0)) %
//          static_cast<int>(BoardConfig::size);
//
//      const auto spatial_repeated =
//          torch::tile(getSpatialTensor<BoardConfig::size>(),
//                      {ship_count * shipyard_count, 1});
//
//      const auto ending_ship_positions_interleaved = torch::repeat_interleave(
//          ending_ship_positions, BoardConfig::size * BoardConfig::size, 0);
//
//      const auto distances =
//          std::get<0>(manhattan<BoardConfig::size>(
//                          spatial_repeated, ending_ship_positions_interleaved)
//                          .reshape({BoardConfig::size, BoardConfig::size,
//                                    ship_count, shipyard_count})
//                          .min(2));
//
//      const auto heats = 1 / (distances + 1);
//
//      const auto flipped_heats = torch::flip(heats,
//                                             {
//                                                 0,
//                                             })
//                                     .transpose(2, 1)
//                                     .transpose(1, 0);
//
//      ftrs_.m_geometric.index_put_(
//          {torch::indexing::Slice(0, shipyard_count, 1), 0},
//          flipped_heats * (1 - remaining) * _board.getPlayerHalite() /
//              BoardConfig::starting_halite);
//    }
//
//    if (ship_count > 0) {
//      torch::Tensor ship_cargo(torch::zeros({ship_count}));
//      _board.getShipCargo(ship_cargo);
//
//      //            ftrs_.m_temporal.index_put_(
//      //                {torch::indexing::Slice(0,ship_count,1), 1},
//      //                ship_cargo / BoardConfig::starting_halite);
//      //
//      //	    min_max_norm(ftrs_.m_temporal.index({torch::indexing::Slice(0,ship_count,1),
//      //1}), -1.f, true);
//    }
//  }
//};

#endif /* FEATURE_BUILDER_HPP */
