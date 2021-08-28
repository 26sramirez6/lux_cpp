#ifndef FEATURE_BUILDER_HPP
#define FEATURE_BUILDER_HPP

#include "board.hpp"
#include "board_config.hpp"
#include "math_util.hpp"
#include <torch/torch.h>


static torch::Tensor construct_resources(const GameMap &_game_map) {
	torch::Tensor resources = torch::zeros(
		{3, _game_map.height, _game_map.width}, 
		torch::dtype(torch::kInt32).requires_grad(false));
	
	auto accessor = resources.accessor<int, 3>();
	for (int y = 0; y < _game_map.height; y++) {
		for (int x = 0; x < _game_map.width; x++) {
			const Cell *cell = game_map.getCell(x, y);
			if (cell->hasResource()) {
				switch (cell->resource.type) {
				case ResourceType::wood:
					accessor[0][y][x] = cell->resource.amount; 
					break;
				case ResourceType::coal:
					accessor[1][y][x] = cell->resource.amount; 
					break;
				case ResourceType::uranium:
					accessor[2][y][x] = cell->resource.amount; 
					break;
				}
				resourceTiles.push_back(cell);
			}
		}
	}
	return resources;
}

template <class Derived> struct FeatureBuilder {

  template <std::size_t size>
  static inline const torch::Tensor &getSpatialTensor() {
    static torch::Tensor spatial(construct_spatial(size));
    return spatial;
  }

  template <typename BoardConfig, typename Board, typename StateFeatureType>
  static void setStateFeatures(const Board &_board, StateFeatureType &ftrs_) {
    Derived::template setStateFeaturesImpl<BoardConfig, Board,
                                           StateFeatureType>(_board, ftrs_);
  }
};

struct WorkerFeatureBuilder : public FeatureBuilder<WorkerFeatureBuilder> {

  template <typename BoardConfig, typename Board, typename StateFeatures>
  static void setStateFeaturesImpl(const Board &_board, StateFeatures &ftrs_) {
    torch::NoGradGuard no_grad;
    ftrs_.m_geometric.zero_();
    // ftrs_.m_temporal.zero_();
    ftrs_.m_reward_ftrs.m_distances.zero_();
    const float remaining =
        static_cast<float>(BoardConfig::episode_steps - _board.m_step) /
        static_cast<float>(BoardConfig::episode_steps);

    // ftrs_.m_temporal.index_put_({torch::indexing::Slice(0,
    // torch::indexing::None, 1), 0}, remaining);
    torch::Tensor halite_tensor = _board.getHaliteTensor().reshape(
        {BoardConfig::size, BoardConfig::size});
    ftrs_.m_geometric.index_put_(
        {torch::indexing::Slice(0, torch::indexing::None, 1), 2}, remaining);

    const float max_halite_cell = halite_tensor.max().item().to<float>();
    const float min_halite_cell = halite_tensor.min().item().to<float>();
    const float diff_halite = max_halite_cell - min_halite_cell;
    halite_tensor.sub_(min_halite_cell);
    halite_tensor.div_(diff_halite);
    halite_tensor.mul_(2);
    halite_tensor.sub_(1);

    const unsigned ship_count = _board.getShipCount();
    const unsigned shipyard_count = _board.getShipyardCount();
    const auto &map = _board.getShipMap();
    int ship_index = 0;

    for (const auto &kv : map) {
      const auto &ship = kv.second;
      const int shift1 = BoardConfig::size / 2 - ship.x;
      const int shift2 = ship.y - BoardConfig::size / 2;

      auto rolled_halite_tensor =
          torch::roll(halite_tensor, {shift1, shift2}, {1, 0});
      ftrs_.m_geometric.index_put_({ship_index++, 0}, rolled_halite_tensor);
    }

    torch::Tensor ship_cargo(torch::zeros({ship_count}));
    _board.getShipCargo(ship_cargo);

    if (shipyard_count > 0 && ship_count > 0) {
      torch::Tensor ship_positions(
          torch::zeros({ship_count, 2}, torch::dtype(torch::kInt16)));
      torch::Tensor shipyard_positions(
          torch::zeros({shipyard_count, 2}, torch::dtype(torch::kInt16)));
      _board.getShipPositions(ship_positions);
      _board.getShipyardPositions(shipyard_positions);

      const torch::Tensor shift_ship_positions =
          static_cast<int>(BoardConfig::size / 2) - ship_positions;

      const torch::Tensor ending_shipyard_positions =
          (torch::tile(shipyard_positions, {ship_count, 1}) +
           torch::repeat_interleave(shift_ship_positions, shipyard_count, 0)) %
          static_cast<int>(BoardConfig::size);

      const auto spatial_repeated =
          torch::tile(getSpatialTensor<BoardConfig::size>(),
                      {ship_count * shipyard_count, 1});

      const auto ending_shipyard_positions_interleaved =
          torch::repeat_interleave(ending_shipyard_positions,
                                   BoardConfig::size * BoardConfig::size, 0);

      const auto distances = std::get<0>(
          manhattan<BoardConfig::size>(spatial_repeated,
                                       ending_shipyard_positions_interleaved)
              .reshape({BoardConfig::size, BoardConfig::size, shipyard_count,
                        ship_count})
              .min(2));

      //                ftrs_.m_reward_ftrs.m_distances.index_put_(
      //                        {torch::indexing::Slice(0, ship_count, 1)},
      //                        std::get<0>(distances.reshape(BoardConfig::size*BoardConfig::size).min(0)));

      const auto heats = 1 / (distances + 1);

      const auto flipped_heats = torch::flip(heats,
                                             {
                                                 0,
                                             })
                                     .transpose(2, 1)
                                     .transpose(1, 0);

      // +1 to ship_cargo to always have non-zero here
      ftrs_.m_geometric.index_put_(
          {torch::indexing::Slice(0, ship_count, 1), 1},
          (1 - remaining) * flipped_heats *
              (((1 + ship_cargo) / max_halite_cell).unsqueeze(1).unsqueeze(1)));
    }

    //        if (ship_count > 0) {
    //            ftrs_.m_temporal.index_put_(
    //                {torch::indexing::Slice(0,ship_count,1), 1},
    //                ship_cargo / BoardConfig::starting_halite);
    //	    min_max_norm(ftrs_.m_temporal.index({torch::indexing::Slice(0,ship_count,1),
    //1}), -1.f, true);
    //        }
  }
};

struct ShipyardFeatureBuilder : public FeatureBuilder<ShipyardFeatureBuilder> {

  template <typename BoardConfig, typename Board, typename StateFeatures>
  static void setStateFeaturesImpl(const Board &_board, StateFeatures &ftrs_) {
    torch::NoGradGuard no_grad;
    ftrs_.m_geometric.zero_();
    // ftrs_.m_temporal.zero_();

    const float remaining =
        static_cast<float>(BoardConfig::episode_steps - _board.m_step) /
        static_cast<float>(BoardConfig::episode_steps);
    // ftrs_.m_temporal.index_put_({torch::indexing::Slice(0,
    // torch::indexing::None, 1), 0}, remaining);
    ftrs_.m_geometric.index_put_(
        {torch::indexing::Slice(0, torch::indexing::None, 1), 1}, remaining);

    torch::Tensor halite_tensor = _board.getHaliteTensor().reshape(
        {BoardConfig::size, BoardConfig::size});
    const float max_halite_cell = halite_tensor.max().item().to<float>();
    const float min_halite_cell = halite_tensor.min().item().to<float>();
    const float diff_halite = max_halite_cell - min_halite_cell;
    halite_tensor.sub_(min_halite_cell);
    halite_tensor.div_(diff_halite);
    halite_tensor.mul_(2);
    halite_tensor.sub_(1);
    const unsigned ship_count = _board.getShipCount();
    const unsigned shipyard_count = _board.getShipyardCount();

    if (shipyard_count > 0 && ship_count > 0) {
      torch::Tensor ship_positions(
          torch::zeros({ship_count, 2}, torch::dtype(torch::kInt16)));
      torch::Tensor shipyard_positions(
          torch::zeros({shipyard_count, 2}, torch::dtype(torch::kInt16)));
      _board.getShipPositions(ship_positions);
      _board.getShipyardPositions(shipyard_positions);

      const auto shift_shipyard_positions =
          static_cast<int>(BoardConfig::size / 2) - shipyard_positions;

      const auto ending_ship_positions =
          (torch::tile(ship_positions, {shipyard_count, 1}) +
           torch::repeat_interleave(shift_shipyard_positions, ship_count, 0)) %
          static_cast<int>(BoardConfig::size);

      const auto spatial_repeated =
          torch::tile(getSpatialTensor<BoardConfig::size>(),
                      {ship_count * shipyard_count, 1});

      const auto ending_ship_positions_interleaved = torch::repeat_interleave(
          ending_ship_positions, BoardConfig::size * BoardConfig::size, 0);

      const auto distances =
          std::get<0>(manhattan<BoardConfig::size>(
                          spatial_repeated, ending_ship_positions_interleaved)
                          .reshape({BoardConfig::size, BoardConfig::size,
                                    ship_count, shipyard_count})
                          .min(2));

      const auto heats = 1 / (distances + 1);

      const auto flipped_heats = torch::flip(heats,
                                             {
                                                 0,
                                             })
                                     .transpose(2, 1)
                                     .transpose(1, 0);

      ftrs_.m_geometric.index_put_(
          {torch::indexing::Slice(0, shipyard_count, 1), 0},
          flipped_heats * (1 - remaining) * _board.getPlayerHalite() /
              BoardConfig::starting_halite);
    }

    if (ship_count > 0) {
      torch::Tensor ship_cargo(torch::zeros({ship_count}));
      _board.getShipCargo(ship_cargo);

      //            ftrs_.m_temporal.index_put_(
      //                {torch::indexing::Slice(0,ship_count,1), 1},
      //                ship_cargo / BoardConfig::starting_halite);
      //
      //	    min_max_norm(ftrs_.m_temporal.index({torch::indexing::Slice(0,ship_count,1),
      //1}), -1.f, true);
    }
  }
};

#endif /* FEATURE_BUILDER_HPP */
