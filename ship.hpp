#pragma once
#include <cinttypes>

enum class ShipAction {
  NONE,
  MOVE_NORTH,
  MOVE_EAST,
  MOVE_SOUTH,
  MOVE_WEST,
  CONVERT,
  Count
};

enum class ShipyardAction { NONE, SPAWN, Count };

struct Ship {
  //	Ship() : id(0), x(0), y(0), cargo(0),
  //			delta_cargo(0), delta_halite(0),
  //			closer(false), action(ShipAction::NONE) {}

  int id;
  uint8_t x;
  uint8_t y;
  float cargo;
  float delta_cargo;
  float delta_halite;
  bool closer;
  ShipAction action;
};

struct Shipyard {
  //	Shipyard() : id(0), x(0), y(0), action(ShipyardAction::NONE) {};
  int id;
  uint8_t x;
  uint8_t y;
  ShipyardAction action;
};
