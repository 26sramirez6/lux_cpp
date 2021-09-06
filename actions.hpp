#ifndef ACTIONS_HPP_
#define ACTIONS_HPP_

enum class WorkerActions {
	CENTER,
	NORTH,
	EAST,
	SOUTH,
	WEST,
	//PILLAGE,
	//TRANSFER,
	BUILD,
	Count
};

enum class CartActions {
	CENTER,
	NORTH,
	EAST,
	SOUTH,
	WEST,
	TRANSFER,
	Count
};

enum class CityTileActions {
	NONE,
	BUILD_WORKER,
	//BUILD_CART,
	RESEARCH,
	Count
};

#endif
