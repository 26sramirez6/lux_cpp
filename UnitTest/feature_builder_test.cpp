#include "feature_builder_test.hpp"

template<typename Board>
void move(int action, Board& board) {
	Eigen::ArrayXi ship_actions(1);
	Eigen::ArrayXi shipyard_actions(1);
	ship_actions(0) = action;
	board.setActions(ship_actions, shipyard_actions);
	board.step();
}


TEST(FeatureBuilder, SimpleExample) {

    auto& re = RandomEngine<float, 123>::getInstance();
    BoardStore<decltype(re), BoardConfig, 1, 1> board_store;

    auto board = board_store.getNextBoard(re);

    
    auto& ship_map = board.getShipMap();
    auto& ship = ship_map.at(0);


    Eigen::ArrayXi ship_actions(1); 
    ship_actions << 5;
    Eigen::ArrayXi shipyard_actions(1);
    shipyard_actions << 0;

    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount()));
    board.step();

    ship_actions(0) = 0;
    shipyard_actions(0) = 1; 
    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount())); 
    board.step();

    ship_actions(0) = 1;
    shipyard_actions(0) = 0; 
    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount())); 
    board.step();
   
    ship_actions(0) = 1;
    shipyard_actions(0) = 0; 
    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount())); 
    board.step();

    ship_actions(0) = 2;
    shipyard_actions(0) = 0; 
    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount())); 
    board.step();

    ship_actions(0) = 1;
    shipyard_actions(0) = 0; 
    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount())); 
    board.step();

    ship_actions(0) = 0;
    shipyard_actions(0) = 0; 
    board.setActions(ship_actions.head(board.getShipCount()), shipyard_actions.head(board.getShipyardCount())); 
    board.step();

    move(2, board);
    std::cout << ship.closer << std::endl;
    move(0, board);
    std::cout << ship.closer << std::endl;
    move(3, board);
    //auto& ship = board.getShip(0);
    std::cout << ship.closer << std::endl;
    //move(1, board);
    //move(0, board);
    //move(3, board);
    board.printBoard();
    //ASSERT_TRUE(true);
}
