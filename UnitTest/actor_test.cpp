#include "actor_test.hpp"
#include <cstdint>
#include <torch/torch.h>

template<typename Board>
static inline void pregame_process(Board& board_) {
    Eigen::ArrayXi start_ship_actions(1);
    start_ship_actions << 5;
    Eigen::ArrayXi start_shipyard_actions(1);
    start_shipyard_actions << 0;
    board_.setActions(start_ship_actions, start_shipyard_actions);
    board_.step();
    start_ship_actions(0) = 0;
    start_shipyard_actions(0) = 1;
    board_.setActions(start_ship_actions, start_shipyard_actions);
    board_.step();
}


TEST(MultiStepPawnManager, SimpleQueueWithTensor) {

	std::queue<torch::Tensor> q;
	torch::Tensor x = torch::ones({5});
	q.push(std::move(x));
	
	auto y = q.front();
	std::cout << y << std::endl;

	auto a = y.accessor<float, 1>();
	a[3] = 10;

	std::cout << y<< std::endl;

	std::cout << x << std::endl;
	q.pop();
	
}

TEST(MultiStepPawnManager, SimpleQueueWithBatch) {

	using Batch = BatchStateFeature<torch::kCPU, BoardConfig, ShipModelConfig>;

	auto batch = Batch(5);

	std::queue<Batch> q;
	q.push(std::move(batch));
	
	Batch& y = q.front();

	//q.pop();

	std::cout << y.m_geometric << std::endl;
	q.pop();
}


TEST(MultiStepPawnManager, IntTest) {

	torch::Tensor x = torch::zeros({5});
	x = x.to(torch::kInt16);

	auto a = x.accessor<int16_t, 1>();
	a[3] = 10;

	std::cout << x<< std::endl;
	
}

template<typename Board>
void move(int action, Board& board) {
	Eigen::ArrayXi ship_actions(1);
	Eigen::ArrayXi shipyard_actions(1);
	ship_actions(0) = action;
	board.setActions(ship_actions, shipyard_actions);
	board.step();
}

template<typename Board>
void move2(int ship1, int shipyard1, Board& board) {
	Eigen::ArrayXi ship_actions(1);
	Eigen::ArrayXi shipyard_actions(1);
	ship_actions(0) = ship1;
	shipyard_actions(0) = shipyard1;
	board.setActions(ship_actions, shipyard_actions);
	board.step();
}

template<typename Board>
void move3(int ship1, int ship2, int shipyard1, Board& board) {
	Eigen::ArrayXi ship_actions(2);
	Eigen::ArrayXi shipyard_actions(1);
	ship_actions(0) = ship1;
	ship_actions(1) = ship2;
	shipyard_actions(0) = shipyard1;
	board.setActions(ship_actions, shipyard_actions);
	board.step();
}



TEST(MultiStepPawnManager, SinglePawn) {
    auto& re = RandomEngine<float, 123>::getInstance();
    BoardStore<decltype(re), BoardConfig, 1, 1> board_store;

    auto board = board_store.getNextBoard(re);
    auto& ship_map = board.getShipMap();
    auto& ship = ship_map.at(0);

    pregame_process(board);

    constexpr torch::DeviceType DeviceType = torch::kCPU;
    using ShipRewardEngine = RewardEngine<DeviceType, BoardConfig, ShipModelConfig>;
	  using ShipBatch = DynamicBatch<DeviceType, BoardConfig, ShipModelConfig>;
	  using ShipyardBatch =
      DynamicBatch<DeviceType, BoardConfig, ShipyardModelConfig>;
 		using ShipExample = SingleExample<DeviceType, BoardConfig, ShipModelConfig>;
		using ShipReplayBuffer = ReplayBuffer<DeviceType, ShipBatch, ShipExample>;

    std::size_t multi_step_n = 4;
    float gamma = .8;
    std::size_t batch_size = 32;

    MultiStepPawnManager<0, DeviceType, ShipRewardEngine, ShipReplayBuffer, ShipModelConfig> multi_step_pawn_manager(multi_step_n, gamma, batch_size);

    HyperParameters hp;

    auto reward_engine = ShipRewardEngine(hp);

    torch::Tensor latest_actions = torch::zeros({5}, torch::dtype(torch::kInt16).requires_grad(false).device(torch::kCPU));

    auto accessor = latest_actions.accessor<int16_t,1>();

    move(1, board);
    accessor[0] = 1;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);  

    move(0, board);
    accessor[0] = 0;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);

    move(4, board);
    accessor[0] = 4;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);

    move(0, board);
    accessor[0] = 0;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);

    move(3, board);
    accessor[0] = 3;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		
		const auto& batch = multi_step_pawn_manager.getFinalBatch();
		ASSERT_TRUE(batch.m_action.index({0}).item().to<int>()==0);
		EXPECT_NEAR(batch.m_reward.index({0}).item().to<float>(), 3.06587, 1e-4);
		EXPECT_NEAR(batch.m_state.m_geometric.index({0,2,0,0}).item().to<float>(), .9925, 1e-4);
		EXPECT_NEAR(batch.m_next_state.m_geometric.index({0,2,0,0}).item().to<float>(), .9825, 1e-4);

		move(3, board);
    accessor[0] = 3;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);

		move(2, board);
    accessor[0] = 2;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);


    move(2, board);
    accessor[0] = 2;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);

		move(0, board);
    accessor[0] = 0;
		multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);

		board.printBoard();

		auto & batch2 = multi_step_pawn_manager.getFinalBatch();
		std::cout << "action: " << batch2.m_action.index({0}) << std::endl;
		std::cout << "reward: " << batch2.m_reward.index({0}) << std::endl;
		std::cout << "state: " << batch2.m_state.m_geometric.index({0}) << std::endl;
		std::cout << "next state: " << batch2.m_next_state.m_geometric.index({0}) << std::endl;
}



TEST(MultiStepPawnManager, MultiPawn) {
    auto& re = RandomEngine<float, 123>::getInstance();
    BoardStore<decltype(re), BoardConfig, 1, 1> board_store;

    auto board = board_store.getNextBoard(re);
    auto& ship_map = board.getShipMap();
    auto& ship = ship_map.at(0);

    pregame_process(board);

    constexpr torch::DeviceType DeviceType = torch::kCPU;
    using ShipRewardEngine = RewardEngine<DeviceType, BoardConfig, ShipModelConfig>;
	  using ShipBatch = DynamicBatch<DeviceType, BoardConfig, ShipModelConfig>;
	  using ShipyardBatch =
      DynamicBatch<DeviceType, BoardConfig, ShipyardModelConfig>;
 		using ShipExample = SingleExample<DeviceType, BoardConfig, ShipModelConfig>;
		using ShipReplayBuffer = ReplayBuffer<DeviceType, ShipBatch, ShipExample>;

    std::size_t multi_step_n = 4;
    float gamma = .8;
    std::size_t batch_size = 32;

    MultiStepPawnManager<0, DeviceType, ShipRewardEngine, ShipReplayBuffer, ShipModelConfig> multi_step_pawn_manager(multi_step_n, gamma, batch_size);

    HyperParameters hp;

    auto reward_engine = ShipRewardEngine(hp);

    torch::Tensor latest_actions = torch::zeros({5}, torch::dtype(torch::kInt16).requires_grad(false).device(torch::kCPU));

    auto accessor = latest_actions.accessor<int16_t,1>();

    move(1, board);
    accessor[0] = 1;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);  
		board.printBoard();


		move2(1, 1, board);
    accessor[0] = 1;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();

		move3(1, 1, 0, board);
    accessor[0] = 1;
		accessor[1] = 1;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();

		move3(0, 0, 0, board);
    accessor[0] = 0;
		accessor[1] = 0;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();


		move3(0, 0, 0, board);
    accessor[0] = 0;
		accessor[1] = 0;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();


		move3(0, 0, 0, board);
    accessor[0] = 0;
		accessor[1] = 0;
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();


		move3(0, 1, 0, board);
    accessor[0] = 0;
		accessor[1] = 1;
		multi_step_pawn_manager.printRewardState();
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();

		auto& batch3 = multi_step_pawn_manager.getFinalBatch();
		std::cout << "actions:" << batch3.m_action.index({torch::indexing::Slice(0,2,1)}) << std::endl;
 		std::cout << "rewards:" << batch3.m_reward.index({torch::indexing::Slice(0,2,1)}) << std::endl; 
		std::cout << "state:" << batch3.m_state.m_geometric.index({0,2,0,0}) << std::endl; 
		std::cout << "next state:" << batch3.m_next_state.m_geometric.index({0,2,0,0}) << std::endl; 
		std::cout << "is non terminal:" << batch3.m_is_non_terminal.index({torch::indexing::Slice(0,2,1)}) << std::endl;



		//collision start
		move3(0, 1, 0, board);
    accessor[0] = 0;
		accessor[1] = 1;
		multi_step_pawn_manager.printRewardState();
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();
		//collision end
		

		auto& batch = multi_step_pawn_manager.getFinalBatch();
		std::cout << "actions:" << batch.m_action.index({torch::indexing::Slice(0,2,1)}) << std::endl;
 		std::cout << "rewards:" << batch.m_reward.index({torch::indexing::Slice(0,2,1)}) << std::endl; 
		std::cout << "state:" << batch.m_state.m_geometric.index({0,2,0,0}) << std::endl; 
		std::cout << "next state:" << batch.m_next_state.m_geometric.index({0,2,0,0}) << std::endl; 
		std::cout << "is non terminal:" << batch.m_is_non_terminal.index({torch::indexing::Slice(0,2,1)}) << std::endl;


		move2(2, 0, board);
    accessor[0] = 2;
		multi_step_pawn_manager.printRewardState();		
    multi_step_pawn_manager.updatePawnStates<ShipFeatureBuilder>(board, reward_engine, latest_actions);
		board.printBoard();

		auto& batch2 = multi_step_pawn_manager.getFinalBatch();
		std::cout << "actions:" << batch2.m_action.index({torch::indexing::Slice(0,2,1)}) << std::endl;
 		std::cout << "rewards:" << batch2.m_reward.index({torch::indexing::Slice(0,2,1)}) << std::endl; 
		std::cout << "state:" << batch2.m_state.m_geometric.index({0,2,0,0}) << std::endl; 
		std::cout << "next state:" << batch2.m_next_state.m_geometric.index({0,2,0,0}) << std::endl; 
		std::cout << "is non terminal:" << batch2.m_is_non_terminal.index({torch::indexing::Slice(0,2,1)}) << std::endl;

}



