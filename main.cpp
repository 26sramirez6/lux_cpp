#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <sstream>
#include <iostream>
#include "lux/kit.hpp"
#include "lux/define.cpp"
#include "server.hpp"
#include "trainer.hpp"
#include "random_engine.hpp"
#include "actions.hpp"

static inline bool wait_for_next_msg(char * membuf_) {
	*membuf_ = ack_input_received;
	while (*membuf_==ack_input_received) {};
	std::cout << "server received: " << membuf_ << std::endl;
	return *membuf_==game_start_key;
}

static inline void wait_for_client_to_forward_actions(char * membuf_) {
	while (*membuf_!=ack_actions_forwarded) {};
}

static inline void initialize_game(kit::Agent& agent_, char * membuf_) {
	membuf_++; // game_start_key first
	agent_.id = *membuf_ - '0';
	membuf_++; // assumes agent id single digit
	std::string map_info(membuf_);
	std::vector<std::string> map_parts = kit::tokenize(map_info, " ");
	agent_.mapWidth = std::stoi(map_parts[0]);
	agent_.mapHeight = std::stoi(map_parts[1]);
	agent_.map = lux::GameMap(agent_.mapWidth, agent_.mapHeight);
	std::cout << "initialized game: id " << agent_.id << ", " << agent_.mapWidth << ", " << agent_.mapHeight << std::endl;
}

template<typename ActionReturn>
static inline void send_actions(
	const kit::Agent& _agent, const ActionReturn& _actions, char * membuf_) {
	std::stringstream ss;
	ss << ack_inputs_processed;

	const auto worker_actions = std::get<0>(_actions);
	const auto ctile_actions = std::get<1>(_actions);
	const auto& player = _agent.players[_agent.id];
	const auto& workers = player.units;
	const auto&	cities = player.cities;
  
	assert(workers.size() == worker_actions.size());
	for (int i = 0; i < workers.size(); i++) {
		const auto& unit = workers[i];
		if (!unit.canAct()) continue;
		if (i!=0) ss << ",";
		const auto action = static_cast<WorkerActions>(worker_actions(i));
		switch (worker_actions) {
		case WorkerActions::CENTER:
			ss << unit.move(lux::DIRECTIONS::CENTER);
			break;
		case WorkerActions::NORTH:
			ss << unit.move(lux::DIRECTIONS::NORTH);
			break;
		case WorkerActions::EAST:
			ss << unit.move(lux::DIRECTIONS::EAST);
			break;
		case WorkerActions::SOUTH:
			ss << unit.move(lux::DIRECTIONS::SOUTH);
			break;
		case WorkerActions::WEST:
			ss << unit.move(lux::DIRECTIONS::WEST);
			break;
//		case WorkerActions::PILLAGE:
//			ss << unit.pillage();
//			break;
//		case WorkerActions::TRANSFER:
//			assert(false);
//			break;
		case WorkerActions::BUILD:
			ss << unit.buildCity();
			break;
		}
	}

	// TODO: massive fixme
	int i = 0;
	for (const auto& kv : cities) {
		const auto& city = kv.second;
		for (const auto& ctile : city.citytiles) {
			if (!ctile.canAct()) continue;
			if (i != 0) ss << ",";
			const auto action = ctile_actions(i++);
			switch (ctile_actions) {
			case CityTileActions::BUILD_WORKER:
				ss << ctile.buildWorker();
				break;
//			case CityTileActions::BUILD_CART:
//				ss << ctile.buildCart();
//				break;
			case CityTileActions::RESEARCH:
				ss << ctile.research();
				break; 
			}	
		}
	}	

	ss << "D_FINISH\n";
	const std::string str(ss.str());
	const char * cstr = str.c_str();
	strcpy(membuf_, cstr);
} 

static char * initialize_memory_map() {
	int shmid;
	char *membuf;

	if ((shmid = shmget(key, buf_size, IPC_CREAT | 0666)) < 0) {
			perror("shmget");
			exit(1);
	}

	if ((membuf = (char *)shmat(shmid, NULL, 0)) == (char *) -1) {
			perror("shmat");
			exit(1);
	}
	return membuf;
}


main() {
		char *membuf = initialize_memory_map();
		kit::Agent agent = kit::Agent();
		auto &random_engine = RandomEngine<float, TrainConfig::train_seed>::getInstance();
		auto trainer = Trainer<BoardConfig::actor_count, TrainConfig::device, decltype(random_engine)>(random_engine); 
		
		std::size_t episode = 0;
		std::size_t frame = 0;
		while (true) {
			bool is_new_game = wait_for_next_msg(membuf);

			if (is_new_game)	{
				agent.resetPlayerStates();
				trainer.resetState();
				initialize_game(agent, membuf);
				episode = 0;
				continue;
			}
			agent.updateServer(membuf);
			const auto actions = trainer.processEpisode(agent, frame, episode);
			send_actions(actions, membuf);
			std::cout << "sent actions: " << membuf << std::endl;					
			wait_for_client_to_forward_actions(membuf);
			episode++; frame++;
			std::cout << "completed episode: " << episode << std::endl;
		}
    exit(0);
}
