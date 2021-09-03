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


static inline void send_actions(const std::vector<std::string>& _actions, char * membuf_) {
	std::stringstream ss;
	ss << ack_inputs_processed;
	for (int i = 0; i < _actions.size(); i++) {
		if (i != 0) {
			ss << ",";
		}
		ss << _actions[i];
	}
	ss << "D_FINISH\n";
	const std::string str(ss.str());
	const char * cstr = str.c_str();
	strcpy(membuf_, cstr);
} 

static std::vector<std::string> build_actions() {
	std::vector<std::string> ret;
	return ret;
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
		HyperParameters hyper_parameters;
		auto &random_engine = RandomEngine<float, TrainConfig::train_seed>::getInstance();
		auto trainer = Trainer<BoardConfig::actor_count, TrainConfig::device, decltype(random_engine)>(random_engine); 
		
		std::size_t episode = 0;
		std::size_t frame = 0;
		while (true) {
			bool is_new_game = wait_for_next_msg(membuf);

			if (is_new_game)	{
				agent.resetPlayerStates();
				initialize_game(agent, membuf);
				episode = 0;
				continue;
			}
			agent.updateServer(membuf);
			trainer.processEpisode(agent);
			std::vector<std::string> actions = build_actions();
			send_actions(actions, membuf);
			std::cout << "sent actions: " << membuf << std::endl;					
			wait_for_client_to_forward_actions(membuf);
			episode++;
			std::cout << "completed episode: " << episode << std::endl;
		}
    exit(0);
}
