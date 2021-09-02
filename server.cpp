#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <iostream>
#include "lux/kit.hpp"
#include "lux/define.cpp"
#include "server.hpp"

static inline void wait_for_next_msg(char * membuf_) {
	*membuf_ = ack;
	while (*membuf_==ack) {};
	std::cout << "server received: " << membuf_ << std::endl;
}

static inline void initialize_game(kit::Agent& agent_, char * membuf_) {
	agent_.id = std::stoi(membuf_);
	wait_for_next_msg(membuf_);
	std::string map_info(membuf_);
	std::vector<std::string> map_parts = kit::tokenize(map_info, " ");
	agent_.mapWidth = std::stoi(map_parts[0]);
	agent_.mapHeight = std::stoi(map_parts[1]);
	agent_.map = lux::GameMap(agent_.mapWidth, agent_.mapHeight);
	std::cout << "initialized game: " << agent_.mapWidth << ", " << agent_.mapHeight << std::endl;
}


static inline void process_message(kit::Agent& agent_, char * membuf_, bool& is_game_in_progress_) {
	if (!is_game_in_progress_) {
		initialize_game(agent_, membuf_);
		is_game_in_progress_ = true;
		return;
	}

	agent_.update(membuf_);
		
}

main() {
    char c;
    int shmid;
    char *membuf, *s;

    /*
     * Create the segment.
     */
    if ((shmid = shmget(key, buf_size, IPC_CREAT | 0666)) < 0) {
        perror("shmget");
        exit(1);
    }

    /*
     * Now we attach the segment to our data space.
     */
    if ((membuf = (char *)shmat(shmid, NULL, 0)) == (char *) -1) {
        perror("shmat");
        exit(1);
    }

		kit::Agent agent = kit::Agent();
		bool is_game_in_progress = false;
		while (true) {
			wait_for_next_msg(membuf);	
			process_message(agent, membuf, is_game_in_progress);
			if (!is_game_in_progress) {
				std::cout << "game completed" << std::endl;
			}
		}
    exit(0);
}
