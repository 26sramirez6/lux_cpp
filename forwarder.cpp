#include "lux/kit.hpp"
#include "lux/define.cpp"
#include "lux/client.hpp"
#include "server.hpp"
#include <sstream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <set>
#include <stdio.h>


static inline void initialize_game(char * membuf_) {
	const std::string id_line = kit::getline();
  const std::string map_info_line = kit::getline();
	std::stringstream game_init_lines;
	game_init_lines << game_start_key << id_line << map_info_line << '\0';
	const std::string game_init_str = game_init_lines.str();
	const char * game_init_cstr = game_init_str.c_str();
	strcpy(membuf_, game_init_cstr);
	while (*membuf_ != ack_input_received) {}
}

int main()
{
	char * membuf = locate_memory_map();
	initialize_game(membuf);

	while (true) {
		std::stringstream full_update;
		while (true) {
			const std::string line_update = kit::getline();
			full_update << line_update << '\n';
			if (line_update == kit::INPUT_CONSTANTS::DONE) {
				break;
			}
		}
		const std::string full_update_str(full_update.str());
		const char * cstr = full_update_str.c_str();
		strcpy(membuf, cstr);
		while (*membuf != ack_inputs_processed) {}
					
		std::cout << &membuf[1] << std::flush;
		*membuf = ack_actions_forwarded;			
	}


  return 0;
}
