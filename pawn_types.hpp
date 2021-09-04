#ifndef PAWN_TYPES_HPP
#define PAWN_TYPES_HPP

#include <unordered_map>
#include <string>
#include "lux/kit.hpp"

struct Worker{

	static inline int string_to_id(const std::string& _strId) {
		char * cstr = (char*) _strId.c_str();
		return std::stoi(++++_strId);
	}
		
	static inline std::vector<int> get_pawn_ids(const kit::Agent& _agent) {
		const auto& player = _agent.players[_agent.id];
		const auto& units = player.units;
		std::vector<int> ids;
		ids.reserve(units.size());
		for (int i = 0; i < units.size(); i++) {
			if (units[i].isWorker()) ids.push_back(string_to_id(units[i].id));
		}
		return ids;
	}

};

struct CityTile {
	static inline int string_to_id(const std::string& _strId) {
		char * cstr = (char*) _strId.c_str();
		std::stoi(++++_strId);
	}

//	static inline std::vector<int> getPawnIds(const kit::Agent& _agent) {
//		const auto& player = _agent.players[_agent.id];
//		const auto& cities = player.cities;
//		std::vector<int> ids;
//		for (auto& kv : cities) {
//			const auto& ctiles = kv.second.citytiles;
//			for (int i = 0; i < ctiles.size(); i++) {
//				ids.push_back(ctiles[i]);
//			}
//		} 
//	}

};

struct Cart {
	static inline int string_to_id(const std::string& _strId) {
		char * cstr = (char*) _strId.c_str();
		return std::stoi(++++_strId);
	}

	static inline std::vector<int> getPawnIds(const kit::Agent& _agent) {
		const auto& player = _agent.players[_agent.id];
		const auto& units = player.units;
		std::vector<int> ids;
		ids.reserve(units.size());
		for (int i = 0; i < units.size(); i++) {
			if (!units[i].isWorker()) ids.push_back(string_to_id(units[i].id));
		}
		return ids;
	}
};
