#include "lux/kit.hpp"
#include "lux/define.cpp"
#include "lux/client.hpp"
#include <sstream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <set>
#include <stdio.h>

using namespace std;
using namespace lux;
static string getline()
{
		// exit if stdin is bad now
		if (!std::cin.good())
				exit(0);

		char str[2048], ch;
		int i = 0;
		ch = getchar();
		while (ch != '\n')
		{
				str[i] = ch;
				i++;
				ch = getchar();
		}

		str[i] = '\0';
		// return the line
		return string(str);
}

std::ostream& operator<< (std::ostream& out, std::istream& in)
{
    in >> std::noskipws;
    char c;
    in >> c;

    while (in)
    {
        out << c;
        in >> c;
    }

    return out;
}


int main()
{
	char * membuf = locate_memory_map();
	while (true) {
		std::string update = getline();
		strcpy(membuf, update.c_str());
		while (*membuf != '!') {}	
		std::cout << "response received" << std::endl;
	}

//  kit::Agent game_state = kit::Agent();
//  // initialize
//  game_state.initialize();
//  while (true)
//  {
//    /** Do not edit! **/
//    // wait for updates
//    game_state.update(membuf);
//
//    vector<string> actions = vector<string>();
//    
//    /** AI Code Goes Below! **/
//		
//
//    Player &player = game_state.players[game_state.id];
//    Player &opponent = game_state.players[(game_state.id + 1) % 2];
//
//    const GameMap &game_map = game_state.map;
//
//    vector<Cell *> resourceTiles;
//    for (int y = 0; y < game_map.height; y++)
//    {
//      for (int x = 0; x < game_map.width; x++)
//      {
//        Cell *cell = (Cell*)game_map.getCell(x, y);
//        if (cell->hasResource())
//        {
//          resourceTiles.push_back(cell);
//        }
//      }
//    }
//
//    // we iterate over all our units and do something with them
//    for (int i = 0; i < player.units.size(); i++)
//    {
//      Unit unit = player.units[i];
//      if (unit.isWorker() && unit.canAct())
//      {
//        if (unit.getCargoSpaceLeft() > 0)
//        {
//          // if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
//          Cell *closestResourceTile;
//          float closestDist = 9999999;
//          for (auto it = resourceTiles.begin(); it != resourceTiles.end(); it++)
//          {
//            auto cell = *it;
//            if (cell->resource.type == ResourceType::coal && !player.researchedCoal()) continue;
//            if (cell->resource.type == ResourceType::uranium && !player.researchedUranium()) continue;
//            float dist = cell->pos.distanceTo(unit.pos);
//            if (dist < closestDist)
//            {
//              closestDist = dist;
//              closestResourceTile = cell;
//            }
//          }
//          if (closestResourceTile != nullptr)
//          {
//            auto dir = unit.pos.directionTo(closestResourceTile->pos);
//            actions.push_back(unit.move(dir));
//          }
//        }
//        else
//        {
//          // if unit is a worker and there is no cargo space left, and we have cities, lets return to them
//          if (player.cities.size() > 0)
//          {
//            auto city_iter = player.cities.begin();
//            auto &city = city_iter->second;
//
//            float closestDist = 999999;
//            CityTile *closestCityTile;
//            for (auto &citytile : city.citytiles)
//            {
//              float dist = citytile.pos.distanceTo(unit.pos);
//              if (dist < closestDist)
//              {
//                closestCityTile = &citytile;
//                closestDist = dist;
//              }
//            }
//            if (closestCityTile != nullptr)
//            {
//              auto dir = unit.pos.directionTo(closestCityTile->pos);
//              actions.push_back(unit.move(dir));
//            }
//          }
//        }
//      }
//    }
//
//    // you can add debug annotations using the methods of the Annotate class.
//    // actions.push_back(Annotate::circle(0, 0));
//
//    /** AI Code Goes Above! **/
//
//    /** Do not edit! **/
//    for (int i = 0; i < actions.size(); i++)
//    {
//      if (i != 0)
//        cout << ",";
//      cout << actions[i];
//    }
//    cout << endl;
//    // end turn
//    game_state.end_turn();
//  }

  return 0;
}
