#include <string>
#include <iostream>
#include <fstream>

/*
 * Importiert ein Array von Vektoren, die Beziehungen repräsentieren. 
 * dest: float-Array der Größe: Anzahl Sätze x Größe Vektoren (25000x50)
 * ids:  int-Array der Größe: Anzahl Sätze (Enthält danach die entsprechenden IDs der jeweiligen Sätze
 * filename: Der Dateiname des zu lesenden Arrays (/home/t2/repr_80354.csv für Lufthansa)
 */
void arrayimport(float *dest, int *ids, const std::string &filename) {
	std::ifstream infile(filename);
	unsigned int current_dest_index = 0;
	unsigned int current_id_index = 0;
	if (infile.is_open()) {
		std::string line;
		while (std::getline(infile, line)) {
			size_t oldpos = 0;
			size_t newpos = line.find_first_of(';', oldpos);
			std::string dataid = line.substr(oldpos, newpos - oldpos);
			ids[current_id_index] = std::stoi(dataid);
			current_id_index++;
			oldpos = newpos + 1;
			newpos = line.find_first_of(';', oldpos);
			while (newpos != std::string::npos) {
				std::string tmp = line.substr(oldpos, newpos - oldpos);
				dest[current_dest_index] = std::stof(tmp);
				current_dest_index++;
				oldpos = newpos + 1;
				newpos = line.find_first_of(';', oldpos);
			}
			std::string tmp = line.substr(oldpos, newpos - oldpos);
			dest[current_dest_index] = std::stof(tmp);
			current_dest_index++;
		}
		infile.close();
	}
}
