/*--------------------------------------------------------------------*
 *                         CuRva                                      *
 *--------------------------------------------------------------------*
 * This is part of the GPU-accelerated, random variable analysis      *
 * library CuRva.                                                     *
 * Copyright (C) 2022 Andy Stokely                                    *
 *                                                                    *
 * This program is free software: you can redistribute it             *
 * and/or modify it under the terms of the GNU General Public License *
 * as published by the Free Software Foundation, either version 3 of  *
 * the License, or (at your option) any later version.                *
 *                                                                    *
 * This program is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *
 * GNU General Public License for more details.                       *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this program.                                           *
 * If not, see <https://www.gnu.org/licenses/>                        *
 * -------------------------------------------------------------------*/

#include "../include/serializer.h"
#include "../include/cnpy/cnpy.h"

Atom *
nlohmann::adl_serializer
		<Atom *>::from_json (
		const json &j
) {
	Atom atom{};
	atom._index             = j.at("_index").get<int>();
	atom._name              = j.at("_name").get<std::string>();
	atom._element           = j.at("_element").get<std::string>();
	atom._residueName       = j.at("_residueName").get<std::string>();
	atom._residueId         = j.at("_residueId").get<int>();
	atom._chainId           = j.at("_chainId").get<std::string>();
	atom._segmentId         = j.at("_segmentId").get<std::string>();
	atom._temperatureFactor = j.at("_temperatureFactor").get<double>();
	atom._occupancy         = j.at("_occupancy").get<double>();
	atom._serial            = j.at("_serial").get<int>();
	atom._tag               = j.at("_tag").get<std::string>();
	atom._mass              = j.at("_mass").get<double>();
	atom._hash              = j.at("_hash").get<unsigned int>();
	return new Atom(atom);
}

void nlohmann::adl_serializer
		<Atom *>::to_json (
		json &j,
		Atom *atom
) {
	j = json{
			{"_index",             atom->_index},
			{"_name",              atom->_name},
			{"_element",           atom->_element},
			{"_residueName",       atom->_residueName},
			{"_residueId",         atom->_residueId},
			{"_chainId",           atom->_chainId},
			{"_segmentId",         atom->_segmentId},
			{"_temperatureFactor", atom->_temperatureFactor},
			{"_occupancy",         atom->_occupancy},
			{"_serial",            atom->_serial},
			{"_tag",               atom->_tag},
			{"_mass",              atom->_mass},
			{"_hash",              atom->_hash}
	};
}

void nlohmann::adl_serializer
		<Node *>::to_json (
		json &j,
		Node *node
) {
	std::stringstream centerOfMassFnameSs;
	centerOfMassFnameSs << node->calculationName() << "_node"
	                    << node->_index << "_centerOfMass.npy";
	std::stringstream averageCenterOfMassFnameSs;
	averageCenterOfMassFnameSs << node->calculationName() << "_node"
	                           << node->_index
	                           << "_averageCenterOfMass.npy";
	std::vector<size_t> centerOfMassShape        = {
			node->_centerOfMass.size()
	};
	std::vector<size_t> averageCenterOfMassShape = {
			node->_averageCenterOfMass.size()
	};
	std::string         centerOfMassFname        =
			                    centerOfMassFnameSs.str();
	std::string         averageCenterOfMassFname =
			                    averageCenterOfMassFnameSs.str();
	cnpy::npy_save(
			node->_serializationDirectory + "/" + centerOfMassFname,
			node->_centerOfMass.data(),
			centerOfMassShape
	);
	cnpy::npy_save(
			node->_serializationDirectory + "/" +
			averageCenterOfMassFname,
			node->_averageCenterOfMass.data(),
			averageCenterOfMassShape
	);
	j = {
			{"_centerOfMass",        centerOfMassFname},
			{"_averageCenterOfMass", averageCenterOfMassFname},
			{"_numAtoms",            node->_numAtoms},
			{"_index",               node->_index},
			{"_totalMass",           node->_totalMass},
			{"_tag",                 node->_tag},
			{"_hash",                node->_hash},
			{"atoms",                node->atoms},
	};
}





