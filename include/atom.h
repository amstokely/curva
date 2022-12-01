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

#ifndef CUDNA_PDB_H
#define CUDNA_PDB_H

#include <string>
#include "coordinates.h"
#include "json/include/nlohmann/json.hpp"

/*!
@brief This class stores the coordinate, physical, and identification
 information of an Atom. This information is parsed from both a PDB
 and DCD file.
*/
class Atom {
public:
/*!
 * Default constructor.
 */
	Atom ();

/*!
 * Constructs a new Atom by parsing a PDB file "ATOM" record line.
 *
 * @param pdbLine PDB file "ATOM" record line.
 *
 * \Example{
 * std::string pdbLine = (
 *     "ATOM      1  N   MET A   1      "
 *     + "40.741  36.600  32.609  1.00  0.00           N";
 * );
 * Atom atom = Atom(pdbLine);
 * }
 */
	explicit Atom (
			std::string &pdbLine
	);

/*!
 * Constructs a new Atom by parsing a PDB file "ATOM" record line.
 *
 * @param pdbLine PDB file "ATOM" record line.
 * @param atomIndex Index of the atom.
 *
 * \Example{
 * std::string pdbLine = (
 *     "ATOM  186a1  O2  AMANG   5     "
 *     + "-78.520   6.295  97.803  0.00  0.00      GC1
 *     "ATOM      1  N   MET A   1      "
 * );
 * int atomIndex = 100000;
 * Atom atom = Atom(pdbLine, atomIndex);
 * }
 */
	Atom (
			std::string &pdbLine,
			int atomIndex
	);


/*!
 * @return The \span{Atom}'s index.
 *
 * @note Atom indices follow 0-based indexing, so the first atom will
 * have an index of 0.
 *
 * \Example{
 * std::string pdbLine = (
 *     "ATOM      5  CA  MET 1   1      "
 *     + "38.506  51.642  51.424  0.00  0.00           C"
 * );
 * Atom atom = Atom(pdbLine);
 * int atomIndex = atom.index();
 * std::cout <<
 *     "The index of Atom object atom is " << atomIndex << "."
 * << std::endl
 * }
 *
 * \Output{
 * The index of Atom object atom is 4.
 * }
 */
	int index () const;

/*!
 * @return The \span{Atom}'s name
 *
 * \Example{
 * std::string pdbLine = (
 *     "ATOM      5  CA  MET 1   1      "
 *     + "38.506  51.642  51.424  0.00  0.00           C"
 * );
 * Atom atom = Atom(pdbLine);
 * std::string atomName = atom.name();
 * std::cout <<
 *     "The name of Atom object atom is " << atomName << "."
 * << std::endl;
 * }
 *
 * \Output{
 * The name of Atom object atom is CA.
 * }
 */
	std::string name ();

/*!
 * @return The \span{Atom}'s element symbol.
 *
 * \Example{
 * std::string pdbLine = (
 *     "ATOM      5  CA  MET 1   1      "
 *     + "38.506  51.642  51.424  0.00  0.00           C"
 * );
 * Atom atom = Atom(pdbLine);
 * std::string atomElementSymbol = atom.element();
 * std::cout <<
 *     "The element symbol of Atom object atom is " <<
 *      atomElementSymbol << "."
 * << std::endl; \\ The element symbol of Atom object atom is C.
 * }
 *
 * \Output{
 * The element symbol of Atom object atom is C.
 * }
 */
	std::string element ();

/*!
 * @return The \span{Atom}'s residue name.
 *
 * \{Example
 * std::string pdbLine = (
 *     "ATOM      5  CA  MET 1   1      "
 *     + "38.506  51.642  51.424  0.00  0.00           C"
 * );
 * Atom atom = Atom(pdbLine);
 * std::string atomResidueName = atom.residueName();
 * std::cout <<
 *     "The residue name of Atom object atom is " <<
 *      atomResidueName << "."
 * << std::endl;
 * }
 *
 * \Output{
 * The residue name of Atom object atom is MET.
 * }
 */
	std::string residueName ();

/*!
* @note Residue IDs follow 1-based indexing, so the first residue will
* have an ID of 1.
*
* @return The \span{Atom}'s residue ID.
*
* \Example{
* std::string pdbLine = (
*     "ATOM      5  CA  MET 1   1      "
*     + "38.506  51.642  51.424  0.00  0.00           C"
* );
* Atom atom = Atom(pdbLine);
* int atomResidueId = atom.residueId();
* std::cout <<
*     "The residue ID of Atom object atom is " <<
*      atomChainId << "."
* << std::endl;
* }
*
* \Output{
* The residue ID of Atom object atom is 1.
* }
*/
	int residueId () const;

/*!
* @return The \span{Atom}'s chain ID.
*
* @note Chain IDs follow 1-based indexing, so the first chain will
* have an ID of 1.
*
* \Example{
* std::string pdbLine = (
*     "ATOM      5  CA  MET 1   1      "
*     + "38.506  51.642  51.424  0.00  0.00           C"
* );
* Atom atom = Atom(pdbLine);
* int atomChainId = atom.chainId();
* std::cout <<
*     "The chain ID of Atom object atom is " <<
*      atomChainId << "."
* << std::endl;
* }
*
* \Output{
* The chain ID of Atom object atom is 1.
* }
 */
	std::string chainId ();

/*!
 * @return The \span{Atom}'s segment ID.
 */
	std::string segmentId ();

/*!
 * @return The \span{Atom}'s temperature factor.
 */
	double temperatureFactor () const;

/*!
 * @return The \span{Atom}'s occupancy.
 */
	double occupancy () const;

/*!
 *
 * @return The \span{Atom}'s serial number
 */
	int serial () const;

/*!
 *
 * @return The \span{Atom}'s tag.
 */
	std::string tag ();

/*!
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 *
 * @return A \span{RandomAccessIterator}<double> pointing to the
 * \span{Atom}'s first x coordinate element in \p coordinates.
 */
	RandomAccessIterator<double>
	xBegin (
			Coordinates *coordinates
	) const;

/*!
 *
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 *
 * @return A \span{RandomAccessIterator}<double> pointing to the
 * \span{Atom}'s last x coordinate element in \p coordinates.
 */
	RandomAccessIterator<double> xEnd (
			Coordinates *coordinates
	) const;

/*!
 *
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 *
 * @return A \span{RandomAccessIterator}<double> pointing to the
 * \span{Atom}'s first y coordinate element in \p coordinates.
 */
	RandomAccessIterator<double> yBegin (Coordinates *coordinates)
	const;

/*!
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 *
 * @return A \span{RandomAccessIterator}<double> pointing to the
 * \span{Atom}'s last y coordinate element in \p coordinates.
 */
	RandomAccessIterator<double> yEnd (Coordinates *coordinates) const;

/*!
 *
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 *
 * @return A \span{RandomAccessIterator}<double> pointing to the
 * \span{Atom}'s first z coordinate element in \p coordinates.
 */
	RandomAccessIterator<double> zBegin (Coordinates *coordinates)
	const;

/*!
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 *
 * @return A \span{RandomAccessIterator}<double> pointing to the
 * \span{Atom}'s last z coordinate element in \p coordinates.
 */
	RandomAccessIterator<double> zEnd (Coordinates *coordinates) const;

/*!
 * Get the \span{Atom}'s x coordinate value at a specific frame.
 *
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 * @param frameIndex The index of the frame the
 * \span{Atom}'s x coordinate value is returned from.
 *
 * @return The \span{Atom}'s x coordinate value at frame \p frameIndex.
 */
	double x (
			Coordinates *coordinates,
			int frameIndex
	) const;

/*!
 * Get the \span{Atom}'s y coordinate value at a specific frame.
 *
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 * @param frameIndex The index of the frame the
 * \span{Atom}'s y coordinate value is returned from.
 *
 * @return The \span{Atom}'s y coordinate value at frame \p frameIndex.
 */
	double y (
			Coordinates *coordinates,
			int frameIndex
	) const;

/*!
 * Get the \span{Atom}'s z coordinate value at a specific frame.
 *
 * @param coordinates Pointer to the Coordinates object that the
 * \span{Atom}'s cartesian coordinates are stored in.
 * @param frameIndex The index of the frame the
 * \span{Atom}'s z coordinate value is returned from.
 *
 * @return The \span{Atom}'s z coordinate value at frame \p frameIndex.
 */
	double z (
			Coordinates *coordinates,
			int frameIndex
	) const;

/*!
 *
 * @return The \span{Atom}'s mass
 */
	double mass () const {
		return _mass;
	}

/*!
 * @return The \span{Atom}'s hash value.
 */
	unsigned int hash () const;

private:
	friend nlohmann::adl_serializer<Atom>;
	friend nlohmann::adl_serializer<Atom *>;
/*!
 * %Index of the Atom
 */
	int          _index;
/*!
 * %Name of the Atom
 */
	std::string  _name;
/*!
 * Element symbol of the Atom
 */
	std::string  _element;
/*!
 * Residue name of the Atom
 */
	std::string  _residueName;
/*!
 * Residue ID of the Atom
 */
	int          _residueId;
/*!
 * Chain ID of the Atom
 */
	std::string  _chainId;
/*!
 * Segment ID of the Atom
 */
	std::string  _segmentId;
/*!
 * Temperature factor of the Atom
 */
	double       _temperatureFactor;
/*!
 * Occupancy of the Atom
 */
	double       _occupancy;
/*!
 * %Serial number of the Atom. The serial number follows 1-based
 * indexing so it is always 1 greater than the index.
 */
	int          _serial;
/*!
 * Key used to assign the Atom to a Node and is defined as the
 * string concatenation #_residueName + "_" +
 * #_residueId + "_" + #_chainId + "_" + #_segmentId.
 */
	std::string  _tag;
/*!
 *Atomic mass of the Atom.
 */
	double       _mass;
/*!
 * Hash value of the Atom, which allows it to be used as a key for
 * C++ <A HREF="https://en.cppreference.com/w/cpp/container/map">
 * std::map</A> and Python <A HREF="https://docs.python
 * .org/3/tutorial/datastructures.html#dictionary">dictionary</A>
 * data structures.
 */
	unsigned int _hash;
};

#endif //CUDNA_PDB_H
