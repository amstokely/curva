//
// Created by andy on 9/9/22.
//

#ifndef CURVA_SERIALIZER_H
#define CURVA_SERIALIZER_H

#include "json/include/nlohmann/json.hpp"
#include "calculation.h"

namespace nlohmann {


	template<>
	struct adl_serializer<Atom *> {
#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

		static Atom *from_json (
				const json &j
		);

#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

		static void to_json (
				json &j,
				Atom *atom
		);
	};

	template<>
	struct adl_serializer<Node *> {


#pragma clang diagnostic push
#pragma ide diagnostic ignored "NotImplementedFunctions"

		static void to_json (
				json &j,
				Node *node
		);
	};

}
#endif //CURVA_SERIALIZER_H

