
%{
#define SWIG_FILE_WITH_INIT
#include <sstream>
#include "../include/curva.h"
#include "../include/calculation.h"
#include "../include/utils.h"
#include "../include/atom.h"
#include "../include/matrix/curva_matrix.h"
#include "../include/matrix/curva_vector.h"
#include "../include/node.h"
#include "../include/atoms.h"
#include "../include/custom_iterators/random_access_iterator.h"
#include "../include/custom_iterators/forward_iterator.h"
#include "../include/custom_iterators/map_value_forward_iterator.h"
#include "../include/nodes.h"
#include "../include/coordinates.h"
#include "../include/cnpy/cnpy.h"
%}