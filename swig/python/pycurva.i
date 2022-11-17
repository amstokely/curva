%module pycurva

%include std_string.i
%include std_vector.i
%template(dVector) std::vector<double>;
%template(ddVector) std::vector<std::vector<double>>;
%template(iVector) std::vector<int>;

%include "pycurva_headers1.i"
%include "swigerators.i"

%include "numpy.i"
%init %{
import_array();
%}

%include "pycurva_ignore.i"
%include "pycurva_typemaps.i"
%include "pycurva_headers2.i"

%template(dCurvaVector) CurvaVector<double>;
%template(dCurvaMatrix) CurvaMatrix<double>;
%template(iCurvaVector) CurvaVector<int>;
%template(iCurvaMatrix) CurvaMatrix<int>;

