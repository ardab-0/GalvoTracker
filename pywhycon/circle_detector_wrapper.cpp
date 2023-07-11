#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "circle_detector.h"


namespace py = pybind11;






PYBIND11_MODULE(circle_detector_module, module_handle)
{
	module_handle.doc() = "I'm a docstring hehe";
	// module_handle.def("some_fn_python_name", &some_fn);
	// module_handle.def("some_class_factory", &some_class_factory);
	// module_handle.def("get_nparray", &get_nparray);

	py::class_<cv::CircleDetector>(
		module_handle, "CircleDetectorClass")
		.def(py::init<int, int, float >(), py::arg("width"), py::arg("height"), py::arg("diameter_ratio") = WHYCON_DEFAULT_DIAMETER_RATIO)
        .def("detect_np", &cv::CircleDetector::detect_np);


    py::class_<cv::CircleDetector::Circle>(module_handle, "CircleClass").def(py::init<>())
    .def_property_readonly("x", [](cv::CircleDetector::Circle &self)  
    {
        return self.x;
    })
    .def_property_readonly("y", [](cv::CircleDetector::Circle &self)  
    {
        return self.y;
    }).def_property_readonly("v0", [](cv::CircleDetector::Circle &self)  
    {
        return self.v0;
    }).def_property_readonly("v1", [](cv::CircleDetector::Circle &self)  
    {
        return self.v1;
    }).def_property_readonly("m0", [](cv::CircleDetector::Circle &self)  
    {
        return self.m0;
    }).def_property_readonly("m1", [](cv::CircleDetector::Circle &self)  
    {
        return self.m1;
    });
		
}