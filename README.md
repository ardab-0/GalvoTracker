1.1) To install optomdc:
    pip install optoKummenberg-0.15.3755-py3-none-any.whl
    pip install optoMDC-1.1.3755-py3-none-any.whl


1.2) To install dependencies run in root folder:
conda create --name <env> --file requirements.txt


1.3) Install libusb-win32 with Zadig (url: https://zadig.akeo.ie/) to use USBTMC.


1.4) To profile the python script use
python -m cProfile -s tottime script.py

1.5) WHYCON Python binding generation

  To setup pybind:
    git clone https://github.com/pybind/pybind11.git


  To compile the circle detection module run in pywhycon folder(Opencv and CMake should be installed on the system):
  mkdir build
  cd build
  cmake ..
  make
  The module file is generated in build folder inside the Release or Debug directory

  -circle detection library must have the same python version with the environment python version
  -you must place opencv dll files next to circle detection module file
  -Must give execution rigts to dll and module files in circle detection library, otherwise you get access denied error during load dll operation




