To install optomdc:
pip install optoKummenberg-0.15.3755-py3-none-any.whl
pip install optoMDC-1.1.3755-py3-none-any.whl


To install dependencies run 
conda create --name <env> --file requirements.txt


Install libusb-win32 with Zadig to use USBTMC.


To profile the python script use
python -m cProfile -s tottime script.py

  
 To setup pybind:
  git clone https://github.com/pybind/pybind11.git


To compile the circle detection module run in pywhycon folder(Opencv and Cmake should be installed on the system):
mkdir build
cd build
cmake ..
make
The module file is generated in build folder inside the Release or Debug directory

-circle detection library must have the same python version with the environment python version
-you must place opencv dll files next to circle detection module file
-Must give execution rigts to dll and module files in circle detection library, otherwise you get access denied error during load dll operation