1.1) To install dependencies run in root folder run with Anaconda prompt:
conda create --name <env> --file requirements.txt

1.2) To install optomdc:
    pip install optoKummenberg-0.15.3755-py3-none-any.whl
    pip install optoMDC-1.1.3755-py3-none-any.whl

1.3) Install libusb-win32 with Zadig (url: https://zadig.akeo.ie/) to use USBTMC.


1.4) To profile the python script use (optional)
python -m cProfile -s tottime script.py

1.5) WHYCON Python binding generation (Optional)

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




2.1) Scripts

- calibrate_chessboard.py : It is the main calibration program. Used with chessboard calibration pattern.
- visualize_calibration_points.py : It displays the recorded calibration points in 3D 
- measure_calibration_error_with_target_plane.py : It measures the calibration error by scanning around the detector and finding its cente location. Measurement is done by pressing "m" key. 
- point_laser_to_mouse_position.py : Test script to check depth camera and mirror controller integration. It points the laser to mouse location.
- pywhycon_track_target_with_laser.py : System points the laser to marker position. It is the combination of all parts of the system.
- mirror_gui.py : Simple GUI program to control mirror. 3D coordinates are entered with sliders and laser is pointed to entered position. 