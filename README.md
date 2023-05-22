To install dependencies run 
conda create --name <env> --file requirements.txt


Install libusb-win32 with Zadig to use USBTMC.


To profile the python script use
python -m cProfile -s tottime script.py
