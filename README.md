# OCT_MT
This is the control software for a Line scan OCT system at Sustech.
# SOFTWARE STRUCTURE

using QThread of PYQT to do multi-threading control of data acquisition, scanning, data processing, display&saving

using Queue to organize threads

spectral domain data are stored in global memory, and memory location (pointers) are shared between threads using Queue
# HARDWARE STRUCTURE

Master: PCIe board for Galvo control   
Slave: Camera in spectrameter

scanning regime: Line illumination in X dimension, Y galvo scan in Y dimension


this is the software framework:
![software frame](https://github.com/user-attachments/assets/962d2162-0599-4fcf-8886-57a50430deae)
This is the software timeline:
![software timeline](https://github.com/user-attachments/assets/c8a3ae85-6902-4988-984e-8a155be562bc)
