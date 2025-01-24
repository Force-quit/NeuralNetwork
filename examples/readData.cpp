#include <iostream>
#include <fstream>
#include <stdint.h>
#include <iomanip>


int readNextInt(std::ifstream& f) 
{
  int x;
  f.read((char*)&x, sizeof(int));
  return x;
}

uint8_t readNextByte(std::ifstream& f) 
{
  uint8_t x;
  f.read((char*)&x, sizeof(char));
  return x;
}

int main()
{
  std::ifstream data;
  data.open("mnist-ubyte", std::ios::binary | std::ios::in);

  std::cout << readNextInt(data) << std::endl;
  std::cout << readNextInt(data) << std::endl;
  std::cout << readNextInt(data) << std::endl;
  for (int k = 0; k < 70000; ++k) 
    {
      std::cout << k << ".\n";
      for (int i = 0; i < 28 ; ++i) 
        {
          for (int j = 0; j < 28 ; ++j) 
            {
              std::cout << " " << std::setw(3) << (int)readNextByte(data);
            }
          std::cout << '\n';
        }
      std::cout << '\n';
      for (int i=0; i<10; ++i) 
        {
          std::cout << " " << (int) readNextByte(data);
        }
      std::cout << '\n';
      for (int i=0; i<10; ++i) 
        {
          std::cout << " " << i;
        }
      std::cout << '\n';
      std::cout << std::endl;
    }
  return 0;
}

