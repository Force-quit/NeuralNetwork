#include <iostream>
#include <fstream>
#include <stdint.h>
#include <iomanip>

int readNextInt(std::ifstream& f)
{
  int x;
  f.read((char*)&x, sizeof(int));
  return __builtin_bswap32(x);
}

uint8_t readNextByte(std::ifstream& f)
{
  uint8_t x;
  f.read((char*) &x, sizeof(char));
  return x;
}


void writeToOutputFile(int nbData, int inputSize, 
                       std::ifstream& images, std::ifstream& labels, 
                       std::ofstream& output)
{
  for (int i=0; i<nbData; ++i)
    {
      for (int j=0; j<inputSize; ++j) 
        {
          char x = readNextByte(images);
          output.write(&x, sizeof(char));
        }
      int expectedOutput = readNextByte(labels);
      for (int k=0; k<10; ++k) 
        {
          char x = (k == expectedOutput) ? 1 : 0;
          output.write(&x, sizeof(char));
        }
    }
}

int main()
{
  std::ifstream images1, labels1, images2, labels2;
  images1.open("./train-images.idx3-ubyte", std::ios::binary | std::ios::in);
  labels1.open("./train-labels.idx1-ubyte", std::ios::binary | std::ios::in);
  images2.open("./t10k-images.idx3-ubyte", std::ios::binary | std::ios::in);
  labels2.open("./t10k-labels.idx1-ubyte", std::ios::binary | std::ios::in);

  std::ofstream output;
  output.open("./mnist-ubyte", std::ios::binary | std::ios::out);

  if (readNextInt(images1) != 2051) {
      return -1;
  }
  if (readNextInt(labels1) != 2049) {
      return -2;
  }
  if (readNextInt(images2) != 2051) {
      return -3;
  }
  if (readNextInt(labels2) != 2049) {
      return -4;
  }


  int nbImages1 = readNextInt(images1);
  int nbLabels1 = readNextInt(labels1);
  int nbImages2 = readNextInt(images2);
  int nbLabels2 = readNextInt(labels2);

  if (nbImages1 != nbLabels1) {
      return -5;
  }
  if (nbImages2 != nbLabels2) {
      return -6;
  }

  int nbData = nbImages1 + nbImages2;
  int nbRows1 = readNextInt(images1);
  int nbCols1 = readNextInt(images1);
  int nbRows2 = readNextInt(images2);
  int nbCols2 = readNextInt(images2);
  if (nbRows1 != nbRows2) {
      return -7;
  }
  if (nbCols1 != nbCols2) {
      return -8;
  }
  int nbRows = nbRows1;
  int nbCols = nbCols1;
  int n = nbRows*nbCols + 10;
  std::cout << "There are " << nbData << " datas to read. Each data is " 
    << nbRows << "x" << nbCols << "=" << (nbRows*nbCols) 
    << " bytes for the image plus 10 bytes for the expected output, which sums up to " 
    << n << " bytes per data." << std::endl;

  int nbInputs = nbRows * nbCols;
  int nbOutputs = 10;
  output.write((char*)&nbData, sizeof(int));
  output.write((char*)&nbInputs, sizeof(int));
  output.write((char*)&nbOutputs, sizeof(int));

  writeToOutputFile(nbImages1, nbRows*nbCols, images1, labels1, output);
  writeToOutputFile(nbImages2, nbRows*nbCols, images2, labels2, output);

  images1.close();
  labels1.close();
  images2.close();
  labels2.close();
  output.close();

  


  //std::cout << readNextInt(images) << std::endl;
  //std::cout << readNextInt(images) << std::endl;
  //std::cout << readNextInt(images) << std::endl;
  //std::cout << readNextInt(images) << std::endl;
  //for (int k = 0; k < 5; k++) {
  //    for (int i = 0; i < 28 ; ++i) {
  //        for (int j = 0; j < 28 ; ++j) {
  //            std::cout << " " << std::setw(3) << (int)readNextByte(images);
  //        }
  //        std::cout << '\n';
  //    }
  //    std::cout << std::endl;
  //}
  //
  //std::cout << readNextInt(labels) << std::endl;
  //std::cout << readNextInt(labels) << std::endl;
  //std::cout << (int)readNextByte(labels) << std::endl;
  //std::cout << (int)readNextByte(labels) << std::endl;
  //std::cout << (int)readNextByte(labels) << std::endl;
  //std::cout << (int)readNextByte(labels) << std::endl;
  //std::cout << (int)readNextByte(labels) << std::endl;





  return 0;
}

