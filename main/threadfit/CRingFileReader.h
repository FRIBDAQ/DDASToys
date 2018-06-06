/**
 * CRingFileReader is a class that handles reading blocks of ring items from file.
 */

#ifndef CRINGFILEREADER_H
#define CRINGFILEREADER_H

#include <cstdint>
#include <stddef.h>
#include <string.h>

class CRingFileReader {
private:
  int           m_nFd;
  std::uint32_t m_nextSize;
public:

  // This is what you get back from a read:
  
  typedef struct _DataDescriptor {
    size_t s_nBytes;		// Number of bytes read.
    size_t s_nItems;		// Number of items actually read.
    void*  s_pData;		// Pointer to the malloc'd data.
    
  } DataDescriptor, *pDataDescriptor;

public:
  CRingFileReader(const char* path);
  virtual ~CRingFileReader();

  DataDescriptor read(size_t nItems);
  void read(pDataDescriptor out, size_t nItems) {
    DataDescriptor info = read(nItems);
    memcpy(out, &info, sizeof(DataDescriptor));
  }
};

#endif
