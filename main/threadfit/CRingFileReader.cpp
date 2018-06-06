/**
 * Implement the ring file reader class.
 *  Note that we do read ahead on the size of the next ring item in order to 
 *  only do one read per ring item.
 */

#include "CRingFileReader.h"
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <errno.h>
#include <string.h>
#include <system_error>

static const size_t Granule(8192);

/**
 * constructor
 *   - Open the file and save the file descriptor.
 *   - Read the size of the first ring item and save it in m_nextSize.
 *     this 'primes the pump' for the size readahead strategy.  See
 *     read for more.
 *
 *  @param path - path to the file we're going to read.
 */
CRingFileReader::CRingFileReader(const char* path)
{
  // Open the file and throw a system error if that fails:
  
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)),
			    "Opening the ring item file");
  }
  m_nFd = fd;

  // Read the size of the first ring item.  Note that:
  // If the size read < sizeof(std::uint32_t) - throw logic_error
  // iF the read is < 0 then throw the appropriate system_error.

  ssize_t  nRead = ::read(fd, &m_nextSize, sizeof(std::uint32_t));
  if (nRead < 0) {
    throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)),
			    "Reading first ring item size");
  }
  if (nRead < sizeof(std::uint32_t)) {
    throw std::logic_error("The file is not a ring item file");
  }
}


/**
 * Destructor - just close the file descriptor.
 */
CRingFileReader::~CRingFileReader()
{
  close(m_nFd);
}
/**
 *  Read a number of ring items from the file.
 *
 * @param nItems - number of items in the file.
 * @return CRingFileReader::DataDescriptor - struct that describes the 
 *      read:
 *         -   s_nBytes - is the total number of bytes read from file.
 *         -   s_nItems - is the total number of items read from file.
 *         -   s_pData  - Pointer to the data read -- free must be called
 *                        on this pionter when you're done using it.
 */
CRingFileReader::DataDescriptor
CRingFileReader::read(size_t nItems)
{
  size_t dataBufferSize(Granule); // Make the initial size big to avoid realloc.
  char* pDataBuffer = reinterpret_cast<char*>(malloc(Granule));
  if (!pDataBuffer) {
    throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)),
			    "Initial buffer allocation");
  }
  DataDescriptor result = {0, 0, pDataBuffer};
					      
  // If m_nextSize is zero we're off the end of the file.
  
  while(( m_nextSize != 0) && (result.s_nItems < nItems)) {
    // Put m_nextSize  in the buffer, read the next item and the following size
    // If needed, realloc is done to expand the buffersize.

    while ((m_nextSize + result.s_nBytes + sizeof(std::uint32_t) + result.s_nBytes) > dataBufferSize) {
      dataBufferSize += Granule;
      pDataBuffer = reinterpret_cast<char*>(realloc(pDataBuffer, dataBufferSize));
      if (!pDataBuffer) {
	throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)),
				 "Expanding buffer allocation");
      }
      result.s_pData = pDataBuffer;
    }
    // At this point we know the data buffer is big enough for the
    // last size, the ring item payload and the next size:

    memcpy(pDataBuffer + result.s_nBytes, &m_nextSize, sizeof(std::uint32_t));
    result.s_nBytes += sizeof(std::uint32_t);

    ssize_t nRead = ::read(m_nFd, pDataBuffer + result.s_nBytes, m_nextSize); // Reads item and next size.

    // Cases:
    //   nRead < 0  - throw an exception somethig bad happened.
    //   nRead < m_nextSize - We read the last ring item and the next item size is zero
    //   nread == m_nextSize - We read a ring item and following it is the size of the next one

    if (nRead < 0) {
      throw std::system_error(std::make_error_code(static_cast<std::errc>(errno)),
			      "Reading a ring item and next size from file");
    }
    if (nRead < m_nextSize) {
      m_nextSize = 0;
      result.s_nBytes += nRead;
      result.s_nItems++;
    }
    if (nRead == m_nextSize) {
      result.s_nBytes += nRead - sizeof(std::uint32_t);
      result.s_nItems++;

      memcpy(&m_nextSize, pDataBuffer + result.s_nBytes, sizeof(std::uint32_t));
    }
  }


  return result;
}
