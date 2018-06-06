//
//  Custom routing Router to Dealer
//
// Olivier Chamoux <olivier.chamoux@fr.thalesgroup.com>

#include "zhelpers.hpp"
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <sstream>
#include <stdint.h>
#include "CRingFileReader.h"

int NBR_WORKERS = 20;
int CHUNK_SIZE  = 1024*1024;
const char* fileName;

static size_t* threadBytes;
static size_t* threadItems;

static void
processRingItems(CRingFileReader::pDataDescriptor descrip, void* pData, int workerId)
{
}

/*
 *  Worker task. void* args is actually an integer worker number
 */
static void *
worker_task(void *args)
{
  long thread = (long)(args);
    zmq::context_t context(1);
    zmq::socket_t worker(context, ZMQ_DEALER);
    int linger(0);
    worker.setsockopt(ZMQ_LINGER, &linger, sizeof(int));

#if (defined (WIN32))
    s_set_id(worker, (intptr_t)args);
#else
    s_set_id(worker);          //  Set a printable identity
#endif

    worker.connect("tcp://localhost:5671");
    std::stringstream ChunkSize;
    ChunkSize << CHUNK_SIZE;
    size_t bytes = 0;
    size_t nItems = 0;
    int total = 0;
    while (1) {
      s_sendmore(worker, "");
      s_sendmore(worker, "fetch");
      s_send(worker, ChunkSize.str());                 // Size of workload
      

      // Work items are in two types.  all start with delimetr and type.
      // Type eof means we're done and need to clean up.
      // type data means there's two more segments, the descriptor and the data.

      s_recv(worker);                               // Delimeter.
      std::string type = s_recv(worker);
      if (type == "eof") {
	break;
      } else if (type == "data") {
	zmq::message_t descriptor;
	zmq::message_t bulkData;

	worker.recv(&descriptor);
	worker.recv(&bulkData);

	void* pRingItems = bulkData.data();
	CRingFileReader::pDataDescriptor pDescriptor =
	  reinterpret_cast<CRingFileReader::pDataDescriptor>(descriptor.data());

	nItems += pDescriptor->s_nItems;
	bytes  += pDescriptor->s_nBytes;

	processRingItems(pDescriptor, pRingItems, thread); // any interesting work goes here.
      } else {
	std::cerr << "Worker " << (long)args << " got a bad work item type " << type << std::endl;
	break;
      }
    }
    threadBytes[thread] = bytes;
    threadItems[thread]  = nItems;
    worker.close();
    return NULL;
}




void freeData(void* pData, void* pHint)
{

  free(pData);
}

static void sendHeader(zmq::socket_t& sock, const std::string& identity)
{
  s_sendmore(sock, identity);
  s_sendmore(sock, "");
}

static void sendEOF(zmq::socket_t& sock, const std::string& identity)
{
  sendHeader(sock, identity);
  s_send(sock, "eof");
}

static void sendData(zmq::socket_t& sock, const std::string& identity, const CRingFileReader::pDataDescriptor data)
{
  sendHeader(sock, identity);
  s_sendmore(sock, "data");

  size_t dataSize = data->s_nBytes;
  zmq::message_t descriptor(data, sizeof(CRingFileReader::DataDescriptor), freeData);
  zmq::message_t dataBytes(data->s_pData, dataSize, freeData);

  sock.send(descriptor, ZMQ_SNDMORE);
  sock.send(dataBytes, 0);

  
}

static size_t bytesSent(0);
static int  sendChunk(zmq::socket_t& sock, const std::string& identity, CRingFileReader& reader,  size_t nItems)
{
  CRingFileReader::pDataDescriptor pDesc =
    reinterpret_cast<CRingFileReader::pDataDescriptor>(malloc(sizeof(CRingFileReader::DataDescriptor)));

  *pDesc = reader.read(nItems);
  if (pDesc->s_nBytes > 0) {
    size_t nSent = pDesc->s_nBytes;
    bytesSent += nSent;;
    sendData(sock ,identity, pDesc);
    return nSent;
   
  } else {
    free(pDesc->s_pData);
    free(pDesc);
    sendEOF(sock, identity);
    return 0;
  }
}


static void*
sender_task(void* args)
{
    zmq::context_t context(1);
    zmq::socket_t broker(context, ZMQ_ROUTER);
    int linger(0);
    broker.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    
    FILE* pFile;

    broker.bind("tcp://*:5671");

    CRingFileReader reader(fileName);


    //  Run for five seconds and then tell workers to end
    int workers_fired = 0;
    bool done = false;
    while (1) {
        //  Next message gives us least recently used worker
        std::string identity = s_recv(broker);
	std::string size;
        {
            s_recv(broker);     //  Envelope delimiter
	    std::string command = s_recv(broker);     //  Command
	    size    = s_recv(broker);     //  size:
        }
	if (!done) {
	  int status =  sendChunk(broker, identity, reader, atoi(size.c_str()));
	  if (status == 0) {
	    done = true;
	    sendEOF(broker, identity);
	    
	    ++workers_fired;
	    if (workers_fired == NBR_WORKERS) break;
	  }
        } else {
	  sendEOF(broker, identity);
	  if (++workers_fired == NBR_WORKERS)
	    break;
        }
    }
    broker.close();

}

//  While this example runs in a single process, that is just to make
//  it easier to start and stop the example. Each thread has its own
// context and conceptually acts as a separate process.
/// Usage:
//    ringdealer workers chunksize filename
//
int main(int argc, char** argv) {
  pthread_t* workers;
  pthread_t sender;

  NBR_WORKERS = atoi(argv[1]);
  CHUNK_SIZE   = atoi(argv[2]);
  fileName     = argv[3];
  
  workers = new pthread_t[NBR_WORKERS];
  threadBytes = new size_t[NBR_WORKERS];
  threadItems = new size_t[NBR_WORKERS];


  
  pthread_create(&sender, nullptr, sender_task,  nullptr);
  
  for (int worker_nbr = 0; worker_nbr < NBR_WORKERS; ++worker_nbr) {
    pthread_create(workers + worker_nbr, NULL, worker_task, (void *)(intptr_t)worker_nbr);
  }
  
  for (int worker_nbr = 0; worker_nbr < NBR_WORKERS; ++worker_nbr) {
    pthread_join(workers[worker_nbr], NULL);
  }
  pthread_join(sender, nullptr);
  
  size_t totalBytes(0);
  size_t totalItems(0);
  for (int i = 0; i < NBR_WORKERS; i++) {
    std::cout << "Thread " << i << " processed " <<
      threadItems[i] << " items containing a total of " << threadBytes[i] << " bytes"  << std::endl;
    totalBytes += threadBytes[i];
    totalItems += threadItems[i];
  }
  
  std::cout << "Items processed " << totalItems << " totalBytesProcessed  " << totalBytes << std::endl;

  delete []workers;
  delete []threadBytes;
  delete []threadItems;
  
  return 0;
}
