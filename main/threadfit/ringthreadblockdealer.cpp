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
#include "CRingThreadedBlockReader.h"

int NBR_WORKERS = 20;
int CHUNK_SIZE  = 1024*1024;
const char* fileName;

static size_t* threadBytes;
static size_t* threadItems;

static void
processRingItems(CRingThreadedBlockReader::pDataDescriptor descrip, void* pData)
{}

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
      s_send(worker, "fetch");
      

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
        CRingThreadedBlockReader::pDataDescriptor pDescriptor =
          reinterpret_cast<CRingThreadedBlockReader::pDataDescriptor>(descriptor.data());
      
        nItems += pDescriptor->s_nItems;
        bytes  += pDescriptor->s_nBytes;
      
        processRingItems(pDescriptor, pRingItems); // any interesting work goes here.
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

static void sendData(
  zmq::socket_t& sock, const std::string& identity,
  const CRingThreadedBlockReader::pDataDescriptor data
)
{
  sendHeader(sock, identity);
  s_sendmore(sock, "data");

  size_t dataSize = data->s_nBytes;
  zmq::message_t descriptor(sizeof(CRingThreadedBlockReader::DataDescriptor));
  memcpy(descriptor.data(), data, sizeof(CRingThreadedBlockReader::DataDescriptor));
  zmq::message_t dataBytes(data->s_pData, dataSize, freeData);

  sock.send(descriptor, ZMQ_SNDMORE);
  sock.send(dataBytes, 0);

  
}

static size_t bytesSent(0);
static int  sendChunk(
    zmq::socket_t& sock, const std::string& identity,
    CRingThreadedBlockReader& reader
)
{
  CRingThreadedBlockReader::DataDescriptor Desc;
  CRingThreadedBlockReader::pDataDescriptor pDesc(&Desc);   // So it matches prior code.
  Desc = reader.read();
  if (pDesc->s_nBytes > 0) {
    size_t nSent = pDesc->s_nBytes;
    bytesSent += nSent;;
    sendData(sock ,identity, pDesc);
    return nSent;
   
  } else {
    free(pDesc->s_pData);
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

    CRingThreadedBlockReader reader(fileName,  context, CHUNK_SIZE, 10, 5);


    //  Run for five seconds and then tell workers to end
    int workers_fired = 0;
    bool done = false;
    while (1) {
        //  Next message gives us least recently used worker
        std::string identity = s_recv(broker);
        {
            s_recv(broker);     //  Envelope delimiter
            std::string command = s_recv(broker);     //  Command
        }
        if (!done) {
          int status =  sendChunk(broker, identity, reader);
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

static void Usage(const char* program, std::ostream& o)
{
  o << "Usage:\n";
  o <<     program << "  num_workers  read_blocksize filename\n";
  o << "Where:\n";
  o << "   num_workers   -   Number of worker threads\n";
  o << "   read_blocksize-   Size of the file reads (in bytes)\n";
  o << "   filename      -    Path to the data file\n";
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

  if (argc != 4) {
    Usage(argv[0], std::cerr);
    exit(EXIT_FAILURE);
  }
  
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
