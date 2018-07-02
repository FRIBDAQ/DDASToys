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
#include <vector>
#include <FragmentIndex.h>
#include <DDASHit.h>

#include "CRingFileBlockReader.h"
#include "CDDASAnalyzer.h"
#include "CZMQRingOutputter.h"

#include <CRingItem.h>
#include <CRingItemFactory.h>
#include "zmqsortedwriter.h"



static int NBR_WORKERS(20);
static int CHUNK_SIZE  = 1024*1024;
static const char* fileName;
static char* outfile;
static size_t* threadBytes;
static size_t* threadItems;

static zmq::context_t globalContext(5);

zmq::context_t& getContext() {return globalContext;}


//  Predicate used to restrict which fragments get fits;

class ChannelSelector : public CDDASAnalyzer::FitPredicate
{
private:
  struct Channel {
    int s_crate;
    int s_slot;
    int s_chan;
  };
  std::vector<Channel> m_channels;
public:
  ChannelSelector() {}
  virtual ~ChannelSelector() {}
  
  void addChannel(int c, int s, int ch) {
    Channel aChan = {c, s, ch};
    m_channels.push_back(aChan);
  }
  virtual std::pair<std::pair<unsigned, unsigned>, unsigned> operator()(
    const FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
    const std::vector<uint16_t>& trace    
  );
};
/**
 * Implement the function call operator of the channel selector.
 * this returns the full trace for channels that match at least one
 * entry in m_channels
 */
std::pair<std::pair<unsigned, unsigned>, unsigned>
ChannelSelector::operator()(
  const FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
  const std::vector<uint16_t>& trace
)
{
  int cr = hit.GetCrateID();
  int sl = hit.GetSlotID();
  int ch = hit.GetChannelID();
  std::pair<std::pair<unsigned, unsigned>, unsigned > fulltrace =
    {{0, trace.size() -1}, 0x3fff};           // For now assume 13 bits.
    
  std::pair<std::pair<unsigned, unsigned>, unsigned> nofit =
    {{0,0}, 0};                           // Saturation value doesn't matter.
  
  for (int i =0; i < m_channels.size(); i++) {
    if (
        (cr == m_channels[i].s_crate) &&
        (sl == m_channels[i].s_slot) &&
        (ch == m_channels[i].s_chan)
      ) {
      return fulltrace;
    }
  }
  // No matches;
  
  return nofit;
}

/**
 *  Where the work gets done.
 *  We iterate over all the ring items in the bulk data
 *  and pass each item off to the analyzeItem function.
 */
static void
processRingItems(CRingFileBlockReader::pDataDescriptor descrip, void* pData,  CDDASAnalyzer&  a)
{
  uint8_t* pItem = reinterpret_cast<uint8_t*>(pData);
  for (int i= 0; i < descrip->s_nItems; i++) {
    CRingItem* pRingItem = CRingItemFactory::createRingItem(pItem);
    a(pRingItem);
    delete pRingItem;
    uint32_t* pItemSize = reinterpret_cast<uint32_t*>(pItem);
    pItem += *pItemSize;
  }
}

static void *
worker_task(void *args)
{
    long thread = (long)(args);
    zmq::context_t& context(getContext());
    zmq::socket_t worker(context, ZMQ_DEALER);
    int linger(0);
    worker.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    
    // Create the outputter and the analyzer as a connected pair:
    
    zmq::socket_t& outsock(makeClientSocket(context));
    CZMQRingOutputter out(outsock);
    CDDASAnalyzer analyzer(thread, out);

    // Register the predicates that restrict the items that will be fit:
    // Note if the first fit is not done, neither is the second so this case
    // we only need to have a single fit predicate.
    
    ChannelSelector pred;
    pred.addChannel(0,2,0);     // Anode channel.
    analyzer.setSingleFitPredicate(&pred);  
    
#if (defined (WIN32))
    s_set_id(worker, (intptr_t)args);
#else
    s_set_id(worker);          //  Set a printable identity
#endif

    worker.connect("inproc://sender");
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
        analyzer.end();
        break;
      } else if (type == "data") {
        zmq::message_t descriptor;
        zmq::message_t bulkData;
      
        worker.recv(&descriptor);
        worker.recv(&bulkData);
      
        void* pRingItems = bulkData.data();
        CRingFileBlockReader::pDataDescriptor pDescriptor =
          reinterpret_cast<CRingFileBlockReader::pDataDescriptor>(descriptor.data());
      
        nItems += pDescriptor->s_nItems;
        bytes  += pDescriptor->s_nBytes;
      
        processRingItems(pDescriptor, pRingItems, analyzer); // any interesting work goes here.
            } else {
        std::cerr << "Worker " << (long)args << " got a bad work item type " << type << std::endl;
        break;
      }
    }
    threadBytes[thread] = bytes;
    threadItems[thread]  = nItems;
    worker.close();
    outsock.close();
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

static void sendData(zmq::socket_t& sock, const std::string& identity, const CRingFileBlockReader::pDataDescriptor data)
{
  sendHeader(sock, identity);
  s_sendmore(sock, "data");

  size_t dataSize = data->s_nBytes;
  zmq::message_t descriptor(data, sizeof(CRingFileBlockReader::DataDescriptor), freeData);
  zmq::message_t dataBytes(data->s_pData, dataSize, freeData);

  sock.send(descriptor, ZMQ_SNDMORE);
  sock.send(dataBytes, 0);

  
}

static size_t bytesSent(0);
static int  sendChunk(zmq::socket_t& sock, const std::string& identity, CRingFileBlockReader& reader,  size_t nItems)
{
  CRingFileBlockReader::pDataDescriptor pDesc =
    reinterpret_cast<CRingFileBlockReader::pDataDescriptor>(malloc(sizeof(CRingFileBlockReader::DataDescriptor)));

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
    zmq::context_t& context(getContext());
    zmq::socket_t broker(context, ZMQ_ROUTER);
    int linger(0);
    broker.setsockopt(ZMQ_LINGER, &linger, sizeof(int));
    
    FILE* pFile;

    broker.bind("inproc://sender");

    CRingFileBlockReader reader(fileName);


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
  pthread_t writer;

  if (argc != 6) {
    std::cerr << "Incorrect number of command line parameters.\n";
    std::cerr << "Usage:\n";
    std::cerr << "    ringblockdealer   workers blocksize infile outfile format\n";
    std::cerr << "Where: \n";
    std::cerr << "   workers   - number of worker threads\n";
    std::cerr << "   blocksize - Size of the read for a block of ringitems\n";
    std::cerr << "   infile    - input file of nscldaq11.x ring items\n";
    std::cerr << "   outfile   - outputfile.\n";
    std::cerr << "   format    - Output file format:  root or ring are acceptable\n";
    exit(EXIT_FAILURE);
  }
  
  NBR_WORKERS = atoi(argv[1]);
  CHUNK_SIZE   = atoi(argv[2]);
  fileName     = argv[3];
  outfile      = argv[4];
  std::string strFormat = argv[5];
  
  if (strFormat != "ring" && strFormat != "root") {
    std::cerr << strFormat << " is not a valid output file format\n";
    std::cerr << "Valid formats are 'ring' for NSCL Ringbuffers and\n";
    std::cerr << "root for root files\n";
    exit(EXIT_FAILURE);
  }
  
  workers = new pthread_t[NBR_WORKERS];
  threadBytes = new size_t[NBR_WORKERS];
  threadItems = new size_t[NBR_WORKERS];


  const char*  writerArgs[2] = {
    outfile,
    strFormat.c_str()
  };
  char** writerParams = const_cast<char**>(writerArgs);
  pthread_create(&writer, nullptr, zmqwriter_thread, writerParams);
  
  // Register the threads with the writer.  Doing this here ensures the
  // queues are built before any data can be  sent since there's no workers yet.
  
  zmq::context_t& context(getContext());
  zmq::socket_t& regsocket(makeRegistrationSocket(context));
  for (int i = 0; i < NBR_WORKERS; i++) {
    registerThread(regsocket, i);
  }
  endRegistrations(regsocket);
  regsocket.close();
  
  // start the sender:
  
  pthread_create(&sender, nullptr, sender_task,  nullptr);

  // Start the workers.
  
  for (int worker_nbr = 0; worker_nbr < NBR_WORKERS; ++worker_nbr) {
    pthread_create(workers + worker_nbr, NULL, worker_task, (void *)(intptr_t)worker_nbr);
  }
  
  for (int worker_nbr = 0; worker_nbr < NBR_WORKERS; ++worker_nbr) {
    pthread_join(workers[worker_nbr], NULL);
  }

  pthread_join(sender, nullptr);
  
  pthread_join(writer, nullptr);
  
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
