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

const int NBR_WORKERS = 20;
const int CHUNK_SIZE  = 1024*1024;
const uint64_t PIPE_DEPTH = 1;

static size_t threadBytes[NBR_WORKERS];

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
    int total = 0;
    while (1) {
        //  Tell the broker we're ready for work

      int credits = PIPE_DEPTH;
      while(credits) {
	//	std::cout << "Sending a fetch\n";
	s_sendmore(worker, "");
	s_sendmore(worker, "fetch");
	s_send(worker, ChunkSize.str());                 // Sizse of workload
	credits--;
      }
      //  Get workload from broker, until finished
      s_recv(worker);     //  Envelope delimiter
      std::string type = s_recv(worker);
      zmq::message_t workload;
      worker.recv(&workload);
      //      std::cout << "Worker " << args << " Got a workitem : " << workload.size() << std::endl;

      if (type == "eof") {
	//        std::cout << "Completed: " << total << " tasks" << " Bytes: " << bytes << " " 
	//		  <<  args << std::endl;
	break;
      } else {
	if (workload.size() != CHUNK_SIZE) {
	  //	  std::cout << "Midget chunk " << workload.size() << " " << type << std::endl;
	  const char* pData = reinterpret_cast<const char*>(workload.data());
	  for (int i = 0; i < workload.size(); i++) {
	    std::cout << pData[i] << " ";
	  }
	  //	  std::cout << std::endl;
	}
      }
      total++;
      
      credits++;
      bytes += workload.size();
      threadBytes[thread] += workload.size();
      
        //  Do some random work
    }
    worker.close();
    // std::cout << "Exiting worker " << args << std::endl;
    return NULL;
}




void freechunk(void* pData, void* pHint)
{

  free(pData);
}
static size_t bytesSent(0);
static int  sendChunk(zmq::socket_t& sock, FILE* pFile,  const std::string& identity, size_t size)
{

  if (size != CHUNK_SIZE) {
    std::cerr << " Asked to send a small chunk " << size << std::endl;
  }
  void* pData = nullptr;


  if (size > 0) {
    // Send actual data chunks.


    pData = malloc(size);
    assert(pData);
    
    size_t actualSize = fread(pData, 1, size, pFile);
    
    
    
    // Make a zero copy message for pData and send it with freechunk as the releaser:
    
    if (actualSize > 0) {
      s_sendmore(sock, identity);
      s_sendmore(sock, "");
      //  std::cerr << "Send 'a chunk of size: " << size << " to " << identity << std::endl;
      s_sendmore(sock, "data");

      if (actualSize != size) {
	std::cout << "Sent midget chunk " << actualSize << std::endl;
      }
      zmq::message_t msg(pData, actualSize, freechunk);
      sock.send(msg, 0);
      
    } 

    return actualSize;
   

  } else {
      s_sendmore(sock, identity);
      s_sendmore(sock, "");

    s_sendmore(sock, "eof");
    // Send End request
    s_send(sock, "");
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
    srandom((unsigned)time(NULL));
    pFile = fopen("testdata", "r");


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
	    //	     std::cout << "Got request from worker "
	    // << identity << " : " <<  command << ", " << size << std::endl;
        }
	if (!done) {
	  int status =  sendChunk(broker, pFile,  identity, atoi(size.c_str()));
	  if (status == 0) {
	    done = true;
	    //	    std::cout << "File all sent\n";
	    sendChunk(broker, pFile, identity, 0);
	    ++workers_fired;
	    if (workers_fired == NBR_WORKERS) break;
	  }
        } else {
	  sendChunk(broker, pFile, identity, 0);
	  // sendChunk(broker, pFile, identity, 0);
	  //	  std::cout << "Fired worker " << workers_fired << std::endl;
	  if (++workers_fired == NBR_WORKERS)
	    break;
        }
    }
    //    std::cout << "All workers got fired\n";
    // std::cout << "Waking up\n";
    broker.close();

}

//  While this example runs in a single process, that is just to make
//  it easier to start and stop the example. Each thread has its own
//  context and conceptually acts as a separate process.
int main() {
    pthread_t workers[NBR_WORKERS];
    pthread_t sender;

    pthread_create(&sender, nullptr, sender_task,  nullptr);
    
    for (int worker_nbr = 0; worker_nbr < NBR_WORKERS; ++worker_nbr) {
        pthread_create(workers + worker_nbr, NULL, worker_task, (void *)(intptr_t)worker_nbr);
    }
    
    for (int worker_nbr = 0; worker_nbr < NBR_WORKERS; ++worker_nbr) {
        pthread_join(workers[worker_nbr], NULL);
    }
    pthread_join(sender, nullptr);

    size_t totalBytes(0);
    
    for (int i = 0; i < NBR_WORKERS; i++) {
      std::cout << "Thread " << i << " processed " << threadBytes[i] << std::endl;
      totalBytes += threadBytes[i];
    }

    std::cout << "totalBytesProcessed  " << totalBytes << std::endl;
    
    return 0;
}
