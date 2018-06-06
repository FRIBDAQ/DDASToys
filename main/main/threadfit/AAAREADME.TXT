This directory tree contains the code needed to build, run, modify etc. the
threaded analyzer.  The code assumes an installation of NSCLDAQ and DDAS.  Any
11.x version of nscldaq should work, though DDAS 2.2 is needed (to get the
trace lengths right).

The Makefile assumes you've set up the environment variables for these beasts.
For example

. /usr/opt/daq/11.3-001/daqsetup.bash
. /usr/opt/ddas/2.2/ddassetup.bash

make

RUNNING IT UNMODIFIED
=====================

To run the program:

./ringblockdealer workers blocksize infile outfile

workers - The number of worker threads that do fitting.  This should be like
the number of CPUS shown on cat /proc/cpuinfo.  Note that you'll only get
linear speedup for the number of physical cores present.  Hyperthreading,
however may add another 10%.

blocksize - The code avoids I/O throughput limitations on input by reading
large blocks of data.  This parameter specifies how large those blocks are.
For genesis (28 hypyerthreaded CPUs), 300000 seems a good number.

infile - The input event file to fit.  Note that for multisegmented runs, it
may be advantageous to cat all segments together to a single event file, or
create a script that can sequentially locate and analyze each segment.

outfile - Name of the output file (see OUPUT FORMAT below).  Note that only
physics items will be output to this file.

Note as well that the actual performance of this program will depend on the
predicate you use to define which channels get fit and how large the region of
interest for each fit is (see HOW TO MODIFY THIS).

INTERNAL STRUCTURE
==================

This section describes the parts of the software and how they hang
together. Some of this information is required knowledge, if you want to use
this as a framework for thread parallel computing.  We'll start with
communication mechanisms.  From there we'll proceed on to the thread structure
itself and how communication between the threads is performed.  Finally, we'll
look in depth at the worker thread and how that's put together so that you can
know where the easy modification points are and where you should, probably,
leave things alone.  As we hit that final chunk we'll also go over the files
that, put together, make up the program.

Communication:
   The ZMQ communications middleware is used to do message based communications
   between threads.  This was used for several reasons:
*  ZMQ provides support for several common communcations patterns (for
   example fan out and fan in which are crucial for parallel computing.
*  ZMQ allows for future systems to be dynamic.  That is; if this framework
   is recast as process parallel, not only could it run on a cluster, but
   workers could be dynamically added or removed as resources become available
   or unavailable.
*  (not the best reason but): I already had some good experience with ZMQ
   as a communications framework in NSCLDAQ 12's development and was happy with
   it.

  Three communications patterns are used:
  *   ROUTER/DEALER  This pattern is a fan out, pull pattern. The ROUTER
  provides data to the DEALERs.  Dealers send messages to ROUTER's which must
  then respond.  ROUTER/DEALER does this isn a point to point manner where
  messages get tagged by source so that this can be extended to a multi-layer
  distribution system.
  *   REQ/REP - This is the more typical point to point client server pattern,
  A client makes a request of a server which then replies.  This is used to
  register worker threads with the output thread in a synchronized manner
  before the output threads are started.   This allows the output thread to
  ensure that it has data queues set up for all workers so that it can properly
  re-sort the output data.
  *  PUSH/PULL - This can be used either as a fan-in or fan-out pattern.
  The framework uses this to fan-in processed data items to the single
  sorter/output thread.

Thread Structure and Interaction:

The framework has four types of threads:

*  The main thread does command line parameter processing, starts the
   sorter/output thread, registers the set of worker thread numbers with it,
   starts the sender and the workers and the waits for all of the threads to
   complete.  Prior to exiting it reports the work done by each worker thread.
   This thread's entry point is main() in ringblockdealer.cpp
*  The sender thread responds to requests for data as follows:  If there's
   unread data in the input file, a block of data is read and the complete ring
   itmes in that data are sent to the worker (the partial ring item is
   assembled into the front of the next read).  If there's no more data, the
   worker is notified that there's no more data and expected to exit.  Once all
   workers have been notified of the end file condition, the sender exits.
   This thread's entry point is sender_task in ringblockdealer.cpp
*  Worker tasks request data from the sender until they're notified there's no
   more data, at which point they exit.  The data units received consist of a
   descriptor and the bulk data containing a set of complete ring items.  The
   descriptor contains the number of bytes of actual data, and the number of
   ring items in that actual data.

   The worker tasks create an analyzer (CDDASAnalyzer) object and a
   ChannelSelector predicate, and a CZMQRingOutputter and hooks them all
   together.
   - The ChannelSelector provides for only fitting a list of channels.
   - The CZMQRingOutputter object is used by the analyzer to transmit data to
     the output/sorting thread.
   -  The CDDASAnalyzer does the actual fitting (conditionalized on the
      predicate) and uses its outputter to transmit its results.

    Entry point for worker threads is worker_task() in ringblockdealer.cpp and it
    uses processRingItems() in that same file to iterate over the ring items in a
    work item.

    Other files:
    CZMQRingOutputter.{h,cpp} - uses the sorter/writer API to transmit data to
        sorter outputter API.  Note this is derived from Outputter defined in Outputter.h
    CDDASAnalyzer.{h,cpp} Performs the analysis of each ring item.  It uses a
    'predicate' to determine both if a fit shouild be done and, if so, what
    range of the trace the fit should be done over (this latter refinement was
    added to deal with an experiment which had a sinusoidal oscillation on the
    back end of the waveforms in some channels).  Predicates must be derived
    from CDDAnalyzer::FitPredicate the ChannelSelector predicate is defined and
    implemented in ringblockdealer.cpp (probably not the best choice).
*   The sorter/outputter thread accepts analyzed ring items from worker
    threads.  It resorts them by timestamp and outputs the sorted ring items to
    file.  Its entry point is zmqwriter_thread() in zmqsortedwriter.cpp.  It
    provides an API to send it appropriate messagse in zmqsortedwriter.h
    (future work should extract all thread types into separated files).  Note
    that at this time, ring items are sent by pointer.  For process
    parallelism, clearly the data must be sent as block data as pointers don't
    cross process boundaries.  A key file is CSortingOutputter.{h,cpp} which
    actually does the timestamp resorting and output.

A note about the sorting algorithm in CSortingOutputter.h/cpp.   It's a bit
different than that used by the NSCLDAQ eventbuilder as that algorithm wound up
being a performance bottleneck for large numbers of worker threads.   Here's
how it works in a nutshell:

*   At startup, the worker threads are registered.  Registration means the
    creation of a data queue for the worker thread's ring items and a mapping
    between the thread number (thread number not id) and its queue.  This
    mapping was intended to allow threads to self register .. in which case
    they might register 'out of order'.
*   Data items are queued by the sorter/sender thread.  The fact that blocks of
    data are shipped to worker threads means that each queue will have 'runs'
    of ordered data.  This fact is _very important to the algorithm used.
    After a data item is entered; flushItems() is called to
    output any items that can be output.
*   As threads finish processing they are shutdown, which is done by putting a
    special marker item in their queue.
*   flushItems only will output data if active queues (those with a mapping
    from thread number to queue) have data.  The queue with the oldest item at its
    head is identified, as is the second oldest timestamp value at the front of
    all queues (if there's only one queue, this is set to MAX_UNIT64).
    Data are output from that queue until the item at its head is
    greater than that second oldest timestamp (now the first oldest timestamp).
*   As end items are encountered or prior to any operation that iterates over
    all queues, reapQueues is called.  This method removes thread to queue
    mappings for any queue that will never have any more data.  This makes that
    queue invislbe to flushItems() for future scans.

This algorithm seems able to keep up with 56 worker threads beating on it.


OUTPUT FORMAT
============

The program output is a file containing only PHYSICS_EVENT ring items.   Each
ring item contains a set of event fragments.  Each fragment contains either a
DDASHit or a DDASHit with an extension that contains the fitting parameters.
This extension is described by the type HitExtension in functions.h

If you point the DDASHitUnpacker at this file to untangle hits, it will
complain bitterly for the hits that have this extension.  Specifically it will
thing the lengths of those hits are corrupt as a result of the extension.

I've provided the FitHitUnpacker (FitHitUnpacker.h,cpp) which determines if a
hit has an extension.  If not, DDASHitUnpacker is used to unpack the hit.  If
so, then DDASHitUnpacker is only handed the original part of the hit to unpack
and FitHitUnpacker unpacks the extensions.

The result is a DDASFitHit (DDASFitHit.h)  Which is derived from DDASHit and
provides you with the extra methods:

hasExtension()  - returns true if the hit has a fit extension.
getExtension()  - Returns the hit extension (throws std::logic_error if there'
not an extension).


HOW TO MODIFY THIS
==================

The actual structure of this program allows it to be used as a pretty generic
framework for filtering/transforming data.  By filtering/transforming, I mean
any process that takes a file of ring items as input and produces ring items as
output.  What I'll cover:

*   Controlling the fits in CDDASAnalyzer via its fit predicates
*   Modifying CDDASAnalyzer
*   Modifying what's output
*   Doing something completely different

Controlling which fits are done:

Fits are expensive.  That's why we've parallelized this code in the first
and why we'll look at cluster parallelism, GPU acceleration and FGPA
acceleration as well.   CDDASAnalyzer provides the ability to control which
are done.  You can define a class derived from CDDASAnalyzer::FitPredicate,
instantiate objects of that class and register them as single or double fit
predicates.

The predicate must implement a function call operator with the following call
signature:

 virtual std::pair<unsigned, unsigned> operator()(
            const FragmentInfo& frag, DAQ::DDAS::DDASHit& hit,
            const std::vector<uint16_t>& trace
        )

The parameters are:
   - frag - the fragment information for the event fragment we might fit.
   - hit  - the original DDAS hit being analyzed.
   - trace - the trace to potentially fit.

This method is called prior to performing fits (see below). The result is a
pair of indices into the trace indicating the region of interest to fit.  If
the region of interest is emtpty, no fit is performed. 

CDDASAnalyzer provices:

    void setSingleFitPredicate(FitPredicate* p);
    void setDoubleFitPredicate(FitPredicate* p);

To register predicates.  That control if a single pulse fit and a double pulse
fit respectively will be performed:

Prior to performing a single pulse fit, the single fit predicate is called.  It
determines if a fit will be performed and if so the range of the trace that
will be fit.  Note that:
*   If no predicate is supplied the fit will be performed over the entire range
    of the trace.
*   If the single fit is not performed, then neither is the double fit as the
    single fit is used to provide information to the double fit used to compute 
    starting points for the fit parameters.

The double pulse fit predicate is called _after_ the single pulse fit is
performed and functions in a similar way to control the double pulse fit.  Note
that at the time the double pulse fit predicate is called, FitHitUnpacker can
be used to access the results of the single pulse fit.

If the single pulse fit is performed, but not the double pulse fit, all members
of the twoPulseFit par tof the fit extension will be zero.

Modifying DDASAnalyzer:

CDDASAnalyzer can be considered an example (future it will be derived from
some base class).  You can modify its processPhysicsItem appropriately to make
it performs some other data analysis/transformation.  Note that the output
should be some vector of fragments which is then passed to the outputter.
Current versions of the outputter are aware that some fragments may contain
dynamic data (future - should move that knowledge back into the analyzer) and
free those data.    This may require modification depending on what the
outputter gets.

Modyfhing what's output:

The output thread uses a CDataSink to output data to file.  Currently
zmqwriter_thread() instantiates a  CFileDataSink. See

http://docs.nscl.msu.edu/daq/newsite/nscldaq-11.2/r24076.html

CDataSink, the base class is described in:

http://docs.nscl.msu.edu/daq/newsite/nscldaq-11.2/r23985.html

You can write a new class derived from CDataSink which writes data in a
different format.  (Future - why not write a CRootDataSink which makes root
files like DDASDumper does? -- maybe that's simpler than a new DDASDumper).
Note that putItem is the only method used by CSortingOutputter, though for
generality put() could be implemented by reconstructing a ring item and calling
putItem().

Doing Something completely different:

Modify processRingItems to do whatever you wnat and to send it to whatever you
want.  That may require modifying the output thread if the output from
processRingItems is not itself a ring item.

SPECTCL (example)
=================

The SpecTcl directory provides sample code that shows how to unpack hits and
create a set of parameters from any fits that are presetn.  See
FitEventProcesor.cpp and FitEventProcessor.h as that's where the real work is
done.  You can also use that as a model for, given the body of a ring item, how
to unpack those items.


OTHER TOOLS
============

DisplayExtensions.cpp compiles to DisplayExtensions.  It provides a tool to
visualize the waveforms, single and double pulse fits.
*  It relies on the Tcl scripts in the plotting directory.
*  Those scripts rely on the Plotchart package which is part of tklib (standard
Tk library).
*  The ReportHitFits function in DisplayExtensions.cpp can be modified to
select which fits are displayed.
*  The legend/title on the plots is incorrect. The red trace is the original
data.  The black trace is the single pulse fit and the blue trace is the double
pulse fit.

The fit parameters are reported to stdout as each hit is analyzed as is the
crate/slot/channel of the hit.  Hits without extensions are ignored.


MAKING THIS A CLUSTER DISTRIBUTED PROGRAM
=========================================

Several work items would be required to turn this into a cluster distributed
program.

*  The data transmitted from workers to outputter would need to include the
   bulk data in the ring items.
*  There are a few cases where shared memory information needs to be
   compartementalized.  Specifically, the sender thread uses NBR_WORKERS which
   would need to be passed to it in some different way.  For exmaple, as a
   start up parameter.
*  It's a good idea to pull each thread out into at least one separate
   implementation file with communication encapsulated in APIs.
*  For communication to be established, it's necessary to know which node the
   sender and outputter processes are running.  One way to do this is to
   implement the program on top of MPI as it starts up.   The MPI rank
   mechanmism could be used for this as follows:
   Rank 0 becomes a manager process.
   Rank 1 is the sender.
   Rank 2 is the writer
   All other ranks are the workers.

   At startup, rank1 and rank2 inform rank 0 which node of the URI's for their
   services.
   Rank 0 registers all the other ranks as workers via ZMQ to the
   rank 2 process as it does now.
   Rank 0 then sends the URIs of the sender and
   writer needed by the workers to a communicator to ranks 3-n+3, the n
   workers.
   Receivers are waiting for that MPI message(s) and use them to do form the
   ZMQ connections they now use.
   From then on everything is as it is now..


Probably this is a pretty decent amount of work to do and get right but pretty straightforward.
