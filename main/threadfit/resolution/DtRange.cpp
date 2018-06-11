/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins 
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  DTRange.cpp
 *  @brief: Generates a file of testdata over a range of dts.
 */

/**
 *  This program is intended to generate a set of data that can be used to
 *  probe time and amplitude resolution as a functionof time difference as
 *  well as single/double pulse discrimination for fits.
 *
 *  Usage:
 *     DTRange  ntraces o a1 a2 k1 k2 k3 k4 dthigh file
 *
 *   ntraces - number of traces each of single and double hits.  Thuse 2*ntraces
 *             events will be created.
 *   o       - Offset on which the pulses sit.
 *   a1, a2  - Amplitudes of the first and second pulses (a1 is amplitude of the
 *             single pulse).
 *   k1, k2  - Rise and decay time constants for the first (only) pulse.
 *   k3, k4  - Rise and decay time constants for the second pulse.
 *   dthigh  - The top end of the dt range that will be created.  The double
 *             pulse traces will have dt uniformly distributed in the range
 *             [0-dthigh).
 *   file    - File to hold the ring items output by this program.
 *
 *  The code writes ring buffer files.   EventFormat data type
 *  for the structure of this event.
 *
 *  @note we use the functions.h/cpp code in our parent directory to
 *        generate and fit waveforms.
 *  @note the traces we generate for this are noise free in this version.
 *  @note For the experiment we're looking at some good values for K's are:
 *        k1, k3 in the range [.5, .7]
 *        k2, k4 in the range [.065, .072]
 *        These values are based on a few observations of single hit traces and
 *        the knowledge that the rise/fall times are determined by the detector
 *        and electronics chain rather than by the actual detector hits.
 * 
 */

#include <functions.h>
#include <CFileDataSink.h>
#include <Exception.h>
#include <CPhysicsEventItem.h>

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <iostream>
#include <stdint>


const size_t TRACE_LENGTH(500);

// Randomization ranges:

const double OFFSET_RANGE(10);
const double AMPLITUDE_RANGE(100);
const double RISE_TIME_RANGE(0.2);
const double FALL_TIME_RANGE(0.01);
const double T0_RANGE(50.0);            // Range around midpoint of first pulse.


/**
 *  This is the format of an event.  Note that the arrays shown have the
 *  number of elements that there are pulses in the generated event:
 *
 */



struct Event {
    uint32_t               s_isDouble;           // True if event is double.
    DDAS::HitExtension     s_fitinfo;            // Fit results.
    
    double                 s_actualOffset;       // Actual offset.
    DDAS::PulseDescription s_pulses[];           // One or two elements.
};
/**
 * randomize
 *   Produce a random double uniformly distributed in the requested range.
 *
 * @param  low  - low limit of the range (inlcusive but unlikely).
 * @param  hi   - High limit of the range (exclusive -- I think but hardly matters).
 * @return double The uniform random.
 */
static double
randomize(double low, double hi)
{
    double range = (hi - low);
    double result = range*drand48() + low;
    
    return result;
}

/**
 * generatePulse
 *    Generate a trace for a single pulse given its specification.
 *
 * @param a   - Pulse amplitude.
 * @param k1  - Pulse rise time specification.
 * @param k2  - Pulse fall time specification.
 * @param o   - Pulse DC Offset.
 * @param t0  - Pulse time 'position'.
 * @param n   - number fo points in the trace.
 * @return std::vector<uint16_t> the trace.
 * 
 */
static std::vector<uint16_t>
generatePulse(double a, double k1, double k2, double o, double t0, size_t n)
{
    std::vector<uint16_t> result;
    for (int i =0; i < n; i++) {
        double t = i;
        result.push_back(DDAS::singlePulse(a, k1, k2, t0, o, t);
    }
    return result;
}

/**
 * writeEvent
 *    Write an event to file.
 *
 *  @param sink - reference to the sink to which the event is put.
 *  @para  p    - Pointer to the Event struct to write.
 *  @throw CErrnoException (from CDataSink::putItem).
 *  
 * @note We just put 0xffffffffffffffff for the timestamp.
 */
static void
writeEvent(CEventSink& sink Event* p)
{
    // Figure out how big the event is so that we get the
    // constructor right (though I doubt we'll hit 8192 since we don't
    // actually write the traces.
    
    size_t eventSize = sizeof(Event) + sizeof(DDAS::PulseDescription);
    if (p->s_isDouble) eventSize += sizeof(DDAS::PulseDescription);
    
    CPhysicsEvent event(0xffffffffffffffff, 0, 0, eventSize);
    void* pBody = event.getBodyCursor();
    memcpy(pBody, p, eventSize);
    uint_t* pByteBody = reinterpret_cast<uint8_t*>(pBody);
    pByteBody += eventSize;
    event.setBodyCursor(pByteBody);
    event.updateSize();                    // Actually I think setBodyCursor does this?
    
    sink.putItem(event);
}
/**
 *doublePulseEvent
 *   Create, fit and output a double pulse event.
 *   We randomize appropriate values for the parameters of the double pulse,
 *   create and fit a double pulse trace then construct, fill in and write
 *   a double pulse ring item
 *
 *   @param sink - reference to the data sink to which the event must be written.
 *   @param o    - Base value for the offset
 *   @param a1   - Base value for the left pulse.
 *   @param k1   - Base value for the left pulse's rise time constant.
 *   @param k2   - Base value for the left pulse's fall time constant.
 *   @param a2   - base value of the amplitude for the right pulse.
 *   @param k3   - base value of the rise time constant for the right pulse.
 *   @param k4   - base value of the fall time constant for the right pulse.
 *   @param dtmax - Maximum allowed difference in t0's for the two pulses.
 *                 actual dt' will be uniformly distributed between 0 and dtmax.
 */
void doublePulseEvent(
    CDataSink& sink,
    double o, double a1, double k1, double k2
    double a2, double k3, double k4,
    double dtmax
)
{    // Randomize the actual values:
    
    o  = randomize(o, OFFSET_RANGE);
    a1 = randomize(a1 - AMPLITUDE_RANGE, a1 + AMPLITUDE_RANGE);
    k1 = randomize(k1 - RISE_TIME_RANGE, k1 + RISE_TIME_RANGE);
    k2 = randomize(k2 - FALL_TIME_RANGE, k2 + FALL_TIME_RANGE);
    double t0 =  randomize(500 - TO_RANGE, 500 + TO_RANGE);
    
    a2 = randomize(a2 - AMPLITUDE_RANGE, a2 + AMPLITUDE_RANGE);
    k3 = randomize(k3 - RISE_TIME_RANGE, k3 + RISE_TIME_RANGE);
    k4 = randomize(k4 - FALL_TIME_RANGE, k4 + FALL_TIME_RANGE);
    double t1 = randomize(t0, t0  + dtmax);
    
    // Generate the double pulse trace:
    
    std::vector<uint16_t> trace1 = generatePulse(a1, k1, k2, o, to, TRACE_LENGTH);
    std::vector<uint16_t> trace2 = generatePulse(a2, k3, k4, 0.0, t1, TRACE_LENGTH);
    std::vector<uint16_t> trace  = add(trace1.begin(), trace1.end(), trace2.begin());
    
    // Fit the trace:
    
    HitExtension fits;
    lmfit1(
        &fits.onePulseFit, trace,
        std::pair<unsigned, unsigned>(0, trace.size()-1)
    );
    lmfit2(
        &fits.twoPulseFit, trace, std::pair<unsigned, unsigned>(0. trace.size-1),
        &fits.onePulseFit
    );
    
    // Create and output the event.  Remember we have two pulses to describe:
    
    Event* pEvent =
        reinterpret_cast<Event*>(malloc(
            sizeof(Event) + 2 * sizeof(DDAS::PulseDescription)
        ));
    if (!pEvent) {
        throw std::bad_alloc();
    }
    pEvent->s_isDouble = true;
    pEvent->s_fitinfo  = fits;
    pEvent->s_actualOffset = o;
    
    pEvent->s_pulses[0].position = t0;
    pEvent->s_pulses[0].amplitude = a1;
    pEvent->s_pulses[0].steepness = k1;
    pEvent->s_pulses[0].decayTime = k2;
    
    pEvent->s_pulses[1].position = t1;
    pEvent->s_pulses[1].amplitude = a2;
    pEvent->s_pulses[1].steepness = k3;
    pEvent->s_pulses[1].decaytime = k4;
    
    writeEvent(sink, pEvent);
    
    free(pEvent);
}


/**
 * singlePulseEvent
 *    Create, fit and output a single pulse event.  Note that both single and
 *    double pulse fits are done.  The result is a ring item with the body
 *    of type Event above and with only one element in s_pulses.
 *
 *  @param sink - Reference to a CEventSink to which the data are written.
 *  @param o    - Offset around which some randomization will be done.
 *  @param a    - Amplitude of the pulse, around which some randomization will be done.
 *  @param k1   - Rise time constant around which some randomization is done.
 *  @param k2   - Fall time constant around which some randomization is done.
 *
 * @note It's possible for the data sink to throw an exception.  We leave that
 *       for callers up the stack to handle.
 */
static void
singlePulseEvent(COutputSink& sink, double o, double a, double k1, double k2)
{
    // Using the  actual values as starting points, randomize new ones.
    // Randomize the t0 as well
    
    o = randomize(o, OFFSET_RANGE);              // Single ended.
    a = randomize(a - AMPLITUDE_RANGE, a + AMPLITUDE_RANGE);
    k1 = randomize(k1 - RISE_TIME_RANGE, k1 + RISE_TIME_RANGE);
    k2 = randomize(k2 - FALL_TIME_RANGE, k2 + FALL_TIME_RANGE);
    double to = randomize(500 - TO_RANGE, 500 + T_RANGE);
    
    // Generate the trace for fitting:
    
    std::vector<uint16_t> trace = generatePulse(a, k1, k2, o, to, TRACE_LENGTH);
    
    // Get the fit results:
    
    
    HitExtension fits;
    lmfit1(
        &fits.onePulseFit, trace,
        std::pair<unsigned, unsigned>(0, trace.size()-1)
    );
    lmfit2(
        &fits.twoPulseFit, trace, std::pair<unsigned, unsigned>(0. trace.size-1),
        &fits.onePulseFit
    );
    
    // Build the event:
    
    Event* pEvent =
        reinterpret_cast<Event*>(malloc(
            sizeof(Event) + sizeof(DDAS::PulseDescription)
        ));
    if (!pEvent) {
        throw std::bad_alloc();
    }
    pEvent->s_isDouble            = false;
    pEvent->s_fitInfo             = fits;
    pEvent->s_actualoffset        = o;
    pEvent->s_pulses[0].position  = t0;
    pEvent->s_pulses[0].amplitude = a;
    pEvent->s_pulses[0].steepness = k1;
    pEvent->s_pulses[0].decayTime = k2;
    
    writeEvent(sink, pEvent);
    
    free(pEvent);
}


/**
 *  convertInt
 *     Convert a const char* into a positive integer.
 *     If the conversion fails, std::invalid_argumnt is thrown.
 *
 *  @param src  - The string to convert.
 *  @param msg  - Message to throw via invalid_argument on failure.
 *  @return int - Result.
 *  @throws std::invalid_argument
 */
static int
convertInt(const char* src, const char* msg)
{
    char* endptr;
    long int result = strtol(src, &endptr, 0);
    if ((result == 0) && (endptr == src)) {
        throw std::invalid_argument(msg);
    }
    if (result <= 0) {
        throw std::invalid_argument(msg);
    }
    return result;
}
/**
 * convertDouble
 *   This is just like convertInt, however the input string is converted as
 *   a positive double precision value.
 *  @param src  - The string to convert.
 *  @param msg  - Message to throw via invalid_argument on failure.
 *  @return int - Result.
 *  @throws std::invalid_argument
 */
static double
convertDouble(const char* src, const char* msg)
{
    char* endptr;
    long int result = strtod(src, &endptr, 0);
    if ((result == 0) && (endptr == src)) {
        throw std::invalid_argument(msg);
    }
    if (result <= 0) {
        throw std::invalid_argument(msg);
    }
    return result;
}

/**
 * usage
 *    Output usage information to a stream.
 *
 *  @param str   - output stream to which the message is put.
 *  @param msg   - Message that prefixes the usage info.
 */
static void
usage(std::ostream& str, const char* msg)
{
    str << msg << std::endl;
    str << "Usage\n";
    str << "   DTRange  ntraces o a1 a2 k1 k2 k3 k4 dthigh file\m";
    str << "Where\n";
    str <<     "ntraces - number of traces each of single and double hits.  \n";
    str << "              Thus 2*ntraces events will be created.\n";
    str << "    o       - Offset on which the pulses sit.\n";
    str << "    a1, a2  - Amplitudes of the first and second pulses (a1 is \n";
    str << "              amplitude of the single pulse).\n";
    str << "    k1, k2  - Rise and decay time constants for the first (only)\n";
    str << "              pulse.\n";
    str << "    k3, k4  - Rise and decay time constants for the second pulse.\n";
    str << "    dthigh  - The top end of the dt range that will be created.\n";
    str << "              The double pulse traces will have dt uniformly distributed\n";
    str << "              in the range [0-dthigh).\n";
    str << "    file    - File to hold the ring items output by this program.\n";
    str << "NOTE:\n";
    str << "  For scintillators and 500MHz digitizers, some good values for K's are:\n";
    str << "   k1, k3 in the range [.5, .7]\n";
    str << "   k2, k4 in the range [.065, .072]\n";
    
}

/**
 * Entry point
 */

int main(int argc, char** argv)
{
    if (argc != 11) {
        usage(std::cerr, "Incorrect number of command parameters");
        exit(EXIT_FAILURE);
    }
    // Let's pull out the parameters; Conversion failures result in
    // an invalid argument.
    
    int ntraces;
    double o;
    double a1, a2;
    double k1, k2, k3, k4;
    double dthigh;
    const char* file;
    
    try {
        ntraces = convertInt(argv[1], "ntraces must be a valid, positive integer");
        o       = convertDouble(argv[2], "o must be a valid positive double\n");
        a1      = convertDouble(argv[3], "a1 must be a valid positive double\n");
        a2      = convertDouble(argv[4], "a2 must be a valid positive double\n");
        k1      = convertDouble(argv[5], "k1 must be a valid postiive double\n");
        k2      = convertDouble(argv[6], "k2 must be a valid positive double\n");
        k3      = convertDouble(argv[7], "k3 must be a valid positive double\n");
        k4      = convertDouble(argv[8], "k4 must be a valid positive double\n");
        dthigh  = convertDouble(argv[9], "dthigh must be a valid positive double\n");
        file    = argv[10];
    }
    catch (std::invalid_argument& e) {
        usage(std::cerr, e.what())
        exit(EXIT_FAILURE);
    }
    catch (...) {
        usage(
            std::cerr,
            "Some unexpected exception type was caught processing command parameters"
        );
        throw;                    // May provide clues about the type of exception.
    }
    /*
     * Create the file data sink:
     */
    CFileDataSink* sink(0);
    try {
        sink = new CFileDataSink(file);
    }
    catch (std::string msg) {
        usage(std::cerr, msg.c_str());
        exit(EXIT_FAILURE);
    }
    catch (...) {
        usage(
            std::cerr,
            "Some unexpected exception type was caught creating the CFileDataSink object"
        );
        throw;
    }
    /** Generate the events;
     *  We'll alternately generate and fit single and double pulse events.
     *  Regardless of the event type, we'll do both fit types so that
     *  we can see how to discriminate between them.
     */
    // Seed the random number generator with the time_t:
    
    time_t seed = time(nullptr);
    srand48(static_cast<unsigned long>(seed));

    // Make the events
    
    try {
        for (int i = 0; i < ntraces; i++) {
            singlePulseEvent(sink, o, a1, k1, k2);
            doublePulseEvent(sink, o, a1, k1, k2, a2, k3, k4, dthigh);
        }
    catch (CException& e) {
        std::cerr << "Caught an NSCLDaq Exception generating events:\n";
        std::cerr << e.ReasonText() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (std::bad_alloc& e) {
        std::cerr << "Unable to allocate dynamic memory generating events:\n";
        std::cerr << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Caught some unexpected exception type generating events\n";
        throw;
    }
    
    
    exit(EXIT_SUCCESS);
}