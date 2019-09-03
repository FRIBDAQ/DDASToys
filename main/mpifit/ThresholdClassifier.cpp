/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Giordano Cerriza
	     NSCL
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

/** @file:  ThresholdClassifier.cpp
 *  @brief: Classifier that requires one of some present set of channels be above threshold.
 */

#include <CRingItemMarkingWorker.h>
#include <CRingItem.h>
#include <DataFormat.h>
#include <map>
#include <string>
#include <stdexcept>
#include <DataFormat.h>
#include <FragmentIndex.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
////////////////////////////// local trim functions   /////////////////////////

static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}


////////////////////////////////////////////////////////////////////////////////

/**
 *  @class CThresholdClassifier
 *     This classifier takes a set of channel specifications from file.
 *     If none of the specified channels are present in the event, the
 *     event is marked with a 1.
 *     If any of the specified channels are present in the event and at least one
 *     of them is above its threshold value, the event is marked with a 1.
 *     If any of the specified channels are present in the event and none of them
 *     are above threshold, the event is marked with a 0.
 *
 *     This class is suited for marking events in e17011 for PIN filtering.
 *     If events have PIN values present but too small, presumably these events
 *     are from heavy ions rather than from decays.
 *
 *     The file loaded is referred to by the environment variable:
 *     THRESHOLD_FILE
 *
 *     The file formt is;
 * \verbatim
 *     crate  slot  channel   threshold
 * \endverbatim
 *
 *     Where crate, slot, channel identify a channel to check (slot is slot id in
 *     the event fragment) and threshold is the value the channel energy  must
 *     at least be to set the event classifier to 1.
 *     
 */
class CThresholdClassifier : public CRingMarkingWorker::Classifier
{
private:
    std::map<unsigned, unsigned> m_channelThresholds;
    int  m_nScaledown;
    int  m_nRejects
public:
    CThresholdClassifier();
    
    virtual uint32_t operator()(CRingItem& item);
private:
    std::string getThresholdFile(const char* envName);
    void readThresholdFile(const char* filename);
    int  getScaledown(const char* envName);
    std::string isComment(std::string line);
};

/**
 * constructor
 *    Read the threshold file into the m_channelThresholds Map.
 *    This map is a lookup by channel in the file returning the
 *    threshold value.  The crate/module/channel are encoded into a 12 bit
 *    number as they are in the header for the data.  Thus we can just take
 *    the bottom 12 bits of the first word of the header and use that to
 *    as the map index.
 */
CThresholdClassifier::CThresholdClassifier()
{
    try {
        std::string filename = getThresholdFile("THRESHOLD_FILE"); // Throws on error.
        readThresholdFile(filename.c_str());
        m_nScaledown = getScaledown("REJECT_SCALEDOWN");
    } catch (std::exception& e) {
        std::cerr << "Failed initializing classifier: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}
/**
 * operator()
 *    Given a ring item that consists of event built fragments from DDAS:
 *    1.  Run over all fragments looking for matches in the m_channelThresholds
 *        map.  Those get counted.
 *    2.  Those which are found have their energies compared with the threshold
 *        in m_channelThresholds.  The number of over thresholds is counted.
 *
 *   -  If there are no map index matches the result is set to 1 indicating the
 *   -  If there is at least one match and there's at least one made threshold,
 *      the result is 1.
 *   -  If there is at least one match but no thresholds m_nRejects is incremented.
 *      if m_nScaledown is zero,  the result is 0.  Otherise, if
 *      m_nRejects % m_nScaledown == 0 the result is 2 else 0.  Allowing
 *      for accepting scaled down rejections.
 */
uint32_t
CThresholdClassifier::operator()(CRingItem& item)
{
    FragmentIndex frags(static_cast<uint16_t*>(item.getBodyPointer()));
    
    /* Rather than use the DDASHit which has a pile of overhead we know
     * The fragment bodies are ring items with body headers and the bodies are:
     *  +--------------------------------+
     *  |  uint32_t item size            |
     *  +--------------------------------+
     *  | uint32_t module type code      |
     *  +--------------------------------+
     *  | Module id info (among others)  | uint32_t
     *  +--------------------------------+
     *  |  Don't really care             |
     *  +--------------------------------+
     *  |  Really don't care             |
     *  +--------------------------------+
     *  | Energy in bottom 16 bits       |
     *  +--------------------------------+
     *  |.....                           |
     */
    
    size_t nFrags = frags.getNumberFragments();
    int    matches(0);
    int    goodValues(0);
    for (int i =0; i < nFrags; i++) {
        pRingItem pFrag =
            reinterpret_cast<pRingItem>(frags.getFragment().s_itembody);
        uint32_t* pHit = reinterpret_cast<uint32_t*>(
            pFrag->s_body.u_hasBodyHeader.s_body    
        );
        int channelId = pHit[2] & 0xfff;    // The encoded hit.
        auto p = m_channelThresholds.find(channelId);
        if (p != m_channelThresholds.end()) {
            matches++;                    // We've got a match.
            int e = pHit[5] & 0xffff;
            if (e >= p->second) {
                return 0;                 // We only need one.
            }
        }
    }
    //  If there are no matches return 0:
    
    if (matches == 0) return 0;
    m_nRejects++;                       // Count a reject.
    if (m_nScaledown) {
        if (m_nRejects % m_nScaledown) {
            return 2;                  // Scaled down reject to keep.
        } else {
            return 1;                  // reject.
        }
    }
    return 1;                         // reject.
    
}
/**
 * getThresholdFile
 *    Get the name of the threshold file:
 *
 *  @param envName - name of environment variable with the channel threshold file.
 *  @return std::string - translation of the name.
 *  @throw std::invalid_argument -if there's not a translation.
 */
std::string
CThresholdClassifier::getThresholdFile(const char* envName)
{
    const char* pFilename = getenv(envName);
    if (pFilename) {
        return std::string(pFilename);
    } else {
        std::string msg(envName);
        msg +=  " is not an environment variable.  It must be set to point to the channel threshold file";
        throw std::invalid_argument(msg);
    }
}
/**
 * readThresholdFile
 *   Read the threshold file into m_channelThresholds:
 *   -   Empty/blank lines are comments.
 *   -   Lines with the first non-whitespace character a # are comments.
 *   -   Other lines have four numeric fields in order:
 *       crate, slot, channel, threshold.
 *
 *  @param filename -name of the file to read
 *  @throw std::invalid_argument - if the file has invalid format, or can't be opened.
 */
void
CThresholdClassifier::readThresholdFile(const char* filename)
{
    std::ifstream f(filename);
    if (f.fail()) {
        std::string msg("Threshold file: ");
        msg += filename;
        msg += " cannot be opened.";
        throw std::invalid_argument(msg);
    }
    
    while (!f.eof()) {
        std::string line("");
        std::geline(f, line, '\n');
        line = isComment(line);
        if (line != "") {                     // Not a comment.
            int crate, channel, slot, thresh;
            std::stringstream s(line);
            s >> crate >>  slot >> channel >> thresh;
            if (s.fail()) {
                std::string msg("Incorrect format of line in threshold file: '");
                msg += line;
                msg += "'";
                throw std::invalid_argument(msg);
            }
            
            // create the lookup value:
            
            int key = (crate << 8) | (slot << 4) | channel;
            m_channelThresholds[key] = thresh;
        }
    }
}
/**
 * getScaledown
 *    Returns the scaledown value:
 *    -   If there's an integer translation of the env variable,
 *       that's the scaledown
 *    -  If there's no translation, 0 is returned (no scaleddown rejects).
 *    -  If there's a non integer translation - that's an error.
 * @param envname - Name of env variable that  has the scaledown.
 * @return int - scaledown value.
 * @throw std::invalid_argument - if the variable has no integer translation.
 */
int
CThresholdClassifier::getScaledown(const char* envname)
{
    const char* pValue = getenv(envname);
    if (!pValue) return 0;
    const char* endp
    long value = strtol(pValue, &endp, 0);
    if (endp == pValue) {
        throw std::invalid_argument("Scaledown value must be an integer");
    }
    return value;
}
/**
 * isComment
 *   - Strips whitespace off the front and back of the input string.
 *   - Empty strings are then comments.
 *   - Strings whose first characters are '#' are comments.
 *
 * @param line - the input string.
 * @return std::string - empty if a comment else the stripped line.
 */
std::string
CThresholdClassifier::isComment(std::string line)
{
    std::string result = line;
    result = trim(result);                 // trim modifies.
    if (result.empty()) return result;
    if (result[0] == '#') return "";
    return result;
}

/////////////////////////////////////////////////////////////////////////////
// Factory method:


extern "C" {
    CRingMarkingWorker::Classifier* createClassifier()
    {
        return new CThresholdClassifier;
    }
}