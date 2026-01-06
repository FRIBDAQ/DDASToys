/*
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2016.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Author:
	     Aaron Chester
	     Facility for Rare Isotope Beams
	     Michigan State University
	     East Lansing, MI 48824-1321
*/

#include <cppunit/extensions/HelperMacros.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Asserts.h"
#include "Configuration.h"

using namespace ddastoys;

namespace std {
    template<class T>
    ostream& operator<<(ostream& stream, const vector<T>& vec)
    {
	stream << "{ ";
	for (auto& element : vec ) stream << element << " ";
	stream << "}";

	return stream;
    }

    template<class T, long unsigned int N>
    ostream& operator<<(ostream& stream, const array<T,N>& vec)
    {
	stream << "{ ";
	for (int i = 0; i < N; ++i) stream << vec[i] << " ";
	stream << "}";

	return stream;
    }
}

class ConfigurationTests : public CppUnit::TestFixture
{
public:
    char* origEnv;
    Configuration config;
    std::string configfilepath;
    std::ofstream configfile;
    std::string templatefilepath;
    std::ofstream templatefile;
    
public:
    CPPUNIT_TEST_SUITE(ConfigurationTests);

    CPPUNIT_TEST(checkmap);      // check the channel map for channels to fit
    CPPUNIT_TEST(gettracelen);   // check trace length
    CPPUNIT_TEST(getlimits);     // check fit limits
    CPPUNIT_TEST(getsat);        // check saturation value
    CPPUNIT_TEST(getmodelpath);  // check model path
    CPPUNIT_TEST(getlist);       // check list of models
    CPPUNIT_TEST(getmodelshape); // check model shape
    CPPUNIT_TEST(gettemplate);   // check template data
    CPPUNIT_TEST(getalign);      // check template aligment point

    CPPUNIT_TEST_SUITE_END();
    
public:
    void setUp()
	{
	    origEnv = getenv("FIT_CONFIGFILE");

	    configfilepath = "/tmp/fitconfig.txt";
	    templatefilepath = "/tmp/template.txt";	
	    
	    configfile.open(configfilepath);
	    configfile << "0 2 0 5 0 4 65530 \"/tmp/model1.pt\" \""
		       << templatefilepath << "\"\n";
	    configfile << "0 2 1 5 0 4 65530 \"/tmp/model2.pt\" \""
		       << templatefilepath << "\"\n";
	    configfile.close();

	    templatefile.open(templatefilepath);
	    templatefile << "1\n";
	    templatefile << "0\n";
	    templatefile << "2\n";
	    templatefile << "2\n";
	    templatefile << "0\n";
	    templatefile << "0\n";

	    templatefile.close();

	    setenv("FIT_CONFIGFILE", configfilepath.c_str(), 1);
	    config.readConfigFile();
	}

    void tearDown()
	{
	    if (remove(configfilepath.c_str())) {
		std::cerr << "ERROR: failed to delete temporary file at "
			  << configfilepath << std::endl;
	    }
	    
	    if (remove(templatefilepath.c_str())) {
		std::cerr << "ERROR: failed to delete temporary file at "
			  << templatefilepath << std::endl;
	    }
	    
	    if (origEnv) {
		setenv("FIT_CONFIGFILE", origEnv, 1);
	    } else {
		unsetenv("FIT_CONFIGFILE");
	    }
	}

    void checkmap()
	{	    
	    EQMSG("channel in the fit map", true,
		  config.fitChannel(0, 2, 0));
	    EQMSG("channel not in the fit map", false,
		  config.fitChannel(0, 2, 2));
	}

    void gettracelen()
	{
	    EQMSG("check trace length",
		  (unsigned)5, config.getTraceLength(0, 2, 0));
	}

    void getlimits()
	{
	    auto lim = config.getFitLimits(0, 2, 0);
	    EQMSG("check low limit", (unsigned)0, lim.first);
	    EQMSG("check high limit", (unsigned)4, lim.second);
	}
    
    void getsat()
	{
	    EQMSG("check saturation value",
		  (unsigned)65530, config.getSaturationValue(0, 2, 0));
	}

    void getmodelpath()
	{
	    std::string expected("/tmp/model1.pt");
	    std::string actual = config.getModelPath(0, 2, 0);
	    EQMSG("check model path is correct", expected, actual);
	}

    void getlist()
	{
	    std::vector<std::string> expected;
	    expected.push_back("/tmp/model1.pt");
	    expected.push_back("/tmp/model2.pt");
	    std::vector<std::string> actual = config.getModelList();
	    EQMSG("check size of model list", expected.size(), actual.size());
	    EQMSG("check model list contents", expected, actual);
	}
    
    void getmodelshape()
	{
	    std::string path = config.getModelPath(0, 2, 0);
	    unsigned shape = config.getModelShape(path);
	    EQMSG("check model shape", (unsigned)5, shape);
	}
    
    void gettemplate()
	{
	    std::vector<double> expected;
	    expected.push_back(0);
	    expected.push_back(2);
	    expected.push_back(2);
	    expected.push_back(0);
	    expected.push_back(0);
	    std::vector<double> actual = config.getTemplate(0, 2, 0);
	    EQMSG("check template contents", expected, actual);	    
	}
    
    void getalign()
	{
	    EQMSG("check template alignment point",
		  (unsigned)1, config.getTemplateAlignPoint(0, 2, 0));	    
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION(ConfigurationTests);
