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
template <class T> ostream &operator<<(ostream &stream, const vector<T> &vec) {
  stream << "{ ";
  for (auto &element : vec)
    stream << element << " ";
  stream << "}";

  return stream;
}

template <class T, long unsigned int N>
ostream &operator<<(ostream &stream, const array<T, N> &vec) {
  stream << "{ ";
  for (int i = 0; i < N; ++i)
    stream << vec[i] << " ";
  stream << "}";

  return stream;
}
} // namespace std

class ConfigurationTests : public CppUnit::TestFixture {
public:
  char *origEnv;
  Configuration config;
  std::string config_file_path;
  std::ofstream config_file;
  std::string template_file_path;
  std::ofstream template_file;

public:
  CPPUNIT_TEST_SUITE(ConfigurationTests);

  CPPUNIT_TEST(check_map);       // check the channel map for channels to fit
  CPPUNIT_TEST(get_trace_len);   // check trace length
  CPPUNIT_TEST(get_limits);      // check fit limits
  CPPUNIT_TEST(get_sat);         // check saturation value
  CPPUNIT_TEST(get_model_path);  // check model path
  CPPUNIT_TEST(get_list);        // check list of models
  CPPUNIT_TEST(get_model_shape); // check model shape
  CPPUNIT_TEST(get_template);    // check template data
  CPPUNIT_TEST(get_align);       // check template aligment point
  CPPUNIT_TEST(check_bad_low_limit_1);  // check bad low limit > high limit
  CPPUNIT_TEST(check_bad_low_limit_2);  // check bad low limit == high limit
  CPPUNIT_TEST(check_bad_high_limit_1); // check bad high limit > trace length
  CPPUNIT_TEST(check_bad_high_limit_2); // check bad high limit == trace length

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {
    origEnv = getenv("FIT_CONFIGFILE");

    config_file_path = "/tmp/fitconfig.txt";
    template_file_path = "/tmp/template.txt";

    config_file.open(config_file_path);
    config_file << "0 2 0 5 0 4 65530 \"/tmp/model1.pt\" \""
                << template_file_path << "\"\n";
    config_file << "0 2 1 5 0 4 65530 \"/tmp/model2.pt\" \""
                << template_file_path << "\"\n";
    config_file << "0 3 0 10 0 9 16380 \"none\" \"none\"\n";
    config_file << "0 3 1 10 0 9 16380 \"\" \"\"\n";
    config_file << "1 4 0 20 0 19 32760 \"/tmp/model3.pt\" \"none\"\n";
    config_file.close();

    template_file.open(template_file_path);
    template_file << "1\n";
    template_file << "0\n";
    template_file << "2\n";
    template_file << "2\n";
    template_file << "0\n";
    template_file << "0\n";

    template_file.close();

    setenv("FIT_CONFIGFILE", config_file_path.c_str(), 1);
    config.readConfigFile();
  }

  void tearDown() {
    if (remove(config_file_path.c_str())) {
      std::cerr << "ERROR: failed to delete temporary file at "
                << config_file_path << std::endl;
    }

    if (remove(template_file_path.c_str())) {
      std::cerr << "ERROR: failed to delete temporary file at "
                << template_file_path << std::endl;
    }

    if (origEnv) {
      setenv("FIT_CONFIGFILE", origEnv, 1);
    } else {
      unsetenv("FIT_CONFIGFILE");
    }
  }

  void check_map() {
    EQMSG("channel in the fit map", true, config.fitChannel(0, 2, 0));
    EQMSG("channel not in the fit map", false, config.fitChannel(0, 2, 2));
  }

  void get_trace_len() {
    EQMSG("check trace length", (unsigned)5, config.getTraceLength(0, 2, 0));
  }

  void get_limits() {
    auto lim = config.getFitLimits(0, 2, 0);
    EQMSG("check low limit", (unsigned)0, lim.first);
    EQMSG("check high limit", (unsigned)4, lim.second);
  }

  void get_sat() {
    EQMSG("check saturation value", (unsigned)65530,
          config.getSaturationValue(0, 2, 0));
  }

  void get_model_path() {
    std::string expected("/tmp/model1.pt");
    std::string actual = config.getModelPath(0, 2, 0);
    EQMSG("check model path is correct", expected, actual);
  }

  void get_list() {
    std::vector<std::string> expected;
    expected.push_back("/tmp/model1.pt");
    expected.push_back("/tmp/model2.pt");
    expected.push_back("/tmp/model3.pt");
    std::vector<std::string> actual = config.getModelList();
    EQMSG("check size of model list", expected.size(), actual.size());
    EQMSG("check model list contents", expected, actual);
  }

  void get_model_shape() {
    std::string path = config.getModelPath(0, 2, 0);
    unsigned shape = config.getModelShape(path);
    EQMSG("check model shape", (unsigned)5, shape);
  }

  void get_template() {
    std::vector<double> expected;
    expected.push_back(0);
    expected.push_back(2);
    expected.push_back(2);
    expected.push_back(0);
    expected.push_back(0);
    std::vector<double> actual = config.getTemplate(0, 2, 0);
    EQMSG("check template contents", expected, actual);
  }

  void get_align() {
    EQMSG("check template alignment point", (unsigned)1,
          config.getTemplateAlignPoint(0, 2, 0));
  }

  // Low limit > high limit in configuration file should throw an exception
  void check_bad_low_limit_1() {
    std::string bad_config_file_path = "/tmp/bad_fitconfig.txt";
    std::ofstream bad_config_file;
    bad_config_file.open(bad_config_file_path);
    bad_config_file << "0 2 0 5 6 4 65530 \"/tmp/model1.pt\" \""
                    << template_file_path << "\"\n";
    bad_config_file.close();

    setenv("FIT_CONFIGFILE", bad_config_file_path.c_str(), 1);
    CPPUNIT_ASSERT_THROW(config.readConfigFile(), std::invalid_argument);

    if (remove(bad_config_file_path.c_str())) {
      std::cerr << "ERROR: failed to delete temporary file at "
                << bad_config_file_path << std::endl;
    }
  }

  // Low limit == high limit in configuration file should throw an exception
  void check_bad_low_limit_2() {
    std::string bad_config_file_path = "/tmp/bad_fitconfig.txt";
    std::ofstream bad_config_file;
    bad_config_file.open(bad_config_file_path);
    bad_config_file << "0 2 0 5 4 4 65530 \"/tmp/model1.pt\" \""
                    << template_file_path << "\"\n";
    bad_config_file.close();

    setenv("FIT_CONFIGFILE", bad_config_file_path.c_str(), 1);
    CPPUNIT_ASSERT_THROW(config.readConfigFile(), std::invalid_argument);

    if (remove(bad_config_file_path.c_str())) {
      std::cerr << "ERROR: failed to delete temporary file at "
                << bad_config_file_path << std::endl;
    }
  }

  // high limit > trace length in configuration file should throw an exception
  void check_bad_high_limit_1() {
    std::string bad_config_file_path = "/tmp/bad_fitconfig.txt";
    std::ofstream bad_config_file;
    bad_config_file.open(bad_config_file_path);
    bad_config_file << "0 2 0 5 0 6 65530 \"/tmp/model1.pt\" \""
                    << template_file_path << "\"\n";
    bad_config_file.close();

    setenv("FIT_CONFIGFILE", bad_config_file_path.c_str(), 1);
    CPPUNIT_ASSERT_THROW(config.readConfigFile(), std::invalid_argument);

    if (remove(bad_config_file_path.c_str())) {
      std::cerr << "ERROR: failed to delete temporary file at "
                << bad_config_file_path << std::endl;
    }
  }

  // high limit == trace length in configuration file should throw an exception
  void check_bad_high_limit_2() {
    std::string bad_config_file_path = "/tmp/bad_fitconfig.txt";
    std::ofstream bad_config_file;
    bad_config_file.open(bad_config_file_path);
    bad_config_file << "0 2 0 5 0 5 65530 \"/tmp/model1.pt\" \""
                    << template_file_path << "\"\n";
    bad_config_file.close();

    setenv("FIT_CONFIGFILE", bad_config_file_path.c_str(), 1);
    CPPUNIT_ASSERT_THROW(config.readConfigFile(), std::invalid_argument);

    if (remove(bad_config_file_path.c_str())) {
      std::cerr << "ERROR: failed to delete temporary file at "
                << bad_config_file_path << std::endl;
    }
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(ConfigurationTests);
