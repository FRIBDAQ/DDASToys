# DDASToys documentation

# Introduction
This page contains source code documentation for DDASToys. This code is used to build shared plugin libraries which can be used by the NSCLDAQ `EventEditor` program to fit recorded trace data. DDASToys is supported as part of NSCLDAQ releases 11.3 and later. The implemented classes and functions in DDASToys are documented with an eye toward guiding users looking to incorporate the fitting subroutines into their own code.

Two companion programs for analyzing DDAS data with fits are provided as part of the DDASToys package. Since many users perform their final analysis using CERN ROOT, a conversion tool called `eeconverter` is provided to convert DDAS data with fits to ROOT format. This is conceptually similar to the NSCLDAQ `ddasdumper` program many users are familiar with. A shared library for I/O in ROOT is provided. Trace data and their associated fits can be examined using the `traceview` program. Please note that `traceview` is a lightweight debugging and diagnostic tool, not an analysis tool; analysis of fitted trace data is left to the user.

# Building DDASToys
Clone the DDASToys repository using `git clone https://github.com/NSCLDAQ/DDASToys.git`. The main branch should be checked out by default. You can verify this using `git branch`. In general it is not advisable to build and install DDASToys from the main branch, you should instead pull down a tag corresponding to a released version. Some tags of note:
* 4.0-001 : Version used at NERSC during Feb., 2024 FDSi experiment e21062. Frozen and not maintained.
* 5.0-002 : Major version 5 (and newer) incorporate an external library to unpack raw DDAS data. The DDASFitHit and DDASRootFitHit classes inherit from DAQ::DDAS::DDASHit, and write their own extension data to the output ROOT file. Last tag compatible with NSCLDAQ 12.0 and earlier.
* 5.1-000 : The final tag prior to incorporating the machine learning inference. This version of DDASToys also requires the user to point at the location of the [UnifiedFormat](https://github.com/FRIBDAQ/UnifiedFormat) library external to this project e.g., from the version of NSCLDAQ that you build against. The variable `UFMT` should point to the top-level installation directory of the unified format version you'd like to use. This tag requires an NSCLDAQ installation built against UnifiedFormat version 2.1 or later and is only compatible with NSCLDAQ releases 12.1 and later.
* 6.0-000 : The first version with the machine learning inference model. Compatible with NSCLDAQ versions 12.1 and later. Besides the machine learning, a number of changes have been implemented in this version:
  - libDDASRootFit.so renamed to libDDASRootFitFormat.so
  - Consistent and clear namespacing. The ROOT format library libDDASRootFitFormat.so is in the `ddastoys` namespace, all fitting functions and plugin-specific code have their own namespaces: `ddastoys::analyticfit`, `ddastoys::templatefit`, `ddastoys::mlinference`.
  - An additional entry in the fit configuration file is required. This entry is used to specify the path to a machine learning inference model for determining pulse parameters for that channel. In the case where you do not want to use the machine learning inference, this input parameter does nothing. Some placeholder must be present in the fit configuration file, which can be an empty string "".
* 6.1-000 : The full hit timestamp is displayed in the `traceview` <em>Hit data</em> box with 1 ps precision (3 decimal places of the full timestamp in nanoseconds). A new feature has been added which allows the user to select PHYSCIS_EVENTs by their event number. Note that the selected events may or may not contain trace data and therefore the hit selection list in `traceview` may be empty.
* 6.2-000 : Users can specify an event list to view a subset of traces in the input data file.
* 6.2-001 : Incorporate changes needed to use new ML models which allow trace positions to vary freely on the time axis.

## Build Instructions for DDASToys 5.1 and later
- Setup the NSCLDAQ environment by sourcing the daqsetup.bash script from NSCLDAQ 12.1-000 or later. This will define the environment variables `DAQLIB`, `DAQINC`, etc.
- Ensure CMake 3.13 or later is installed. CMake 3.13+ is required to build the DDASFormat library.
- Ensure Qt 5.11.3 or later is installed. Qt is required by the `traceview` GUI.
- Configure the same CERN ROOT environment used to compile the NSCLDAQ version you are compiling the DDASToys code against. You can verify the ROOT version by examining the output of `ldd $DAQLIB/libddasrootformat.so | grep root` provided that the NSCLDAQ environment is set. Source the script /bin/thisroot.sh located under the top-level ROOT installation directory. In the FRIB bullseye container it is most likely /usr/opt/root/6.26.04/bin/thisroot.sh.

The DDAS format library is incorporated as a git submodule. Before proceeding with the following installation steps, run the command `git submodule init --recursive` in the top-level DDAS toys source directory. This will clone the DDAS format repository and checkout the correct tag for the version of DDASToys you are installing.

Once the environment and submodule are correctly configured, you can build DDASToys using the command `UFMT=/path/to/ufmt PREFIX=/where/to/install/ddastoys make` from the top-level source directory. If no `PREFIX` is specified, it will default to /user/\<yourname\>/ddastoys. This will:
* Build and install the DDASFormat library libDDASFormat.so and DDAS format and unpacker headers at `$(PREFIX)/DDASFormat`,
* Build the various fit editor, format, and unpacker libraries,
* Build the `traceview` and `eeconverter` programs (note `traceview` is only built if Qt 5.11.3+ is found), and
* Build the full DDASToys documentation.
Type `UFMT=/path/to/ufmt PREFIX=/where/to/install/ddastoys make install` to install the DDASToys software and documentation somewhere. Note that this install `PREFIX` also defaults to /user/\<yourname\>/ddastoys if not specified. The `PREFIX` for the format library and the DDASToys code can in principle be different but they are intended to be installed under the same directory tree. You can also build and install the project using a single command: `UFMT=/path/to/ufmt PREFIX=/where/to/install/ddastoys make all install`.

## Build Instructions for DDASToys 5.0
- Setup the NSCLDAQ environment by sourcing the daqsetup.bash script from NSCLDAQ 12.0-005 or later. This will define the environment variables `DAQLIB`, `DAQINC`, etc.
- Ensure CMake 3.13 or later is installed. CMake 3.13+ is required to build the DDASFormat library.
- Ensure Qt 5.11.3 or later is installed. Qt is required by the `traceview` GUI.
- Configure the same CERN ROOT environment used to compile the NSCLDAQ version you are compiling the DDASToys code against. You can verify the ROOT version by examining the output of `ldd $DAQLIB/libddasrootformat.so | grep root` provided that the NSCLDAQ environment is set. Source the script /bin/thisroot.sh located under the top-level ROOT installation directory. In the FRIB buster container this is most likely /usr/opt/root/root-6.24.06/bin/thisroot.sh; in the FRIB bullseye container it is most likely /usr/opt/root/6.26.04/bin/thisroot.sh.

Once the environment is correctly configured, navigate into the cloned repository directory and build DDASToys using `PREFIX=/where/to/install/ddastoys make`. If no `PREFIX` is specified, it will default to /user/\<yourname\>/ddastoys. This will:
* Build and install the DDASFormat library libDDASFormat.so and DDAS format and unpacker headers at `$(PREFIX)/DDASFormat`,
* Build the `libFitEditorAnalytic.so`, `libFitEditorTemplate.so`, `libDDASFitHitUnpacker.so` and `libDDASRootFit.so` libraries,
* Build the `traceview` and `eeconverter` programs (note `traceview` is only built if Qt 5.11.3+ is found),
* The full DDASToys documentation.
Type `PREFIX=/where/to/install/ddastoys make install` to install the DDASToys software and documentation somewhere. Note that this install `PREFIX` also defaults to /user/\<yourname\>/ddastoys if not specified. The `PREFIX` for the format library and the DDASToys code can in principle be different but they are intended to be installed under the same directory tree. You can also build and install the project using a single command: `PREFIX=/where/to/install/ddastoys make all install`.

## Build Instructions for DDASToys 4.0 and Earlier
- This tag is most likely only usable with NSCLDAQ 12.0 due to changes to the DDAS format library first implemented in 12.1. Setup the NSCLDAQ environment by sourcing the daqsetup.bash script from NSCLDAQ 12.0-005 or later.
- Ensure Qt 5.11.3 or later is installed.
- Configure the same CERN ROOT environment used to compile the NSCLDAQ version you are compiling the DDASToys code against.

Once the environment is correctly configured, navigate into the cloned repository directory and build DDASToys using `make`. This will create the following:
* The `libFitEditorAnalytic.so`, `libFitEditorTemplate.so`, `libDDASFitHitUnpacker.so` and `libDDASRootFitFormat.so` libraries,
* The `traceview` and `eeconverter` programs, and,
* The full DDASToys documentation.
Type `PREFIX=/where/to/install/ddastoys make install` to install the DDASToys software and documentation at a directory of your choosing.

# DDASToys Overview

The FitEditor shared libraries are plugin extensions for the `EventEditor` program allowing fits to be parallelized using either ZMQ threading or MPI. `$DAQBIN/EventEditor --help` will provide some guidance on how to run this code. The `libDDASFitHitUnpacker.so` library defines an unpacker for DDAS hits with fit extensions. Event fragments are unpacked into the DDASFitHit class. The `libDDASRootFitFormat.so` library (`libDDASRootFit.so` in pre-6.0 releases) contains the dictionary needed by ROOT for I/O of custom classes and defines the data structure in the output ROOT file.

For more information refer to the DDASToys Manual installed in <span>$</span>(PREFIX)/share/manual/manual.pdf or point a web browser at <span>$</span>(PREFIX)/share/manual/manual.html.

# Running DDASToys Codes
For more information about how to run the `EventEditor` codes please refer to the DDASToys manual.

## Fitting Traces Using the Plugin Libraries
For an explanation of how to run the `EventEditor` trace fitting framework, please refer to the DDASToys Manual or the output of the command `$DAQBIN/EventEditor --help` run from a terminal. We will assume that you have installed the latest version of the DDASToys package. In that case, the `DAQBIN` variable must point to an NSCLDAQ version 12.1 or later where the `EventEditor` software is installed. The manual also describes how to run the fitting software at NERSC and configure an analysis pipeline for trace fitting. Notably, the structure of the fit extension appended to each event is defined in the fit_extensions.h header.

## Converting Event Files Containing Fits to ROOT Format
The `eeconverter` program converts `EventEditor` output into a ROOT file format suitable for further analysis. Running `eeconverter --help` from the command line will show how to use the program and how to pass it arguments; running without any command line parameters will show you the minimum number of required arguments.

The `eeconverter` program reads ring items from a data source -- in this case built NSCLDAQ event data possibly containing fit information -- and hands them off to a ring item processor. The processor performs type-independent processing of each ring item, converting each NSCLDAQ PHYSICS_EVENT to a ROOT-ized data format and writing it to a ROOT file sink.

## Viewing Traces and Fits Using Traceview
The `traceview` program can be used to display trace data and their respective fits (if present). Currently, `traceview` reads the fit and template configuration information from the files pointed to by the environment variables `FIT_CONFIGFILE` and `TEMPLATE_CONFIGFILE`. For analytic or machine-learning inference fitting, only the former must be defined; to correctly display template fitting results, both must be defined and point to valid files. Refer to the DDASToys Manual for more information about the format of these configuration files.

The `traceview` top menu is used to load data files and to exit the program. Successfully loading a file enables the GUI elements which allow you to parse the file and view its contents.

Crate/slot/channel identifying information for traces you wish to inspect are configured through the <em>Channel Selection</em> box. A `*` character is interpreted as a wildcard i.e. crate/slot/channel = 1/2/`*` will show traces for all channels located in crate 1, slot 2 for a given event. The <em>Event Selection</em> box allows you to skip events or select a particular event. Both features read the value in the event selection text box. The <em>Main Control</em> box buttons are used to view the next event containing trace data, update the list of viewable events based on the channel selection box values, and exit the program, respectively. Once a file has been loaded, you must hit the \a Next button to view the first physics event containing trace data or otherwise advance the file using the \a Skip or \a Select buttons.

The <em>Hit Data</em> and <em>Classifier Data</em> boxes display basic event information and classifier output. Both classification probabilities may be displayed as N/A in the case that no fit is associated with the channel or zero in the case where a fit is present without classification data. The <em>Fit Data</em> box allows you to configure the fitting method and print fit results for traces with fit data. A warning is issued if the program believes the wrong fitting method has been selected.

Once a physics event containing trace data has been found, a list of channels with traces matching the current channel selection box data is populated. Clicking on one of the list members will draw the trace and its fits (if present) on the embedded ROOT canvas. The ROOT canvas can be interacted with in the normal way (i.e. clicking and dragging along an axis will zoom). 

Some `traceview` options--the loaded data file and the fitting method--can be configured from the command line as well as from the GUI. To see a list of commands, run `traceview --help` or `traceview -h`.
