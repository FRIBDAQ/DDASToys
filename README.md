# DDASToys documentation

# Introduction
This page contains source code documentation for DDASToys. This code is used to build shared plugin libraries which can then be used by the FRIBDAQ `EventEditor` program to fit recorded trace data. DDASToys is supported as part of FRIBDAQ releases 12.0 and later. The implemented classes and functions in DDASToys are documented with an eye toward guiding users looking to incorporate the fitting subroutines into their own code.

Two companion programs for analyzing DDAS data with fits are provided as part of the DDASToys package. Trace data and their associated fits can be examined using the `traceview` program. Similar to the `ddasdumper`, the DDASToys `eeconverter` program converts FRIBDAQ event file data containing traces and fit information into ROOT format for downstream analysis. A shared library for I/O in ROOT is provided.

# Building DDASToys
Clone the DDASToys repository using `git clone https://github.com/FRIBDAQ/DDASToys.git`. The main branch should be checked out by default. You can verify this using `git branch`. In general it is not advisable to build and install DDASToys from the main branch, you should instead pull down a tagged version of the repository. Some tags of note:
* 4.0-001 : Version used at NERSC during Feb., 2024 FDSi experiment e21062. Frozen and not maintained.
* 5.0-002 : Major version 5 (and newer) incorporate an external library to unpack raw DDAS data. The DDASFitHit and DDASRootFitHit classes inherit from DDASHit, and write their own extension data to the output ROOT file.

## Build Instructions for DDASToys 5.0-000 and Later
- Setup the FRIBDAQ environment by sourcing the daqsetup.bash script from FRIBDAQ 12.0-008 or later. This will define the environment variables `DAQLIB`, `DAQINC`, etc.
- Ensure CMake 3.13 or later is installed. CMake 3.13+ is required to build the DDASFormat library.
- Ensure Qt 5.11.3 or later is installed. Qt is required by the `traceview` GUI.
- Configure the same CERN ROOT environment used to compile the FRIBDAQ version you are compiling the DDASToys code against. You can verify the ROOT version by examining the output of `ldd $DAQLIB/libddasrootformat.so | grep root` provided that the FRIBDAQ environment is set. Source the script /bin/thisroot.sh located under the top-level ROOT installation directory. In the FRIB buster container this is most likely /usr/opt/root/root-6.24.06/bin/thisroot.sh; in the FRIB bullseye container it is most likely /usr/opt/root/6.26.04/bin/thisroot.sh.

Once the environment is correctly configured, navigate into the cloned repository directory and build DDASToys using `make PREFIX=/where/to/install/ddastoys`. If no `PREFIX` is specified, it will default to /user/\<yourname\>/ddastoys. This will:
* Build and install the DDASFormat library libDDASFormat.so and DDAS format and unpacker headers at `$(PREFIX)/DDASFormat`,
* Build the `libFitEditorAnalytic.so`, `libFitEditorTemplate.so`, `libDDASFitHitUnpacker.so` and `libDDASRootFit.so` libraries,
* Build the `traceview` and `eeconverter` programs (note `traceview` is only built if Qt 5.11.3+ is found),
* The full DDASToys documentation.
Type `make install PREFIX=/where/to/install/ddastoys` to install the DDASToys software and documentation somewhere. Note that this install `PREFIX` also defaults to /user/\<yourname\>/ddastoys if not specified. The `PREFIX` for the format library and the DDASToys code can in principle be different but they are intended to be installed under the same directory tree.

## Build Instructions for DDASToys 4.0-001 and Earlier
- This tag is most likely only usable with FRIBDAQ 12.0 due to changes to the DDAS format library first implemented in 12.1. Setup the FRIBDAQ environment by sourcing the daqsetup.bash script from FRIBDAQ 12.0-005 or later.
- Ensure Qt 5.11.3 or later is installed.
- Configure the same CERN ROOT environment used to compile the FRIBDAQ version you are compiling the DDASToys code against.

Once the environment is correctly configured, navigate into the cloned repository directory and build DDASToys using `make`. This will create the following:
* The `libFitEditorAnalytic.so`, `libFitEditorTemplate.so`, `libDDASFitHitUnpacker.so` and `libDDASRootFit.so` libraries,
* The `traceview` and `eeconverter` programs
* The full DDASToys documentation.
Type `make install PREFIX=/where/to/install/ddastoys` to install the DDASToys software and documentation at a directory of your choosing.

# DDASToys Overview

The `libFitEditorX.so` libraries are plugin extensions for the `EventEditor` program allowing fits to be parallelized using either ZMQ threading or MPI. `$DAQBIN/EventEditor --help` will provide some guidance on how to run this code. The `libDDASFitHitUnpacker.so` library defines an unpacker for DDAS hits with fit extensions. Event fragments are unpacked into the DDASFitHit class. The `libDDASRootFit.so` library contains the dictionary needed by ROOT for I/O of custom classes and sources defining the data structure in the output ROOT file.

For more information refer to the DDASToys Manual installed in <span>$</span>(PREFIX)/share/manual/manual.pdf or point a web browser at <span>$</span>(PREFIX)/share/manual/manual.html.

# Running DDASToys Codes
For more information about how to run the `EventEditor` codes please refer to the DDASToys manual.

## Fitting Traces Using the Plugin Libraries
For an explanation of how to run the `EventEditor` trace fitting framework, please refer to the DDASToys Manual or the output of the command `$DAQBIN/EventEditor --help` run from a terminal. The `DAQBIN` variable must point to an FRIBDAQ version 12.0 or later where the `EventEditor` software is installed. The manual also describes how to run the fitting software at NERSC and configure an analysis pipeline for trace fitting. Implementation of the fitting routines and their source code documentation is provided here. Notably, the structure of the fit extensions appended to each event is defined in the fit_extensions.h header.

## Converting Event Files Containing Fits to ROOT Format
The `eeconverter` program converts `EventEditor` output into a ROOT file format suitable for further analysis. Running `eeconverter --help` from the command line will show how to use the program and how to pass it arguments; running without any command line parameters will show you the minimum number of required arguments.

The `eeconverter` program reads ring items from a data source -- in this case built NSCLDAQ events possibly containing fit information -- and hands them off to a ring item processor. The processor performs type-independent processing of each ring item, converting each FRIBDAQ PHYSICS_EVENT to a ROOT-ized data format and writing it to a ROOT file sink.

## Viewing Traces and Fits Using Traceview
The `traceview` program can be used to display trace data and their respective fits (if present). Currently, `traceview` reads the fit and template configuration information from the files pointed to by the environment variables `FIT_CONFIGFILE` and `TEMPLATE_CONFIGFILE`. Refer to the DDASToys Manual for more information about the format of these configuration files.

The `traceview` top menu is used to load data files and to exit the program. Successfully loading a file enables the GUI elements which allow you to parse the file and view its contents.

Crate/slot/channel identifying information for traces you wish to inspect is configured through the <em>Channel selection box</em>. A `*` character is interpreted as a wildcard i.e. crate/slot/channel = 1/2/`*` will show traces for all channels located in crate 1, slot 2 for a given event. The <em>Skip control box</em> allows you to skip ahead in the data file by the number of events shown in the text box. The <em>Main control box</em> buttons are used to view the next event, update the list of viewable events based on the channel selection box values and exit the program, respectively. Once a file has been loaded, you must hit the \a Next button to view the first physics event.

The <em>Hit data</em> and <em>Classifier data</em> boxes display basic event information and classifier output provided a classifier is used. The <em>Fit data</em> box allows you to configure the fitting method and print fit results for traces with fit data. A warning is issued if the program believes the wrong fitting method has been selected.

Once a physics event containing trace data has been found, a list of channels with traces matching the current channel selection box data is populated. Clicking on one of the list members will draw the trace and its fits (if present) on the embedded ROOT canvas. The ROOT canvas can be interacted with in the normal ROOT fashion (i.e. clicking and dragging along an axis will zoom). 

Some `traceview` options--the loaded data file and the fitting method--can be configured from the command line as well as from the GUI. To see a list of commands, run `traceview --help` or `traceview -h`.