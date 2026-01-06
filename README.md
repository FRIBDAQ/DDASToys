# DDASToys documentation

## Introduction
This page contains source code documentation for DDASToys. This code is used to build shared plugin libraries which can be used by the NSCLDAQ `EventEditor` program to fit recorded trace data. DDASToys is supported as part of NSCLDAQ releases 11.3 and later. The classes and functions comprising the DDASToys software package are documented with an eye toward guiding users looking to incorporate the fitting subroutines into their own code.

Two companion programs for analyzing DDAS data with fits are provided as part of the DDASToys package. Since many users perform their final analysis using CERN ROOT, a conversion tool called `eeconverter` is provided to convert DDAS data with fits to ROOT format. This is conceptually similar to the NSCLDAQ `ddasdumper` program many users are familiar with. A shared library for I/O in ROOT is provided. Trace data and their associated fits can be examined using the `traceview` program. Please note that `traceview` is a lightweight debugging and diagnostic tool, not an analysis tool; analysis of fitted trace data is left to the user.

## DDASToys Overview

DDASToys provides three FitEditor libraries as plugin extensions for the `EventEditor` program. These libraries allow fits to be parallelized using either ZMQ threading or MPI. The three libraries allow users to fit traces using either:
* `libFidEditorAnalytic.so` - An analytical fitting method which models the trace using a logistic risetime and exponential decay
* `libFidEditorTemplate.so` - A template fit method in which a pre-defined "super pulse" representing a "typical" pulse shape is fit to the data
* `libFidEditorMLInference.so` - A machine-learning inference fitting using the same model response function as the analytic fit method.

`$DAQBIN/EventEditor --help` will provide some guidance on how to run this code. Two more libraries are provided:
* `libDDASFitHitUnpacker.so` : defines an unpacker for DDAS hits with fit extensions which unpacks event fragments into the DDASFitHit class.
* `libDDASRootFitFormat.so` : contains the dictionary needed by ROOT for I/O with DDASFitHits and defines the data structure in the output ROOT file. Note that this library is called `libDDASRootFit.so` in pre-6.0 releases. 

For more information refer to the DDASToys Manual installed in <span>$</span>(PREFIX)/share/manual/manual.pdf or point a web browser at <span>$</span>(PREFIX)/share/manual/manual.html. Note that you may want to copy the entire share directory somewhere more convenient e.g., your user home directory before viewing in a browser.

## Building DDASToys
Clone the DDASToys repository using `git clone https://github.com/NSCLDAQ/DDASToys.git`. The main branch should be checked out by default. You can verify this using `git branch`. In general it is not advisable to build and install DDASToys from the main branch, you should instead pull down a tag corresponding to a released version. Most often you will want to grab the latest tagged release.

To build the DDASToys code:
* Setup the NSCLDAQ environment by sourcing the daqsetup.bash script from NSCLDAQ 12.2-000 or later. This will define the environment variables `DAQLIB`, `DAQINC`, etc.
* Ensure CMake 3.13 or later is installed. CMake 3.13+ is required to build the DDASFormat library.
* Ensure Qt 5.11.3 or later is installed. Qt is required by the `traceview` GUI.
* Configure the same CERN ROOT environment used to compile the NSCLDAQ version you are compiling the DDASToys code against. You can verify the ROOT version by examining the output of `ldd $DAQLIB/libddasrootformat.so | grep root` provided that the NSCLDAQ environment is set. Source the script thisroot.sh located under the top-level ROOT installation bin/ directory.

The DDASFormat library is incorporated as a git submodule. Before proceeding with the following installation steps, run the command `git submodule init --recursive` in the top-level DDASToys source directory. This will clone the DDASFormat repository and checkout the correct tag for the version of DDASToys you are installing.

Once the environment and submodule are correctly configured, you can build DDASToys using the command `PREFIX=/where/to/install/ddastoys make all install` from the top-level source directory. If no `PREFIX` is specified, it will default to /user/\<yourname\>/ddastoys. This will:
* Build and install the DDASFormat library libDDASFormat.so and DDAS format and unpacker headers under `$(PREFIX)/DDASFormat`,
* Build the various DDASToys fit editor, format, and unpacker libraries,
* Build the `traceview` and `eeconverter` programs (note `traceview` is only built if Qt 5.11.3+ is found), and
* Build the full DDASToys documentation.
* Install all of this stuff in the proper location.
To build with inference timing loops enabled, compile with `ENABLE_TIMING=1` defined in the environment.

The unit tests, run by typing `make check` should all pass.

## Running DDASToys Codes
For detailed information about how to run the `EventEditor` codes please refer to the DDASToys manual installed in <span>$</span>(PREFIX)/share/manual/manual.pdf.

### Fitting Traces Using the Plugin Libraries
For an explanation of how to run the `EventEditor` trace fitting framework, please refer to the DDASToys Manual or the output of the command `$DAQBIN/EventEditor --help` run from a terminal. We will assume that you have installed the latest version of the DDASToys package. In that case, the `DAQBIN` variable must point to an NSCLDAQ version 12.2-000 or later where the `EventEditor` software is installed. The manual describes how to run the fitting software at NERSC (or, with some minor edits, SLURM batch systems more generically) and configure an analysis pipeline for trace fitting. The structure of the fit extension appended to each event is defined in the fit_extensions.h header.

### Converting Event Files Containing Fits to ROOT Format
The `eeconverter` program converts `EventEditor` output into a ROOT file format suitable for further analysis. Running `eeconverter --help` from the command line will show how to use the program and how to pass it arguments; running without any command line parameters will show you the minimum number of required arguments.

The `eeconverter` program reads ring items from a data source -- in this case built NSCLDAQ event data possibly containing fit information -- and hands them off to a ring item processor. The processor performs type-independent processing of each ring item, converting each NSCLDAQ physics event to a ROOT-ized data format and writing it to a ROOT file sink.

### Viewing Traces and Fits Using Traceview
The `traceview` program can be used to display trace data and their respective fits (if present). Currently, `traceview` reads the fit and template configuration information from the file pointed to by the environment variable `FIT_CONFIGFILE`. Refer to the DDASToys Manual for more information about the format of the configuration file.

The `traceview` top menu is used to load data files and to exit the program. Successfully loading a file enables the GUI elements which allow you to parse the file and view its contents.

Crate/slot/channel identifying information for traces you wish to inspect are configured through the <em>Channel Selection</em> box. A `*` character is interpreted as a wildcard i.e. crate/slot/channel = 1/2/`*` will show traces for all channels located in crate 1, slot 2 for a given event. The <em>Event Selection</em> box allows you to skip events or select a particular event. Both features read the value in the event selection text box. The <em>Main Control</em> box buttons are used to view the next event containing trace data, update the list of viewable events based on the channel selection box values, and exit the program, respectively. Once a file has been loaded, you must hit the *Next* button to view the first physics event containing trace data or otherwise advance the file using the *Skip* or *Select* buttons.

The <em>Hit Data</em> and <em>Classifier Data</em> boxes display basic event information and classifier output. Both classification probabilities may be displayed as N/A in the case that no fit is associated with the channel or zero in the case where a fit is present without classification data. The <em>Fit Data</em> box allows you to configure the fitting method and print fit results for traces with fit data. A warning is issued if the program believes the wrong fitting method has been selected.

Once a physics event containing trace data has been found, a list of channels with traces matching the current channel selection box data is populated. Clicking on one of the list members will draw the trace and its fits (if present) on the embedded ROOT canvas. The ROOT canvas can be interacted with in the same way as a normal ROOT TCanvas. 

Some `traceview` options -- the loaded data file and the fitting method -- can be configured from the command line as well as from the GUI. To see a list of command-line parameters, run `traceview --help` or `traceview -h`.

## Appendices

### Appendix A: Release Notes

* 4.0-001 : Version used at NERSC during Feb., 2024 FDSi experiment e21062. Frozen and not maintained. This version is only compatible with NSCLDAQ 12.0 and earlier.
* 5.0-002 : Major version 5 (and newer) incorporate an external library to unpack raw DDAS data. The DDASFitHit and DDASRootFitHit classes inherit from DAQ::DDAS::DDASHit, and write their own extension data to the output ROOT file. Last tag compatible with NSCLDAQ 12.0 and earlier.
* 5.1-000 : The final tag prior to incorporating the machine learning inference. This version of DDASToys also requires the user to point at the location of the [UnifiedFormat](https://github.com/FRIBDAQ/UnifiedFormat) library external to this project e.g., from the version of NSCLDAQ that you build against. The variable `UFMT` should point to the top-level installation directory of the unified format version you'd like to use. This tag requires an NSCLDAQ installation built against UnifiedFormat version 2.1 or later and is only compatible with NSCLDAQ releases 12.1 and later.
* 6.0-000 : The first version with the machine learning inference model. Compatible with NSCLDAQ versions 12.1 and later. Besides the machine learning, a number of changes have been implemented in this version:
  - libDDASRootFit.so renamed to libDDASRootFitFormat.so
  - Consistent and clear namespacing. The ROOT format library libDDASRootFitFormat.so is in the `ddastoys` namespace, all fitting functions and plugin-specific code have their own namespaces: `ddastoys::analyticfit`, `ddastoys::templatefit`, `ddastoys::mlinference`.
  - An additional entry in the fit configuration file is required. This entry is used to specify the path to a machine learning inference model for determining pulse parameters for that channel. In the case where you do not want to use the machine learning inference, this input parameter does nothing. Some placeholder must be present in the fit configuration file, which can be an empty string "".
* 6.1-000 : The full hit timestamp is displayed in the `traceview` <em>Hit data</em> box with 1 ps precision (3 decimal places of the full timestamp in nanoseconds). A new feature has been added which allows the user to select PHYSICS_EVENTs by their event number. Note that the selected events may or may not contain trace data and therefore the hit selection list in `traceview` may be empty.
* 6.2-000 : Users can specify an event list to view a subset of traces in the input data file.
* 6.2-001 : Incorporate changes needed to use new ML models which allow trace positions to vary freely on the time axis.
* 6.3-000 : Updated for ML model used in e23055 (Crider).
* 6.3-001 : Optimizations for ML inference, added some simple inference profiling tools and option to build DDASToys with profiling output.
* 6.4-000 : User provides trace length in fit configuration file. Remove dependence on template file; support per-channel trace templates. Allow "none" as model or template path.