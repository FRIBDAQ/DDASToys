# Introduction
This page contains source code documentation for DDASToys. This code is used to build shared plugin libraries which can then be used by the FRIBDAQ `EventEditor` program to fit recorded trace data and is supported as part of FRIBDAQ releases 12.0 and later. The implemented classes and functions in DDASToys are documented with an eye toward guiding users looking to incorporate the fitting subroutines into their own code. Additionally, the `traceview` program provided as part of the DDASToys package can be used to view channel traces and their associated fit data with a Qt-based GUI running an embedded ROOT canvas.

# Building DDASToys
Clone the repository from https://github.com/FRIBDAQ/DDASToys.git and checkout the main branch. The DDASToys code to build and install is located under main/mpifit.

To build this code you must:

- Setup the FRIBDAQ environment by sourcing the daqsetup.bash script from FRIBDAQ 12.0 or later. This will define the environment variables `DAQLIB`, `DAQINC`, etc.
- Define the environment variable `SPECTCLHOME` to point to the installation directory of SpecTcl 5.13-003 or later.
- Define `XIAAPIDIR` to point to the top of the XIA API version install tree generally found in under /usr/opt/xiaapi/<version_number>.
- Ensure Qt 5.11.3 or later is installed (required by the `traceview` GUI).
- Configure the same CERN ROOT environment as used to compile the FRIBDAQ version you are compiling the DDASToys code against. You can verify the ROOT version by examining the output of `ldd $DAQLIB/libddaschannel.so | grep root` provided that the FRIBDAQ environment is set.

Once the environment setup is correct, navigate to the main/mpifit directory and build DDASToys using `make`. This will create the `libFitEditorAnalytic.so`, `libFitEditorTemplate.so` and `libDDASFitHitUnpacker.so` libraries, `traceview` program and the full DDASToys documentation. The `libFitEditorXXX.so` libraries are plugin extensions for the `EventEditor` program allowing fits to be parallelized using either ZMQ threading or MPI. `$DAQBIN/EventEditor --help` will provide some guidance on how to run this code. Once built, the headers, libraries, and documentation can be installed in a central location using the command `make install PREFIX=/path/to/install/dir`. For more information refer to the DDASToys Manual installed in $PREFIX/share/manual/manual.pdf or point a web browser at $PREFIX/share/manual/manual.html.

# Running DDASToys code
An overview of the `traceview` program and its graphical interface is presented [here]{#viewing-traces-and-fits-using-traceview}. For more information about how to run the `EventEditor` codes refer to the DDASToys manual.

## Fitting traces using the plugin libraries
For an explanation of how to run the `EventEditor` trace fitting framework, please refer to the DDASToys Manual or the output of the command `$DAQBIN/EventEditor --help` run from a terminal. The `DAQBIN` variable must point to an FRIBDAQ version 12.0 or later where the `EventEditor` software is installed. The manual also describes how to run the fitting software at NERSC and configure an analysis pipeline for trace fitting. Implementation of the fitting routines and their source code documentation is provided here. Notably, the structure of the fit extensions appended to each event is defined in the fit_extensions.h header.

## Viewing traces and fits using traceview
The `traceview` program can be used to display trace data and their respective fits (if present). Currently, `traceview` reads the fit and template configuration information from the files pointed to by the environment variables `FIT_CONFIGFILE` and `TEMPLATE_CONFIGFILE`. Refer to the DDASToys Manual for more information about the format of these configuration files.

The `traceview` top menu is used to load data files and to exit the program. Successfully loading a file enables the GUI elements which allow you to parse the file and view its contents.

Crate/slot/channel identifying information for traces you wish to inspect is configured through the <em>Channel selection box</em>. A `*` character is interpreted as a wildcard i.e. crate/slot/channel = 1/2/`*` will show traces for all channels located in crate 1, slot 2 for a given event. The <em>Skip control box</em> allows you to skip ahead in the data file by the number of events shown in the text box. The <em>Main control box</em> buttons are used to view the next event, update the list of viewable events based on the channel selection box values and exit the program, respectively. Once a file has been loaded, you must hit the \a Next button to view the first physics event.

The <em>Hit data</em> and <em>Classifier data</em> boxes display basic event information and classifier output provided a classifier is used. The <em>Fit data</em> box allows you to configure the fitting method and print fit results for traces with fit data. A warning is issued if the program believes the wrong fitting method has been selected.

Once a physics event containing trace data has been found, a list of channels with traces matching the current channel selection box data is populated. Clicking on one of the list members will draw the trace and its fits (if present) on the embedded ROOT canvas. The ROOT canvas can be interacted with in the normal ROOT fashion (i.e. clicking and dragging along an axis will zoom). 

Some `traceview` options--the loaded data file and the fitting method--can be configured from the command line as well as from the GUI. To see a list of commands, run `traceview --help` or `traceview -h`.