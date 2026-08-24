# DDASToys documentation

## Introduction
This page contains source code documentation for DDASToys. This code is used to build shared plugin libraries which can be used by the NSCLDAQ `EventEditor` program to fit recorded trace data. The DDASToys software packages require NSCLDAQ, and are supported for releases 11.3 and later. The classes and functions comprising the DDASToys software package are documented with an eye toward guiding users looking to incorporate the fitting subroutines into their own code.

Two companion programs for analyzing DDAS data with fits are provided as part of the DDASToys package. Since many users perform their final analysis using CERN ROOT, a conversion tool called `eeconverter` is provided to convert DDAS data with fits to ROOT format. This is conceptually similar to the NSCLDAQ `ddasdumper` program many users are familiar with. A shared library for I/O in ROOT is provided. Trace data and their associated fits can be examined using the `traceview` program. Please note that `traceview` is a lightweight debugging and diagnostic tool, not an analysis tool; analysis of fitted trace data is left to the user.

## DDASToys Overview

DDASToys provides three FitEditor libraries as plugin extensions for the `EventEditor` program. These libraries allow fits to be parallelized using either ZMQ threading or MPI. The three libraries allow users to fit traces using either:
* `libFidEditorAnalytic.so` - An analytical fitting method which models the trace using a logistic risetime and exponential decay
* `libFidEditorTemplate.so` - A template fit method in which a pre-defined "super pulse" representing a "typical" pulse shape is fit to the data
* `libFidEditorMLInference.so` - A machine-learning inference fitting using the same model response function as the analytic fit method (requires LibTorch).

`$DAQBIN/EventEditor --help` will provide some guidance on how to run this code. Two more libraries are provided:
* `libDDASFitHitUnpacker.so` : defines an unpacker for DDAS hits with fit extensions which unpacks event fragments into the DDASFitHit class.
* `libDDASRootFitFormat.so` : contains the dictionary needed by ROOT for I/O with DDASFitHits and defines the data structure in the output ROOT file. Note that this library is called `libDDASRootFit.so` in pre-6.0 releases. 

For more information refer to the DDASToys Manual installed in <span>$</span>(PREFIX)/share/manual/manual.pdf or point a web browser at <span>$</span>(PREFIX)/share/manual/manual.html. Note that you may want to copy the entire share directory somewhere more convenient e.g., your user home directory before viewing in a browser.

## Building DDASToys

Clone the DDASToys repository and checkout the desired release tag:

```bash
git clone https://github.com/NSCLDAQ/DDASToys.git
cd DDASToys
git checkout <release-tag>
```

### Requirements

DDASToys requires:

- NSCLDAQ 12.1-000 or later
- CMake 3.13 or later
- CERN ROOT compatible with the NSCLDAQ installation being used
- Qt 5.11 or later (for `traceview`)
- LibTorch (for machine-learning inference support)
- NVIDIA CUDA Toolkit and `libCudaOptimize` (optional, for GPU-accelerated fitting)

Before building, configure the appropriate NSCLDAQ and ROOT environments:

```bash
source /path/to/nscldaq/daqsetup.bash
source /path/to/root/bin/thisroot.sh
```

The DDASFormat library is included as a git submodule. After cloning the repository, initialize and update the submodules:

```bash
git submodule update --init --recursive
```

### Configuring and Building

Create a build directory and configure the project:

```bash
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=/path/to/install
```

For example:

```bash
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=/aaron/dev-install/ddastoys/6.5-dev
```

Build and install:

```bash
cmake --build build
cmake --install build
```

### Running Unit Tests

For CMake >= 3.20, you can run the test suite with:

```bash
ctest --test-dir build --VV
```

While for older CMake versions you will have to run from the build directory:

```bash
cd build && ctest --VV
```

All tests should complete successfully.

### Build Options

The following options may be specified during configuration using
`-D<option>=ON|OFF`:

| Option | Default | Description |
|----------|---------|-------------|
| `DDASTOYS_CUDA_DEV` | `OFF` | Build the developmental CUDA GPU-accelerated fit engine plugin libraries. Requires the NVIDIA CUDA Toolkit (`nvcc`) and the `libCudaOptimize` library. Not intended for production analysis! |
| `DDASTOYS_MLINFERENCE` | `ON` | Build the machine-learning inference fit editor. Requires LibTorch. |
| `DDASTOYS_TRACEVIEW` | `ON` | Build the Qt-based `traceview` diagnostic GUI. Requires Qt 5.11 or newer. |
| `DDASTOYS_DOCS` | `ON` | Build the Doxygen API documentation and DocBook user manual. |
| `DDASTOYS_TIMING` | `OFF` | Enable inference timing and profiling instrumentation in the FitEditors. |

For example, to disable documentation and TraceView:

```bash
cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=/path/to/install \
    -DDDASTOYS_DOCS=OFF \
    -DDDASTOYS_TRACEVIEW=OFF
```

### CUDA Configuration

`DDASTOYS_CUDA_DEV` builds two additional, experimental analytic plugins next to
the normal CPU `libFitEditorAnalytic.so` (which is unaffected):

* `libFitEditorCudaAnalytic.so` — the analytic fit compiled against the CUDA fit
  engine (GPU-computed residuals/Jacobian for the GSL Levenberg-Marquardt fit).
* `libFitEditorCudaPSO.so` — the analytic fit driven by the `libCudaOptimize` swarm
  (PSO/DE) optimizer, always performing both the single- and double-pulse fits.

Both require `nvcc` and `libCudaOptimize`, and you must specify the target GPU
architecture, e.g. for the Pascal card on spdaq-cuda (compute capability 6.1):

    cmake -S . -B build -DDDASTOYS_CUDA_DEV=ON -DCMAKE_CUDA_ARCHITECTURES=61

When `DDASTOYS_CUDA_DEV=ON`, the maximum supported trace length compiled into the CUDA fit engine may be configured via:

```bash
-DDDASTOYS_CUDA_MAXPOINTS=<N>
```

The default value is:

```text
DDASTOYS_CUDA_MAXPOINTS=1024
```

This value is compiled into the CUDA implementation as the `MAXPOINTS`
macro and should be chosen to accommodate the longest traces expected
during analysis. The maximum allowed value is 1024.

> **Note:** GPU fitting is a development/experimental feature and is not on the
> production path — the CPU pipeline is fast enough for real-time processing.
> Per-trace GPU fitting does not currently beat CPU fitting; see `gpu_fitting.md`
> for the findings and the batched-fitting plan.


### Installation

The installation places the DDASToys libraries, executables, headers,
documentation, and the bundled DDASFormat library under the directory
specified by `CMAKE_INSTALL_PREFIX`.

After installation, the DDASToys manual can be found in:

```text
<prefix>/share/manual/
```

Both PDF and HTML versions of the documentation are installed when
`DDASTOYS_DOCS=ON`.

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
* 6.4-001 : Pin DDASFormat 2.1-001.
* 6.5-000 : CMake build environment, stability fixes, improved checking of input configuration, split GPU fitting into its own plugin libraries for development