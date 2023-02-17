This directory contains the code for a shared library that can be used to
append fits to DDAS data using the Transformer and EventEditor programs
present in NSCLDAQ installations 11.4 and later.

To build this code you must:

1. Source the daqsetup.bash script from NSCLDAQ-11.4-002 or later.
2. Define the environment variable SPECTCLHOME to point to the installation
   directory of SpecTcl 5.2 or greater.
3. Source the ddassetup.bash script from DDAS 3.2-001 or later. Note this
   should be done last so that it does not override the NSCLDAQ definition.
4. If the version of DDAS/NSCLDAQ use the broken out XIAAPI, define XIAAPIDIR
   to point to the top of the API version installed tree.

Note: The repository is not the NSCL gitlab repo because this is shared with
non FRIB/NSCL folks but

https://your-netid@gitlab.msu.edu/foxr/ParallelAnalysis.git

is a public repository.

The Makefile produces the libFitEditorAnalytic.so and libFitEditorTemplate.so
shared objects. These can be used as the extension libraries for
$DAQBIN/Transformer or $DAQBIN/EventEditor allowing fits to be parallelized
using either ZMQ threading or MPI. $DAQBIN/Transformer --help or
$DAQBIN/EventEditor --help provide some simple documentation describing how
to do this.

The fitter requires the environment variable FIT_CONFIGFILE to point at a file
that describes which channels to fit, the limits over which the fit is
performed (usually all but the end points of the trace) and the saturation
value of the digitizer. The file can contain blank lines. Leading and trailing
spaces are not significant. If the first non-whitespace character is a '#' the
line is ignored (treated as a comment). Non-comment lines contain five
whitespace separated unsigned integers that are, in order:

* crate   - The crate id of a channel to fit.
* slot    - The slot number of a channel to fit.
* channel - The channel number within the crate/slot that should be fitted.
* first   - The index of the first trace point to consider for the fit
  	    (usually 1, as the trace is zero-indexed).
* last    - The index of the last trace point to consider for the fit (usually
  	    length - 2 where length is the number of samples contained in the
	    trace).
* saturation - The last highest legal digitizer trace value, trace points with
  	       this value are assumed to be saturationand are not considered
	       in the fit. For a digitizer with a bit depth B this value should
	       less than or equal to 2^B - 1.

Additionally the template fitting library requires the environment variable
TEMPLATE_CONFIGFILE to point to a file which contains the template and some
metadata regarding its creation. Rules regardig whitespace and comments are
the same as described above for the file pointed to by FIT_CONFIGFILE. The
first line of the tempate configuration file consists of two
whitespace-separated unsigned integer values:

* align-point - the sample number of the template alignment point
* npts - number of samples contained in the template

The next npts lines each contain the double-precision template data with one
value per line.

An additional shared library, libDDASFitHitUnpacker.so, is also created by the
Makefile. This shared library contains the interface for unpacking DDAS hit
data with the fit extensions used by the currently implemented analytic and
template fit functions appended to each fit hit.
