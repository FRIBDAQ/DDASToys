This directory contains the code for a shared library that can be used
to append fits to DDAS data using the NSCLDAQ 11.4 Transformer program.
The fits created arte identical to those created by ringblockdealer and,
as such, this directory replaces that code for ring item output.

To build this code you must:
1. Source the daqsetup.bash script from NSCLDAQ-11.4-002 or greater.
2. Define the environment variable SPECTCLHOME to point to the
   installation directory of SpecTcl 5.2 or greater.
3. source the ddassetup.bash script from DDAS 3.2-001 or later.
   Note this should be done last so that it does not override
   the NSCLDAQ definition.
   
The Makefile produces the libFitter.so shared object.  This can be

1. Used as the extension library for $DAQBIN/Transformer allowing fits to be
parallelized using either Threading and ZMQ or MPI. ($DAQBIN/Transformer --help
provides some simple documentation describing how to do this).  The
fitter requires the environment variable FIT_CONFIGFILE to point at a file
that describes which channels to fit, the limits over which the fit is performed
(usually all but the end points of the trace) and the saturation value of
the digitizer.   The file an contain blank lines.  Leading and trailing
spaces are not significant.  If the first non-whitespace character is a #
the line is ignored (treated as a comment).   Non comment lines contain five
whitespace separated unsigned integers that are, in order:
   * crate - the crate id of a channel to fit.
	* slot  - the slot number of a channel to fit.
	* channel - the channel number within the crate/slot that should be fitted.
	* first   - The index of the first trace point to consider for the fit (usually 1).
	* last    - The index of the last trace point to consider for the fit
	            (usually tracelength -2).
	* saturation - The last highest legal digitizer trace value, e.g.
	               16383  trace points with this value are assumed to be saturation
						and are not considered in the fit.

2. Used as a loadable extension to Tcl that allows you to treat DDAS data from
within pure Tcl scripts.  The Makefile creates the pgkIndex.tcl index file.
3. The tracepeek.tcl script is a pure Tcl script that can be used to visualize
DDAS data with traces and optionally fits.  To use this, TCLLIBPATH must
include $DAQROOT/TclLibs and the directory containing libFitter.so for example:
TCLLIBPATH="$DAQROOT/TclLibs ." tclsh tracepeek.tcl
4. Provide the DDASFitHit and FitHitUnpacker  classes which can unpack hits
that have fit data appended to them.  This class can unpack:
   * Hits with no fit extension.
   * Hits with fit extensions from ringblockdealer.
   * Hits with fit extensions from Transformer

Use the libFitter.so library with Transformer as described in the manpage
for that program.  The remainder of this document describes
the Tcl interface to DDAS data.

To load this interface into your program:

package require ddasunpack

The directory containing the library and its pkgIndex.tcl file must be
in the auto_path patch.   The package requires the command ensemble
ddas_unpack:

ddasunpack use filename - Specifies a data source.  The data source must be a file.
                 The command returns a handle that must be used with all
		 other commands in the ensemble.
ddasunpack close handle - Closes the data source whose use returned the handle.
ddasunpack next handle - gets the next physics event from the source.  The
                 event is returned as the command result. This result is a
                 list of dicts. One dict per hit in the built event.
		 Some keys in the dict are always there.  Others are
		 only there if the appropriate data are there:
		 These keys are always present:

		 crate -- crate id of the hit.
		 slot  -- the slot id of the hit.
		 channel-- the channel id of the hit.
		 energy -- the extracted energy of the hit.
		 time   -- the nanosecond time of the hit.

		 If the hit has a trace the dict will contain a trace key whose
		 value is the list of trace points.

		 If the trace has one or more fits associated with it, the dict
		 will have a fits dictionary key.  The value of this dict is
		 iteslf a dict that can one or both of fit1 and fit2 keys.
		 
		 fit1 if present describes the single pulse fit for the trace.
		 This is a dict that has the following keys:
		 position - the logistic position parameter.
		 amplitude - the scale factor of the fit.
		 steepness - the rise time parameter of the logistic of the
		             fit.
		 decaytime - the decay time parameter of the decay of the fit.
		 offset    - the DC Offset of the fit.
		 chisqure  - The fit chi square goodness.
		 fitstatus - The GSL Fit status.
		 
		 fit2 if present describes the double pulse fit.
		 This dict has the same keys as fit1 however position,
		 amplitude, steepness and decaytime are two element lists. The
		 first element is the associated fit parameter for the the first
		 pulse while the second is the fit parameter for the second
		 pulse. For example:  position {100 550} means the two fits
		 have positions at 100 and 550.
		 
		 
    
