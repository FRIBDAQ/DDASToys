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
parallelized using either Threading and ZMQ or MPI.
2. Used as a loadable extension to Tcl that allows you to treat DDAS data from
within pure Tcl scripts.  The Makefile creates the pgkIndex.tcl index file.
3. The tracepeek.tcl script is a pure Tcl script that can be used to
   

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
                 dict. Some keys in the dict are always there.  Others are
		 only there if the appropriate data are there:
		 These keys are always present:
		 
    
