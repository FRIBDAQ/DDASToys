#!/bin/sh
# -*- tcl -*-
# The next line is executed by /bin/sh, but not tcl \
exec tclsh "$0" ${1+"$@"}

#    This software is Copyright by the Board of Trustees of Michigan
#    State University (c) Copyright 2014.
#
#    You may use this software under the terms of the GNU public license
#    (GPL).  The terms of this license are described at:
#
#     http://www.gnu.org/licenses/gpl.txt
#
#    Authors:
#             Ron Fox
#             Jeromy Tompkins 
#	     NSCL
#	     Michigan State University
#	     East Lansing, MI 48824-1321



##
# @file plotTraceFile.tcl
# @brief plots traces in a n ascci file.
#

##
# Usage:
#     plotTraceFile file
#
#   The file argument is the name of a file with traces as ascii.  Each trace hs
#   a header telling how many samples in the trace each sample then follows
#   one per line.
#
lappend auto_path [file join $::env(DAQROOT) TclLibs]
package require Tk
package require Plotchart 3.0




set utilDir [file join [file dirname [info script]] plotting]
source $utilDir/plotUtils.tcl


set canvasWidth 800
set canvasHeight 500




    
set fd [open $argv r]
set trace 0
while {![eof $fd]} {
    incr trace
    canvas .c -height $canvasHeight -width $canvasWidth
    bind .c <Button-1> [list incr ::hit]
    
    # Read the trace into x/y
    
    set traceLen [gets $fd]
    set traceLen [lindex $traceLen 0]
    set x [list]
    set y [list]
    set title "Raw trace $trace"
    set sample 0
    for {set i 0} {$i < $traceLen} {incr i} {
	lappend x $sample
	lappend y [lindex [gets $fd] 1]
	incr sample
    }
    # Make a data set:

    set dataSet [dict create title $title x $x y $y]
    set plot [plotDataSet $dataSet .c]

    pack .c
    vwait ::hit

    $plot deletedata

    destroy .c
}
