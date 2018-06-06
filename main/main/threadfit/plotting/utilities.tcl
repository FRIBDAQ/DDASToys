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
# @file utilities.tcl
# @brief Utility tcl functions for the DDAS analysis playground.
# @author Ron Fox <fox@nscl.msu.edu>
#


##
# readDataSet
#   Reads a data set from file. Each data set is of the form described in the
#   header comments.  After the read is completed the file is either at EOF
#   (if there are no more data sets), or positioned so the next gets will give
#   the title of the next data set in the file.
#
# @param fd    - The file descriptor to read the data set from
# @param index - which column of the file has the y values (defaults to 1).
# @return dict - A dict containing the following keys:
#                *  title - the title of the plot.
#                *  x     - X coordinates of the points.
#                *  y     - Y coordinatds of the points.
# The assumption is that titles start with non numeric strings.
# so  "Plot of 16C cross sections" is ok but "16C cross sections" will break
# the program.  Sorry but this is supposed to be simple to use and program.
#
proc readDataSet {fd {index 1}} {
    
    # Create the empty result dict and get the title:
    
    set result [dict create title "" x [list] y [list]]
    set line [gets $fd]
    dict set result title $line
    gets $fd;                             # Skip the following blank line.
    
    # Get the data points stoppin when the first char of the line is non digit
    # or eof.
    
    while {![eof $fd]} {
        set here [tell $fd];              # So we can reset the file position.
        set line [gets $fd]
            
        # See if this is another title:
        
        if {![string is digit -strict [string range $line 0 0]]} {
            seek $fd $here start;        # Restore the file position
            break
        }
        #  We should have data:

	dict lappend result x [lindex $line 0]
	dict lappend result y [lindex $line $index]
	
    }
    return $result
}

proc readFitDataSet fd {
    set result [dict create title "" x [list] y1 [list] y2 [list] diff [list]]
    set line [gets $fd]
    dict set result title $line

    while {![eof $fd]} {
	set line [gets $fd]
	if {[llength $line] == 4} {
	    dict  lappend result x [lindex $line 0]
	    dict  lappend result y1 [lindex $line 1]
	    dict lappend result y2 [lindex $line 2]
	    dict lappend result diff [lindex $line 3]
	}
    }
    return $result
}

##
# writeDataSet
#
#  Write a data set that is in the normal form.
#
# @param fd      - the output file.
# @param dataset - dict containing the data set to write.
#
proc writeDataSet {fd dataset} {
    puts $fd [dict get $dataset title]
    puts $fd ""
    
    foreach x [dict get $dataset x] y [dict get $dataset y] {
        puts $fd "$x $y" 
    }
    
}

##
# differentiate
#   Produces a new dataSet that is the derivative of the old one.
#   title becomes dydx (original).
#
# The series will be one point fewer (first new point is at the second)
# old data point
#
# @param dataSet - input series
# @return dataSet - differentiated otuput.
#
proc differentiate dataSet {
    set xpts [dict get $dataSet x]
    set ypts [dict get $dataSet y]
    
    set outX [list]
    set outY [list]
    
    
    
    set npts [llength $xpts]
    set priorX [lindex $xpts 0]
    set priorY [lindex $ypts 0]
    for {set i 1} {$i < $npts} {incr i} {
    
        set x [lindex $xpts $i]
        set y [lindex $ypts $i]
    
        # may be a blank line at the end of input:
    
        if {$x ne "" } {
            set dydx [expr {($y - $priorY)/($x - $priorX)}]
            lappend outX $x
            lappend outY $dydx
            
            set priorX $x
            set priorY $y
            
        }
    }
    
    set outputSet [dict create \
        x $outX y $outY title "dydx [dict get $dataSet title]" \
    ]
    
    
    return $outputSet
}

##
# getParameter
#   Get the value of a parameter that might be in an environment variable but
#   has a default if not.
#
# @param default - Default value of the parameter.
# @param envName - Name of environment variable that can override the default
#
# @return The actual value of the parameters.
#
proc getParameter {default envName} {
    if {[array names ::env $envName] eq $envName} {
        return $::env($envName)
    } else {
        return $default
    }
}
##
# findPeaks
#    Given a data set, locate the extents of all peaks whose heights exceed
#    a threshold.
# @param dataSet   - the data set to look at.
# @param threshold - Y threshold that defines a peak start.
# @param low       - Y threshold that defines a peak end.
# @return list of two element lists.  Each pair is the starting and ending
#         X coordinate of the peak.  If the peak never comes down, the right most
#         x cooreinate is returned.
# @note naturally the list can be empty.

proc findPeaks {dataSet threshold low} {
    set xpts [dict get $dataSet x]
    set ypts [dict get $dataSet y]
    set npts [llength $xpts]
    
    set result [list]
    
    set above 0;                  # 1 when we are above the threshold
    set peak [list];              # x1, x2 of one peak.
    for {set i 0} {$i < $npts} {incr i} {

        set y [lindex $ypts $i]       

        # If above is true we're looking for the endpoint
        # if not we're looking for the starting point.
        
        if {$above} {

            if {$y < $low} {
                lappend peak [lindex $xpts $i]
                lappend result $peak
                
                set above 0
                set peak [list]
            }
            
        } else {
            if {$y > $threshold} {
                set peak [lindex $xpts $i]
                set above 1
            }
        }
    }
    # If we're still above then finish off the last peak:
    
    if {$above} {
        lappend peak [lindex $xpts end]
        lappend result $peak
    }
    return $result
    
}
##
# logistic
#   Compute a logistic function.  See doubleLogistic below.
#
# @param A - scale factor.
# @param k - Steepness.
# @param x0 - position.
# @param x  - Where to evaluate the function.
#
proc logistic {A k x0 x} {
    return [expr {$A/(1+ exp(-$k*($x - $x0)))}]
}

##
# doubleLogistic
#   Compute a sum of two logistic functions sitting on a constant backgroun.
#   The form of a logistic is:
#     f(x) = A/(1+exp(-k(x - x0)))).  where A is a scale factor, k the steepness
#     and x0 the position.   The second logistic is also multiplied by a decay term.
#
# @param C    - Constant value.
# @param A1   - Scale factor of the first logistic.
# @param k1   - Steepness of the first logistic.
# @param x1   - position of the first logistic.
# @param A2   - Scale factor of the second logistic.'
# @param k2   - Steepness of the second logistic.
# @param x2   - position of the second logistic.
# @param m    - decay term slope.
# @param b    - intercept term of decay.
# @param  x   - Where to evaluate the function.
#
#
proc doubleLogistic {C A1 k1 x1 A2 k2 x2 m b x} {
    set L1 [logistic $A1 $k1 $x1 $x]
    set L2 [logistic $A2 $k2 $x2 $x]
    set decay [expr {[logistic 1.0 1000.0 $x1 $x] * ($m * ($x - $x1) + $b)}]
   
    return [expr {$C + $L1 + $L2 + $decay}]
}
##
# genLogisticFitSet
#    Generate a fit data set from a fit file.
#    The fit file contains:
#    chilsqr
#    C
#    A1 k1 x1
#    A2 k2 x2
#
#   Where C,A1,k1,x1,A2,k2,x2 are parameters in the fit function:
#
#   f(y) = C + A1/(1+exp(-k1(x-x1))) + A2/(1+exp(-k2(x-x2)))
#
#   Note that the two non-constant terms are logistic functions which analytically
#   approximate step functions whos height is An, whose 'steepness', is kn with
#   half way point at xn
#
# @param fname - name of the file containing the goodies.
# @param npts  - Number of points for which to generate the fit.
#
#  @note the fit data set is a dict like any other with the keys
#    -  title - "Fit from file: fname"
#    -  x     - Coordinates of x points.
#    -  y     - Coordinates of y points.
#    -  chisquare - Chisquare value
#    -  C     - C fit parameter.
#    -  A1    - A1 fit parameter
#    etc. with keys for each fit parameter.
#

proc genLogisticFitSet {fname npts} {
    set result [dict create title "Fit fromfile $fname"]
    set fd [open $fname r]
    
    dict set result chisquare [gets $fd]
    dict set result C [gets $fd]
    set C [dict get $result C]
    
    scan [gets $fd] "%f %f %f" A1 k1 x1
    dict set result A1 $A1
    dict set result k1 $k1
    dict set result x1 $x1
    
    scan [gets $fd] "%f %f %f" A2 k2 x2
    dict set result A2 $A2
    dict set result k2 $k2
    dict set result x2 $x2

    scan [gets $fd] "%f %f" m b

    dict set result m $m;
    dict set result b $m;       
    
    
    #  Now compute the function in the range [0...npts)
    
    dict set result x [list]
    dict set result y [list]
    for {set x 0} {$x < $npts} {incr x} {
        dict lappend result x $x
        set y  [doubleLogistic $C $A1 $k1 $x1 $A2 $k2 $x2 $m $b $x]
        dict lappend result y $y
    }
        
    
    
    
    close $fd
    return $result
}
