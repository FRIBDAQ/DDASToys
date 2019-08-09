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
#             Giordano Cerriza
#	     NSCL
#	     Michigan State University
#	     East Lansing, MI 48824-1321


##
# @file tracepeek.tcl
# @brief Visualize traces and, if available their fits.
# @author Ron Fox <fox@nscl.msu.edu>
#

package require Tk
package require ddasunpack
package require Plotchart 3.0

set filename ""
set currentEvent [list]
set chart ""


##
# pulse
#   For a given set of pulse parameters and x position, compute
#   The pulse value.
#
# @param p - position parameter of the pulse.
# @param a - amplitude scale factor.
# @param r - Rise time parameter.
# @param f - Fall time parameter.
# @param o - DC Offset.
# @param x - Position at which to evaluate the pulse.
#
proc pulse {p a r f o x} {
    set num [expr {$a*exp(-$f*($x - $p))}]
    set den [expr {1 + exp(-$r*($x - $p))}]
    set result [expr {$o + $num/$den}]
    
    return $result
}


##
# showFits
#   Update the fit widgets with the fit values in the event.
#   We don't assume fit1 or fit2 are defined:
#
# @param fits - fits dict.
#
proc showFits fits {
    if {[dict exists $fits fit1]} {
        set fit1 [dict get $fits fit1]
        foreach w [list spos samp ssteep sdecay soffset schi]  \
                k [list position amplitude steepness decaytime offset chisquare] {
            set value [dict get $fit1 $k]
            set value [format %7.4f $value]
            .data.$w configure -text $value
        }
    }
    if {[dict exists $fits fit2]} {
        set fit2 [dict get $fits fit2]
        foreach w [list pos1 amp1 steep1 decay1 doffset dchi] \
            k [list position amplitude steepness decaytime offset chisquare] {
            
            set value [lindex [dict get $fit2 $k] 0];    # first pulse.
            set value [format %7.4f $value]
            .data.$w configure -text $value
        }
        foreach w [list pos2 amp2 steep2 decay2] \
            k [list position amplitude steepness decaytime] {
            set value [lindex [dict get $fit2 $k] 1]
            set value [format %7.4f $value]
            .data.$w configure -text $value
        }
    }
}
##
# computeXcoords
#   Makes a list of x coordinates for the plot
# @param ys- coords are 0 - [llength $ys] -1
#
# @return list of x coords.
#
proc computeXcoords ys {
    set result [list]
    set x 0
    for {set i 0} {$i < [llength $ys]} {incr i} {
        lappend result $x
        incr x
    }
    return $result
}
##
# initPlot
#    Initialize the plot
#
# @param nx -number of samples.  We assume data is 14 bits.
proc initPlot nx {
    if {$::chart ne ""} {
        $::chart deletedata
    }
    destroy .plot
    
    set c [canvas    .plot -height 600 -width 800]
    grid $c -sticky nsew 
    set ::chart [::Plotchart::createXYPlot $c [list 0 $nx 50] [list 0 16400 500] -xlabels sample ]
}
proc plot {name xs ys color} {
    $::chart dataconfig $name -color $color
    foreach x $xs y $ys {
        $::chart plot $name $x $y
    }
    $::chart legend $name $name
}
##
# computeSinglePulse
#   Computes the single pulse fit values for a set of x coordinates.
#
# @param xs - x coordinate list.
# @param fit1  fit1 dict entry for a frag.
# @return list - y points corresponding to x points.
#
proc computeSinglePulse {xs fit1} {
    set pos [dict get $fit1 position]
    set amp [dict get $fit1 amplitude]
    set rise [dict get $fit1 steepness]
    set fall [dict get $fit1 decaytime]
    set o    [dict get $fit1 offset]
    
    set result [list]
    foreach x $xs {
        lappend result [pulse $pos $amp $rise $fall $o $x]
    }
    return $result
}
##
# computeDoublePulse
#
#  Compute a double pulse.
#
# @param xs - x coordinate list in which to compute the fit.
# @param fit2 - the 2 pulse fit dict entry from a fragment.
# @return list - y coordinates for the fitted pulse.
#
proc computeDoublePulse {xs fit2} {
    set p1 [lindex [dict get $fit2 position] 0]
    set p2 [lindex [dict get $fit2 position] 1]
    set a1 [lindex [dict get $fit2 amplitude] 0]
    set a2 [lindex [dict get $fit2 amplitude] 1]
    set r1 [lindex [dict get $fit2 steepness] 0]
    set r2 [lindex [dict get $fit2 steepness] 1]
    set d1 [lindex [dict get $fit2 decaytime] 0]
    set d2 [lindex [dict get $fit2 decaytime] 1]
    set o1 [dict get $fit2 offset]
    set o2 0;                     #nice cheat for common offset
    
    set result [list]
    foreach x $xs {
        lappend result [expr \
            {[pulse $p1 $a1 $r1 $d1 $o1 $x] + [pulse $p2 $a2 $r2 $d2 $o2 $x]}]
    }
    return $result
}

##
# showFrag
#   Display trace and fit information for the selected event.
#   This is normally bound to the double-1 event in the list box.
#
# @param w - widget in which the fragment has been selected.
#
proc showFrag w {
    set i [$w curselection]
    if {$i ne ""} {
        #  There's a selection.
        
        set frag [lindex $::currentEvent $i]
        
        if {[dict exists $frag fits]} {
            showFits [dict get $frag fits]
        }
        #  There's always a trace..though there may not be fits.
        
        set trace [dict get $frag trace]
        set x     [computeXcoords $trace]
        initPlot [llength $x]
        plot trace $x $trace black

        # Note we do double pulse fit first so since most pulses are
        # single pusles.  This means in most cases there's not a real
        # difference between single and double pulse fits. Putting the
        # single pulse on top makes it visible in those cases.
        
        if {[dict exists $frag fits] } {
            set fits [dict get $frag fits]
            if {[dict exists $fits fit2]} {
                set y [computeDoublePulse $x [dict get $fits fit2]]
                plot double-pulse $x $y blue
            }
            
            if {[dict exists $fits fit1]} {
                set y [computeSinglePulse $x [dict get $fits fit1]]
                plot single-pulse $x $y red
            }
        }

    }
    
    
    
}
##
# setupUi
#    We have a button for the next event,
#    A list box for the hits that have a trace.
#    An area to show the fits.
#    A title spot for the file.

proc setupUi {} {
    wm title . $::filename
    
    set c [ttk::frame .control]
    ttk::label $c.llabel -text {Crate:slot:channel of fragments with traces}
    listbox    $c.fraglist -yscrollcommand [list $c.fragscroll set]
    ttk::scrollbar  $c.fragscroll -orient vertical -command [list $c.fraglist yview]
    
    grid $c.llabel
    grid $c.fraglist $c.fragscroll  -sticky nsew
    
    ttk::button $c.next -text {Next Event} -command nextEvent
    grid   $c.next
    
    grid $c -sticky nsew
    
    set d [ttk::frame .data]
    
    ttk::label $d.slabel -text {Single}
    ttk::label $d.p1label -text {First}
    ttk::label $d.p2label -text {Second}
    ttk::label $d.plabel -text {Position}
    ttk::label $d.alabel -text {Amplitude}
    ttk::label $d.stlabel -text {Steepness}
    ttk::label $d.dlabel -text {Decay Time}
    ttk::label $d.olabel -text {Offset}
    ttk::label $d.chlabel -text {Chi Squre}

    
    #  Single pulse items:
    
    ttk::label $d.spos   -text {***}
    ttk::label $d.samp   -text {***}
    ttk::label $d.ssteep -text {***}
    ttk::label $d.sdecay -text {***}
    ttk::label $d.soffset -text {***}
    ttk::label $d.schi  -text {***}

    
    # First of two pulse itesm:
    
    ttk::label $d.pos1 -text {***}
    ttk::label $d.amp1 -text {***}
    ttk::label $d.steep1 -text {***}
    ttk::label $d.decay1 -text {***}
    ttk::label $d.doffset -text {***}
    ttk::label $d.dchi -text {***}

    
    ttk::label $d.pos2 -text {***}
    ttk::label $d.amp2 -text {***}
    ttk::label $d.steep2 -text {***}
    ttk::label $d.decay2 -text {***}
    
    grid x $d.slabel $d.p1label $d.p2label   -sticky w
    grid $d.plabel $d.spos     $d.pos1  $d.pos2     -sticky w
    grid $d.alabel $d.samp     $d.amp1  $d.amp2 -sticky w
    grid $d.stlabel $d.ssteep  $d.steep1 $d.steep2 -sticky w
    grid $d.dlabel  $d.sdecay  $d.decay1 $d.decay2 -sticky w
    grid $d.olabel  $d.soffset $d.doffset -sticky w
    grid $d.chlabel $d.schi    $d.dchi -sticky w
    
    grid $d -sticky nsew
    
    bind $c.fraglist <Double-1> [list showFrag %W]
}
##
# clearData
#   Clears the fit data
#
proc clearData {} {
    foreach w [list spos samp ssteep sdecay soffset schi] {
        .data.$w configure -text {***}
    }
    foreach w [list pos1 pos2 amp1 amp2 steep1 steep2 decay1 decay2 doffset dchi] {
        .data.$w configure -text {***}
    }
}
##
# nextEvent
#   Gets the next event from file.
#   removes all channels that don't have traces and populates the
#   .control.fraglist box with the  crate:slot:channel of those that
#   Remain.
#
proc nextEvent {} {
    set rawEvent [ddasunpack next $::handle]
    set ::currentEvent [list]
    .control.fraglist delete 0 end
    foreach hit $rawEvent {
        
        if {[dict exists $hit trace]} {
            
            lappend ::currentEvent $hit
            set crate [dict get $hit crate]
            set slot  [dict get $hit slot]
            set chan  [dict get $hit channel]
            
            .control.fraglist insert end $crate:$slot:$chan
        }
    }
    
    
    clearData
    
}

##
#  Choose the file to peek, transform it into a
#  URI and open it:

set filename [tk_getOpenFile -title {Choose file} -filetypes [list    \
        [list {Event Files} {.evt}  ]                          \
        [list {All Files}    * ]                               \
        ]]
        
set uri file://$filename

set handle [ddasunpack use $uri]

setupUi 
