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

set cratemask *
set slotmask  *
set chanmask  *

set eventNumber 0

##
#  Stuff to use for expansions:
#
set startX  0;            #  starting point of the expansion box
set startY  0

set endX    0;           # ending point of the expansion box.
set endY    0

set rubberBox "";        # canvas id of the rubbrerband box.

# This array helps us properly handle multiple hits from the same channel.
#  populateListbox uses it to maintain a correspondence between the
#  indices in the listbox and fragment indices in the event.

array set listboxFragmentIndices [list];    # List box index -> fragment index map.

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
# amplitude
#   This mimics DDAS::pulseAmplitude - which computes he
#   amplitude of a pulse  the amplitude dict entry isn't actually that
#   but an overal scale factor.  This computes the amplitude at the peak.
#
# @param fit  - fit dict for a fit.
# @return double - actual amplitude.
#
proc amplitude {fit} {
    set a [dict get $fit amplitude]
    set k1 [dict get $fit steepness]
    set k2 [dict get $fit decaytime]
    set x0 [dict get $fit position]
    
    set frac [expr {$k1/$k2}]
    if {$frac <= 1.0} {
        return -1;             # Can't compute due to log of negative.
    }
    set pos [expr {$x0 + log($frac - 1.0)/$k1}];  # position at peak.
    return [pulse $x0 $a $k1 $k2 0.0 $pos]
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
	if {[dict get $fit1 iterations] != 0} { #  fit actually performed.

	    foreach w [list spos samp ssteep sdecay soffset schi iter1]  \
                k [list position amplitude steepness decaytime offset chisquare iterations] {
		    set value [dict get $fit1 $k]
		    set value [format %7.4f $value]
		    .data.$w configure -text $value
		}
	    # The actual amplitude:
	    
	    set amp [amplitude $fit1]
	    .data.amplitude config -text [format %7.4f $amp]
	}
    }
    if {[dict exists $fits fit2]} {
	set fit2 [dict get $fits fit2]
	if {[dict get $fit2 iterations] != 0} {
	    set first [dict create]
	    foreach w [list pos1 amp1 steep1 decay1 doffset dchi iter2] \
		k [list position amplitude steepness decaytime offset chisquare iterations] {
		    
		    set value [lindex [dict get $fit2 $k] 0];    # first pulse.
		    dict set first $k $value
		    set value [format %7.4f $value]
		    .data.$w configure -text $value
		    
		}
	    set second [dict create]
	    foreach w [list pos2 amp2 steep2 decay2] \
		k [list position amplitude steepness decaytime] {
		    set value [lindex [dict get $fit2 $k] 1]
		    dict set second $k $value
		    set value [format %7.4f $value]
		    .data.$w configure -text $value
		}
	    set a1 [amplitude $first]
	    set a2 [amplitude $second]
	    
	    .data.amplitude1 config -text [format %7.4f $a1] 
	    .data.amplitude2 config -text [format %7.4f $a2]
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
# stretchBox
#   Responds to a motion event in the canvas while stretching a rubber band
#   expansion box.
#    - Save the new end x end y points.
#    - Update the rectangle
# @param canvas - canvas we're drawing on.
# @param x,y    - canvas positions we moved to.
#
proc stretchBox {canvas x y} {
    set ::endX $x
    set ::endY $y
    
    $canvas coords $::rubberBox $::startX $::startY $::endX $::endY
}

##
# expandPlot
#    Called when the MB1 is released to end selection of an expansion window.
#    - The endx/y are updated
#    - the rectangle is destroyed.
#    - The plot is expanded as indicated by x/y.
#    - Motion and Release events are turned off.
#    - presss event is re-enabled.
#
# @param canvas - the canvas being expanded.
# @param x,y    - Cursor position.
#
proc expandPlot {canvas x y} {
    set ::endX $x
    set ::endY $y

    bind $canvas <Motion> ""
    bind $canvas <ButtonRelease-1> ""
    bind $canvas <ButtonPress-1> [list startRubberBox $canvas %x %y]
    
    $canvas delete $::rubberBox
    
    # Convert the coordinates to plot coords and make them a lower left
    # and upper right pair of coords.
    
    set pt1 [::Plotchart::pixelToCoords $canvas $::startX $::startY]
    set pt2 [::Plotchart::pixelToCoords $canvas $::endX $::endY]
    
    set xll [expr {min([lindex $pt1 0], [lindex $pt2 0])}]  ; # lower left coords
    set yll [expr {min([lindex $pt1 1], [lindex $pt2 1])}]
    
    set xur  [expr {max([lindex $pt1 0], [lindex $pt2 0])}]
    set yur [expr {max([lindex $pt1 1], [lindex $pt2 1])}]
    
    #  Get sane axis specs for the expansion:
    
    set xAxis [::Plotchart::determineScale $xll $xur]
    set yAxis [::Plotchart::determineScale $yll $yur]
    
    reinitPlot $xAxis $yAxis
    showFrag .control.fraglist 0
    
}
##
#  startRubberBox
#    Starts making a rubber-band expansion box.
#    Set the starting point in pixels.
#    Remove the mb1 down event binding.
#    Set a motion event binding.
#    Set an mb up event binding.
#    Create a zero extent rectangle on the canvas in dashed line.
#
# @param canvas - the canvas we're drawing on.
# @param x      - X position of the mouse.
# @param y      - Y position of the mouse.
#
proc startRubberBox {canvas x y} {
    set ::startX $x
    set ::startY $y
    
    set ::endX   $x
    set ::endY   $y
    
    bind $canvas <ButtonPress-1> "";      # Removes the down binding.
    bind $canvas <Motion> [list stretchBox $canvas %x %y]
    bind $canvas <ButtonRelease-1> [list expandPlot $canvas %x %y]
    
    set ::rubberBox \
        [$canvas create rectangle $::startX $::startY $::endX $::endY \
         -dash {2 2}]
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
    
    bind .plot <ButtonPress-1> [list startRubberBox .plot %x %y]
}
##
# reinitPlot
#   Reinitialize the plot with specified axes (e.g. expansion).
# @param xaxis - x axis specifications.
# @param yaxis - y axis specifications.
#
proc reinitPlot {xaxis yaxis} {
    if {$::chart ne ""} {
        $::chart deletedata
    }
    destroy .plot
    
    set c [canvas .plot -height 600 -width 800]
    grid $c -sticky nsew
    set ::chart [::Plotchart::createXYPlot $c $xaxis $yaxis]
    bind .plot <ButtonPress-1> [list startRubberBox .plot %x %y]
}

##
# plot
#   Plot a series with a given name, points and color
#
# @param name - plotchart series name.
# @param xs   - X coordinates.
# @param ys   - Y coordinates.
# @param color - color for plot.
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
# getFragment
#   Returns the fragment from the current event whose crate/slot/ch
#   matches that in the selection string.
#
# @param sel - selection string crate:slot:chan
# @return fragment dict.
#
proc getFragment sel {
    set selList [split $sel :]
    set crate [lindex $selList 0]
    set slot  [lindex $selList 1]
    set chan  [lindex $selList 2]
    
    foreach frag $::currentEvent {
        if {($crate == [dict get $frag crate])      &&
            ($slot   == [dict get $frag slot])       &&
            ($chan   == [dict get $frag channel])} {
                return $frag
            }
    }
    error "No matching fragment for c:s:chan : $sel ($crate $slot $chan)"
}


##
# showFrag
#   Display trace and fit information for the selected event.
#   This is normally bound to the double-1 event in the list box.
#
# @param w - widget in which the fragment has been selected.
#
proc showFrag {w {init 1}} {
    set i [$w curselection]
    if {$i ne ""} {
        #  There's a selection.
        clearData
	
        set selection [$w get $i]
        set frag [lindex $::currentEvent $::listboxFragmentIndices($i)]
        #  set frag [getFragment $selection]

	#  Put in the classifier probabilities if they're there:

	if {[dict exists $frag singlePulseProbability]} {
	    set single [dict get $frag singlePulseProbability]
	} else {
	    set single ****
	}
	if {[dict exists $frag doublePulseProbability]} {
	    set double [dict get $frag doublePulseProbability]
	} else {
	    set double ****
	}

	.data.singleprob configure -text $single
	.data.doubleprob configure -text $double
	
        if {[dict exists $frag fits]} {
            showFits [dict get $frag fits]
        }
        #  There's always a trace..though there may not be fits.
        
        set trace [dict get $frag trace]
        set x     [computeXcoords $trace]
        if {$init} {
            initPlot [llength $x]
        }
        plot trace $x $trace black

        # Note we do double pulse fit first so since most pulses are
        # single pusles.  This means in most cases there's not a real
        # difference between single and double pulse fits. Putting the
        # single pulse on top makes it visible in those cases.
        
        if {[dict exists $frag fits] } {
            set fits [dict get $frag fits]
            if {[dict exists $fits fit2]} {
		set fit2 [dict get $fits fit2]
		if {[dict get $fit2 iterations] != 0} {
		    set y [computeDoublePulse $x [dict get $fit2]]
		    plot double-pulse $x $y blue
		}
	    }
            
            if {[dict exists $fits fit1]} {
		set fit1 [dict get $fits fit1]
		if {[dict get $fit1, iterations] != 0} {
		    set y [computeSinglePulse $x $fit1]
		    plot single-pulse $x $y red
		}
	    }
        }

    } 
}
##
# populateListbox
#   Populates the list box with the channels from the event that
#   have traces and match the glob patterns in the cratemask, slotmask and chanmask
#   globals.
#

proc populateListbox {} {
    array unset ::listboxFragmentIndices *;   # Clear the correspondences.
    
    .control.fraglist delete 0 end;           # clear
    set fragno 0
    set itemno 0
    
    foreach fragment $::currentEvent {
        set crate [dict get $fragment crate]
        set slot  [dict get $fragment slot]
        set ch    [dict get $fragment channel]
    
        set crmask [split $::cratemask ,]
        set slmask [split $::slotmask ,]
        set chmask [split $::chanmask ,]
        
        set populated 0
        foreach cmask $crmask {
            foreach smask $slmask {
                foreach chanmask $chmask {
                    if {[string match $crmask $crate]     &&
                        [string match $smask $slot]       &&
                        [string match $chanmask $ch]} {
                            if {!$populated} {
                                .control.fraglist insert end $crate:$slot:$ch
                                set populated 1;   #in case of multiple matches.
                                set ::listboxFragmentIndices($itemno) $fragno
                                incr itemno
                            }
                        }
                }
            }
        }
        incr fragno

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
    
    set filter [ttk::frame $c.filter -relief groove -borderwidth 3]
    ttk::label $filter.crfiltl -text crate
    ttk::label $filter.slfiltl -text slot
    ttk::label $filter.chfiltl -text channel
    grid $filter.crfiltl $filter.slfiltl $filter.chfiltl -sticky w
    
    ttk::entry $filter.crfilter -textvariable ::cratemask
    ttk::entry $filter.slfilter -textvariable ::slotmask
    ttk::entry $filter.chfilter -textvariable ::chanmask
    grid $filter.crfilter $filter.slfilter $filter.chfilter
    foreach w [list $filter.crfilter $filter.slfilter $filter.chfilter] {
        bind $w <Key-Return>  populateListbox
    }
    ttk::button $filter.reset -text "Reset filter" -command {
        set ::cratemask *
        set ::slotmask *
        set ::chanmask *
        populateListbox
    }
    grid $filter.reset
    
    grid $c.llabel
    grid $c.fraglist $c.fragscroll  -sticky nsew
    grid $filter -row 1 -column 2   -sticky ns
    
    
    set nextframe [ttk::frame $c.next]
    ttk::button $nextframe.next -text {Next Event} -command nextEvent
    ttk::label  $nextframe.evtlbl -text "Event "
    ttk::label  $nextframe.evno   -textvariable ::eventNumber
    
    grid $nextframe.next $nextframe.evtlbl $nextframe.evno -padx 5
    
    set skip [ttk::frame  $c.skip -relief groove -borderwidth 3]
    ttk::spinbox $skip.count -from 1 -to 10000 -increment 1 -width 5
    $skip.count set 1
    ttk::button $skip.skip -text {Skip} -command skipEvents
    
    grid $skip.count $skip.skip
    grid   $nextframe - $skip
    
    grid $c -sticky nsew
    
    set d [ttk::frame .data]
    
    ttk::label $d.slabel -text {Single}
    ttk::label $d.p1label -text {First}
    ttk::label $d.p2label -text {Second}
    ttk::label $d.plabel -text {Position}
    ttk::label $d.alabel -text {Scale}
    ttk::label $d.amplabel -text {Amplitude}
    ttk::label $d.stlabel -text {Steepness}
    ttk::label $d.dlabel -text {Decay Time}
    ttk::label $d.olabel -text {Offset}
    ttk::label $d.chlabel -text {Chi Squre}
    ttk::label $d.iterlabel -text {Iterations}

    
    #  Single pulse items:
    
    ttk::label $d.spos   -text {***}
    ttk::label $d.samp   -text {***}
    ttk::label $d.amplitude -text {***}
    ttk::label $d.ssteep -text {***}
    ttk::label $d.sdecay -text {***}
    ttk::label $d.soffset -text {***}
    ttk::label $d.schi  -text {***}
    ttk::label $d.iter1 -text {***}

    
    # First of two pulse itesm:
    
    ttk::label $d.pos1 -text {***}
    ttk::label $d.amp1 -text {***}
    ttk::label $d.amplitude1 -text {***}
    ttk::label $d.steep1 -text {***}
    ttk::label $d.decay1 -text {***}
    ttk::label $d.doffset -text {***}
    ttk::label $d.dchi -text {***}
    ttk::label $d.iter2  -text {***}

    
    ttk::label $d.pos2 -text {***}
    ttk::label $d.amp2 -text {***}
    ttk::label $d.amplitude2 -text {***}
    ttk::label $d.steep2 -text {***}
    ttk::label $d.decay2 -text {***}

    # Classifier items:

    ttk::label $d.psinglel   -text {Single pulse probability}
    ttk::label $d.singleprob -text {****}
    ttk::label $d.pdoublel   -text {Double pulse probability}
    ttk::label $d.doubleprob -text {****}
    
    grid x $d.slabel $d.p1label $d.p2label \
	$d.psinglel $d.singleprob    -sticky w
    grid $d.plabel $d.spos     $d.pos1  $d.pos2 \
	$d.pdoublel $d.doubleprob     -sticky w
    grid $d.alabel $d.samp     $d.amp1  $d.amp2 -sticky w
    grid $d.amplabel $d.amplitude $d.amplitude1 $d.amplitude2 -sticky w
    grid $d.stlabel $d.ssteep  $d.steep1 $d.steep2 -sticky w
    grid $d.dlabel  $d.sdecay  $d.decay1 $d.decay2 -sticky w
    grid $d.olabel  $d.soffset $d.doffset -sticky w
    grid $d.chlabel $d.schi    $d.dchi -sticky w
    grid $d.iterlabel $d.iter1 $d.iter2 -stick w
    
    grid $d -sticky nsew
    
    bind $c.fraglist <Double-1> [list showFrag %W]
}
##
# clearData
#   Clears the fit data
#
proc clearData {} {
    foreach w [list spos samp amplitude ssteep sdecay soffset schi singleprob doubleprob] {
        .data.$w configure -text {***}
    }
    foreach w [list pos1 pos2 amp1 amplitude1  amp2  amplitude2 steep1 steep2 decay1 decay2 doffset dchi] {
        .data.$w configure -text {***}
    }
}
##
# notifyEndFile
#   Let the user know we've got no more data:
#
proc notifyEndFile {} {
    tk_messageBox -icon info -parent . -title "End of file" -type ok \
            -message {No more physics events in this file.}    
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
    
    if {$rawEvent eq ""} {         ; # End of file.
        notifyEndFile
    } else  {
        incr ::eventNumber        
        set ::currentEvent [list]
    
        foreach hit $rawEvent {
            
            if {[dict exists $hit trace]} {
                
                lappend ::currentEvent $hit
                set crate [dict get $hit crate]
                set slot  [dict get $hit slot]
                set chan  [dict get $hit channel]
                
            }
        }
        populateListbox
        
        clearData
    }
    
}
##
# skipEvents
#   Skip forward n events where n is the number in .control.skip.count
#   If skipping hits or results in an end file; then we pop up a notification
#   to that effect.
#
proc skipEvents {} {
    set count [.control.skip.count get]
    while {$count > 1} {
        set rawEvent [ddasunpack next $::handle]
        if {$rawEvent eq ""} {;                # endfile.
            notifyEndFile
            return
        }
        incr ::eventNumber
        incr count -1
    }
    nextEvent;              # Done skipping
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
