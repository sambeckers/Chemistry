#!/usr/bin/python

# a short script to read information from file to 
#	set up input files and run chemical model


import os
import numpy as np

print " "
print "******************************************"
print "Python wrapper for running chemical models"
print "******************************************"
print " "

# set path to input files
particlepath = './Cloud_Filaments/Sample_Files/'

print "Directory path is : ", particlepath
print " "

# open and read selected trace particle numbers for high Av cut
tracefile = open(particlepath+"highest_av_points_sorted.txt", "r")

traceid = []
tracestr = []

for x in tracefile:
	traceid.append(x.split()[0])
	
# for "full" model, create a list containing particle ids 
# set limits here too that identifies the directory

limits = np.array([0,1000])
#traceid = range(limits[0]+1,limits[1])

# set strings for appending to filenames

# create file=name for local file_parameter.txt

parafile = "file_parameters_"+str(limits[1])+".txt"
	
for i in traceid:	
	if (int(i) < 10): tracestr.append('000'+i)	
	if (int(i) >= 10) and (int(i) < 100): tracestr.append('00'+i)	
	if (int(i) >= 100) and (int(i) < 1000): tracestr.append('0'+i)	
	if (int(i) > 1000): tracestr.append(i)
		
# loop over traceid, set pathway to files
	
infile = []
outfile = []
ratesfile = []
logfile = []
physfile = []

ntrace = 0

for i in range(len(traceid)):
				
	if (int(traceid[i]) >= limits[0]) and (int(traceid[i])< limits[1]):
	
		ntrace=ntrace+1
				
		if (limits[1] < 10000): 
			infile.append (particlepath+'Particles_0'+str(limits[1])+'/Input/particle_trace_'+tracestr[i]+'.txt')
			outfile.append (particlepath+'Particles_0'+str(limits[1])+'/Output/particle_output_'+tracestr[i]+'.txt')
			ratesfile.append (particlepath+'Particles_0'+str(limits[1])+'/Output/particle_rates_'+tracestr[i]+'.txt')
			logfile.append (particlepath+'Particles_0'+str(limits[1])+'/Output/particle_log_'+tracestr[i]+'.txt')
			physfile.append (particlepath+'Particles_0'+str(limits[1])+'/Output/particle_phys_'+tracestr[i])
				
		if (limits[1] >= 10000): 
			infile.append (particlepath+'Particles_'+str(limits[1])+'/Input/particle_trace_'+tracestr[i]+'.txt')
			outfile.append (particlepath+'Particles_'+str(limits[1])+'/Output/particle_output_'+tracestr[i]+'.txt')
			ratesfile.append (particlepath+'Particles_'+str(limits[1])+'/Output/particle_rates_'+tracestr[i]+'.txt')
			logfile.append (particlepath+'Particles_'+str(limits[1])+'/Output/particle_log_'+tracestr[i]+'.txt')	
			physfile.append (particlepath+'Particles_'+str(limits[1])+'/Output/particle_phys_'+tracestr[i])	

# generate the file_parameters file

fpfile = open(parafile, "w")

fpfile.write("! FILE PARAMETERS FOR ENVELOPE MODEL)\n")
fpfile.write("! Physical conditions input file\n")      
fpfile.write("./model_input_"+str(limits[1])+".dat\n")
fpfile.write("! Reaction file\n")
fpfile.write("./rate12_complex.rates\n")
fpfile.write("! Species file\n")
fpfile.write("./rate12_complex_atomic.specs\n")
fpfile.write("! Binding energies file\n")
fpfile.write("./rate12_binding.dat\n")
fpfile.write("! Reaction parameters file\n")
fpfile.write("./reaction_parameters.txt\n")
fpfile.write("! Grain parameters file\n")
fpfile.write("./grain_parameters.txt\n")
fpfile.write("! Radiation field parameters file\n")
fpfile.write("./radiation_parameters.txt\n")
fpfile.write("! Output abundances file\n")
fpfile.write("./model_output_"+str(limits[1])+".dat\n")
fpfile.write("! Output rates file\n")
fpfile.write("./model_rates_"+str(limits[1])+".dat\n")
fpfile.write("! Call analyse subroutine (T(RUE) or F(ALSE))?\n")
fpfile.write("F\n")
fpfile.write("! Grid point to run analyse\n")
fpfile.write("0\n")
fpfile.write("! Analyse file\n")
fpfile.write("./model_analyse_"+str(limits[1])+".dat\n")

fpfile.close()
						
for i in range(ntrace):
	
	print ("Running chemistry on "+str(i+1)+"/"+str(ntrace)+" particle traces from "+str(limits[0]+1)+" to "+str(limits[1]-1)+" ...")
	
# Run models in sequence	
	
	print (" ")
	print ("	> cp "+infile[i]+" ./model_input_"+str(limits[1])+".dat")
	print (" ")

	os.system("cp "+infile[i]+" ./model_input_"+str(limits[1])+".dat")

	print ("	> time ./model file_parameters_"+str(limits[1])+".txt > "+logfile[i])

	os.system("time ./model file_parameters_"+str(limits[1])+".txt > "+logfile[i])

	print (" ")
	print ("	 ... chemistry done ...")
	print (" ")
	print ("	> cp ./model_output_"+str(limits[1])+".dat "+outfile[i])
	
	os.system("cp ./model_output_"+str(limits[1])+".dat "+outfile[i])

	print ("	> cp ./model_rates_"+str(limits[1])+".dat "+ratesfile[i])

	os.system("cp ./model_rates_"+str(limits[1])+".dat "+ratesfile[i])

# Create and run the gnuplot file for plotting the physical structure of each trace

	print (" ")
	print (" 	... creating Gnuplot file ...")
	print (" ")
			
	gpfile = open(physfile[i]+'.gp', "w")
	
	gpfile.write("set term pdfcairo color enhance font 'Helvetica, 12' \n")
	gpfile.write("set output '"+physfile[i]+".pdf' \n")
	gpfile.write("\n")
	gpfile.write("myr2sec = 60*60*24*365*1e6 \n")
	gpfile.write("\n")
	gpfile.write("col1 = '#472c7a' \n")
	gpfile.write("col2 = '#2c718e' \n")
	gpfile.write("col3 = '#27ad81' \n")
	gpfile.write("\n")
	gpfile.write("set multiplot \n")
	gpfile.write("\n")
	gpfile.write("set size 0.8 \n") 
	gpfile.write("set size ratio 0.67 \n")
	gpfile.write("set origin 0.1, 0.1 \n")
	gpfile.write("\n")
	gpfile.write("set xlabel 'Time (Myr)' \n")
	gpfile.write("set xrange [0:6] \n")
	gpfile.write("set mxtics 10 \n")
	gpfile.write("set xtics \n")
	gpfile.write("\n")
	gpfile.write("set ylabel 'Number density (cm^{-3})' offset -0.5 textcolor rgb col1 \n")
	gpfile.write("set logscale y \n")
	gpfile.write("set yrange [1e2:1e7] \n")
	gpfile.write("set format y '10^{%T}' \n")
	gpfile.write("set ytics textcolor rgb col1 \n")
	gpfile.write("\n")
	gpfile.write("set y2label 'Temperature (K)' textcolor rgb col2 offset -2 \n")
	gpfile.write("set my2tics 10 \n")
	gpfile.write("set y2tics textcolor rgb col2 \n")
	gpfile.write("set ytics nomirror \n")
	gpfile.write("\n")
	gpfile.write("set label 'Density' at 4.8, 2e4 textcolor rgb col1 \n")
	gpfile.write("set label 'Temperature' at 2.5, 1e6 textcolor rgb col2 \n")
	gpfile.write("set label 'A_{V}' at 1.0, 1e4 textcolor rgb col3 \n")
	gpfile.write("\n")
	gpfile.write("plot 'model_input.dat' u ($10/myr2sec):($4) w lines lw 2 lc rgb col1 axes x1y1 t '',  \\\n")
	gpfile.write("     'model_input.dat' u ($10/myr2sec):($5) w lines lw 2 lc rgb col2 axes x1y2 t '' \n")
	gpfile.write("\n")
	gpfile.write("# Plot Av using x1y1 with an offset y axis and same scale \n")
	gpfile.write("\n")
	gpfile.write("unset border \n")
	gpfile.write("\n")
	gpfile.write("unset label \n")
	gpfile.write("unset ytics \n")
	gpfile.write("unset y2tics \n")
	gpfile.write("unset y2label \n")
	gpfile.write("\n")
	gpfile.write("set size 0.72 \n")
	gpfile.write("set size ratio 0.67 \n")
	gpfile.write("set origin 0.090, 0.181 \n")
	gpfile.write("\n")
	gpfile.write("unset xlabel \n")
	gpfile.write("unset xtics \n")
	gpfile.write("\n")
	gpfile.write("set ytic scale 0 \n")
	gpfile.write("set ylabel 'Visual extinction (mag)' offset -6 textcolor rgb col3 \n")
	gpfile.write("set yrange[1e-2:1e3] \n")
	gpfile.write("set logscale y \n")
	gpfile.write("set format y '10^{%T}' \n") 
	gpfile.write("set ytics offset -6 textcolor rgb col3 \n")
	gpfile.write("\n")
	gpfile.write("plot 'model_input.dat' u ($10/myr2sec):($7) w lines lw 2 lc rgb col3 axes x1y1 t '' \n")
	gpfile.write("\n")
	gpfile.write("unset multiplot \n")
	gpfile.write("\n")

	print (" ... DONE!")
	print (" ")
	print "******************************************"
	print (" ")


