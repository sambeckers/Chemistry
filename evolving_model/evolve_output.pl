#!/usr/bin/perl

# Author: Catherine Walsh 08/2013
# Adapted from graph4.pl by David Tideswell et al for collapse model
# Program to extact abundances of a selected species from ouput
# Abundances returned as a function of space and time
# Produces GNUPLOT friendly data and GNUPLOT script
# Produces CARTESIAN r-z axes graph

# Adapted in 2019 by C Walsh to take in 3D data and multiple species

use POSIX;

# Check for correct number of command line arguments @ARGV

if (@ARGV < 3)
{
die("USAGE: evolve_output.pl <model output> <output file> <species list>  
e.g. evolve_output.pl output_model.dat abundances_file CO HCO+ HCN CN ... \n");
}

# Set variable $infile to first command line argument - file to extract data from
# Set variable $outfile to second command line argument - file to write data to

chomp(my $infile = shift @ARGV);
chomp(my $outfile = shift @ARGV);

$outfile = $outfile. ".dat";

# chomp - removes trailing strings such as newline \n
# shift - shifts off first value of an array and decreases array by one
# The dot "." after $outfile is the string concatenation operator

# Cycle through remaining arguments and put into a species array
# $smax counts number of species

my @species;
my $smax = 0;
my $s;

while(<@ARGV>)
{
   	push @species,$_;
        $smax = $smax + 1;
}

$smax = $smax - 1;

# Open files for reading and writing

open(FILE1,">",$outfile);
open(FILE3,"<",$infile) or die("Unable to open file - check filename!");

# open(FILEHANDLE,MODE,EXPRESSION)
# "<" indicates file is opened for reading and ">" indicates file is opened for writing

# Declare names of remaining variables (scalar and array) before use
# Initialise if necessary

my @xcoord;
my @ycoord;
my @zcoord;
my @density;
my @temperature;
my @guv;
my @av;
my @time;
my @abundance;

my $i;
my $imax = 0;
my $ab;
my %species_warned;

# Read input file using a while loop - close when finished reading

while (<FILE3>)

{

# Remove all newlines from the file

	chomp;
	
# Read each line and split the line into strings and store in an array

	my @line = split /\s+/, $_;
		
# split /PATTERN/,EXPRESSION,LIMIT
# $_ is the default input and pattern matching space
# Pattern / / splits on white space - leading white space produces a null first field

# If the first element of the line equals "RADIUS" and $switch is set to 0

        if ($line[1] eq "X")
		
        	{
	
# eq - string binary equals operator
# push(array,list) - adds values of list onto end of array
	
			push @xcoord,$line[4];
		
# Add one onto counter to count number of points

			$imax = $imax + 1;				
		}
 		
	if ($line[1] eq "Y")	
		{
			push @ycoord,$line[4];		
		}                
			
	if ($line[1] eq "Z")	
		{
			push @zcoord,$line[4];		
		}                
			
	if ($line[1] eq "DENSITY")	
		{
			push @density,$line[3];		
		}                
		
	if ($line[1] eq "GAS")	
		{
			push @temperature,$line[4];		
		}                
			
	if ($line[1] eq "UV")	
		{
			push @guv,$line[4];
		}                

	if ($line[1] eq "VISUAL")	
		{
			push @av,$line[4];		
		}                

	if ($line[1] eq "TIME")	
		{
			push @time,$line[2];		
		}                

# Loop over species array

        for $s (0..$smax)
        	{
        	
                	if ($line[1] eq $species[$s])
                	{
                        	$abundance[$s][$imax-1] = $line[2];
					
# Check for existence of species in input file
# Check if @abundance value is empty

				if (!defined($abundance[$s][$imax-1]) || $abundance[$s][$imax-1] == 0)
				{
					unless ($species_warned{$species[$s]}) {
						warn("Warning: Species $species[$s] has zero or undefined abundance - continuing with 0.\n");
						$species_warned{$species[$s]} = 1;
					}
					$abundance[$s][$imax-1] = 0;
				}

                	}
                }

}

$imax = $imax - 1;

# Ensure any species not found in the output file are set to 0 for all grid points
for $s (0..$smax) {
    for $i (0..$imax) {
        $abundance[$s][$i] = 0 unless defined($abundance[$s][$i]);
    }
    unless (defined($abundance[$s][0])) {
        warn("Warning: Species $species[$s] was not found in the model output - all abundances set to 0.\n");
    }
}

# Now print output for GNUPLOT data
# Print to FILE1 - the .dat file

print FILE1 "# Collapse Model Output \n \n";
print FILE1 "# $outfile in GNUPLOT format produced by $0 \n \n";

# print FILE1 "#X               Y               Z              DENSITY         TEMP            AV              TIME            ";
	
# 	for $s (0..$smax)
# 	{		
# 		$species[$s] = sprintf("%-12s",$species[$s]);
# 		$count = $s+1;
# 		$count = sprintf("%-2i",$count);
# 		print FILE1 "$count $species[$s]";
		
# 	}

# Adapted for np.genfromtxt (no # before header & no count for species)
print FILE1 "X               Y               Z              DENSITY         TEMP            AV              TIME            ";
	
	for $s (0..$smax)
	{		
		$species[$s] = sprintf("%-12s",$species[$s]);
		$count = $s+1;
		$count = sprintf("%-2i",$count);
		print FILE1 "$species[$s]";
		
	}

print FILE1 "\n";
	
print FILE1 "#(PC)            (PC)            (PC)           (CM-3)          (K)             (MAG)           (YR)	       (wrt Hnuc)     \n \n";

# Print data to file
# Format data for printing

my $au2cm = 1.49597871e+13;
my $se2yr = 3.16888e-8;

for $i (0..$imax)
{

$xcoord[$i]      = sprintf("%-15.5e",$xcoord[$i]);     
$ycoord[$i]      = sprintf("%-15.5e",$ycoord[$i]);     
$zcoord[$i]      = sprintf("%-15.5e",$zcoord[$i]);     
$density[$i]     = sprintf("%-15.5e",$density[$i]);    
$temperature[$i] = sprintf("%-15.5e",$temperature[$i]);
$guv[$i]         = sprintf("%-15.5e",$guv[$i]);        
$av[$i]          = sprintf("%-15.5e",$av[$i]);         
$time[$i]        = sprintf("%-20.12e",$time[$i]);

print FILE1 "$xcoord[$i] $ycoord[$i] $zcoord[$i] $density[$i] $temperature[$i] $av[$i] $time[$i] ";
	
	for $s (0..$smax)
        {	                        
		$abundance[$s][$i] = sprintf('%-15.3e',$abundance[$s][$i]);		
		print FILE1 "$abundance[$s][$i]";
        }

print FILE1 "\n";

}

close FILE1;
