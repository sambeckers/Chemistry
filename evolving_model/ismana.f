
      SUBROUTINE ANALYSE(NSPEC,SPEC,NREAC,Y,X,K,TOUT,
     *   RE1,RE2,RE3,PR1,PR2,PR3,PR4,ANAFILE)

C NSPEC = number of species
C NREAC = number of reactions
C Y = absolute abundances of species 
C K = reaction rates
C DENS = particle number density
C TOUT = time at which ANALYSE is called
C ANAFILE = output filename

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Declaration of variables
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      
      INTEGER E,F,G
      
      DOUBLE PRECISION K
      
      PARAMETER(E=10000)
      PARAMETER(F=1000)
      PARAMETER(G=150)
      
      CHARACTER*10 SPEC(F),RE1(E),RE2(E),RE3(E),PR1(E),
     *             PR2(E),PR3(E),PR4(E),
     *             RD1(E),RD2(E),PD1(E),PD2(E),
     *             RP1(E),RP2(E),PP1(E),PP2(E)
      
      DIMENSION D(E),P(E),NDR(E),NPR(E),Y(F),X(1),K(E)
      
      CHARACTER*70 ANAFILE

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Open input and output files
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      OPEN(UNIT=11, file=ANAFILE)
                  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Initial output
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      TYEARS = TOUT*3.1709791E-08

      WRITE(11,2) TYEARS

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Begin analysis of each species
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
            
      DO I=1,NSPEC
      
      	DTOT = 0.0
      	PTOT = 0.0
        
      	WRITE(11,3) I, SPEC(I)

      	ID = 0
      	IP = 0        

      	DO J=1,NREAC
                
                DO M=1,NSPEC                           		
                   IF(SPEC(M).EQ.RE1(J)) Y1=Y(M)
                   IF(SPEC(M).EQ.RE2(J)) Y2=Y(M)
 		END DO
                
                IF(RE2(J).EQ.'e-') Y2=X(1)
                
         	RMULT = 0.0
                
         	IF((RE1(J).EQ.SPEC(I)).OR.(RE2(J).EQ.SPEC(I))) THEN
            	
                	IF(RE1(J).EQ.SPEC(I)) RMULT = 1.0
            	
                	IF(RE2(J).EQ.SPEC(I)) RMULT = RMULT + 1.0
            	
                	IF(PR1(J).EQ.SPEC(I)) RMULT = RMULT - 1.0
            	
                	IF(PR2(J).EQ.SPEC(I)) RMULT = RMULT - 1.0
            	
                	IF(PR3(J).EQ.SPEC(I)) RMULT = RMULT - 1.0
                
			IF(PR4(J).EQ.SPEC(I)) RMULT = RMULT - 1.0
            
            		ID = ID + 1
            
            		IF((RE2(J).EQ.'CRP').OR.(RE2(J).EQ.'PHOTON').OR.
     *           	(RE2(J).EQ.'CRPHOT').OR.(RE2(J).EQ.'GRAIN').OR.
     *                  (RE2(J).EQ.'XRAY').OR.(PR2(J).EQ.'GRAIN')) THEN               		
                        	D(ID) = K(J)*Y1*RMULT  
            		ELSE           
                        	D(ID) = K(J)*Y1*Y2*RMULT 
                        END IF
                        
            		DTOT = DTOT + D(ID)
           		
                        NDR(ID) = J
            		RD1(ID) = RE1(J)
            		RD2(ID) = RE2(J)
            		PD1(ID) = PR1(J)
            		PD2(ID) = PR2(J)
                        
         	END IF
                
                RMULT = 0.0
                
         	IF((PR1(J).EQ.SPEC(I)).OR.(PR2(J).EQ.SPEC(I)).OR.
     *        		(PR3(J).EQ.SPEC(I))) THEN
            
            		IF(PR1(J).EQ.SPEC(I)) RMULT = RMULT + 1.0
                        
            		IF(PR2(J).EQ.SPEC(I)) RMULT = RMULT + 1.0
            		
                        IF(PR3(J).EQ.SPEC(I)) RMULT = RMULT + 1.0
			
                        IF(PR4(J).EQ.SPEC(I)) RMULT = RMULT + 1.0
            		
                        IF(RE1(J).EQ.SPEC(I)) RMULT = RMULT - 1.0
            		
                        IF(RE2(J).EQ.SPEC(I)) RMULT = RMULT - 1.0
            		
                        IP = IP + 1
                        
            		IF((RE2(J).EQ.'PHOTON').OR.(RE2(J).EQ.'CRP').OR.
     *           	(RE2(J).EQ.'CRPHOT').OR.(RE2(J).EQ.'GRAIN').OR.
     *                  (RE2(J).EQ.'XRAY').OR.(PR2(J).EQ.'GRAIN')) THEN               			
                                P(IP) = K(J)*Y1*RMULT
            		ELSE
               			P(IP) = K(J)*Y1*Y2*RMULT
            		END IF
                        
            		PTOT = PTOT + P(IP)
            	
                	NPR(IP) = J
            		RP1(IP) = RE1(J)
            		RP2(IP) = RE2(J)
            		PP1(IP) = PR1(J)
            		PP2(IP) = PR2(J)
         	
                END IF
         
    	END DO
                
	DO IL = 1, ID
         
         	DPC = 100*(D(IL)/DTOT)
         
         	IF(DPC.GE.1.0) THEN
      			WRITE(11,4) NDR(IL),RD1(IL),RD2(IL),PD1(IL),
     *                  PD2(IL),-NINT(DPC)
         	END IF
      	
	END DO
      
      	DO IL = 1, IP
      
         	PPC = 100*(P(IL)/PTOT)
      
         	IF(PPC.GE.1.0) THEN
      		 	WRITE(11,4) NPR(IL),RP1(IL),RP2(IL),PP1(IL),
     *                  PP2(IL),NINT(PPC)
         	END IF
         
      	END DO
        
        RNET = PTOT-DTOT
      
      	WRITE(11,5) DTOT, PTOT, RNET
        
      END DO
      
      CLOSE (UNIT=11)

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Format statements
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
  
  1   FORMAT (I4,5(1X,A8),2(1X,A4))

  2   FORMAT(1X,'MAIN FORMATION AND DESTRUCTION PROCESSES AT TIME  = ',
     *    1PE11.3,2X,'YEARS')

c   3   FORMAT(10X,I3,4X,A8,10X)
  3   FORMAT(10X, I3, 4X, '* ', A8, 10X)
  4   FORMAT(2X,I4,4(1X,A8),I4,'%')

  5   FORMAT(10X,'DRATE= ',E9.3,5X,'PRATE= ',E9.3,/
     *    10X,'NETRATE = ',E10.3/)

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCC     RETURN    CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      RETURN
      
      END SUBROUTINE ANALYSE
