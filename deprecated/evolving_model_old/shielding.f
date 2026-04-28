
      SUBROUTINE SHIELDING(SPEC,N_H2,N_CO,N_N2,TEMP,SHIELD)
      
      IMPLICIT NONE
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Declaration of variables
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      INTEGER I,J,K
      INTEGER D,E
      INTEGER NH2,NX,NT
      
      PARAMETER(D=110)
      PARAMETER(E=5)
      
      DOUBLE PRECISION TEMP
      DOUBLE PRECISION N_H2,N_CO,N_N2
      DOUBLE PRECISION H2_COL(D),CO_COL(D),N2_COL(D)
      DOUBLE PRECISION TRANGE(E),THETA(D,D,E)
      DOUBLE PRECISION SHIELD
      
      INTEGER IL,IU,JL,JU,KL,KU
      DOUBLE PRECISION X0,X1,Y0,Y1,Z0,Z1,MX,MY,MZ
      
      DOUBLE PRECISION DUMMY
      
      CHARACTER*8 SPEC
                  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     H2
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
C If H2, only one file necessary
C Using Lee et al. 1996 tabulated values for b = 3.0 km/s
      
      IF(SPEC.EQ.'H2') THEN
      
      OPEN(UNIT=1,FILE='Self_Shielding/H2_Shielding/H2_shielding.dat')
      
      READ(1,*)
      
      I = 1 
      NH2 = 0
      
 100  READ(1,*,END=101) H2_COL(I), THETA(I,1,1)
      
      I = I + 1 
      NH2 = NH2 + 1
      
      GO TO 100

 101  CONTINUE
 
      CLOSE(UNIT=1)
      
C Interpolate to find H2 shielding factor 
      
      DO I=1,NH2-1
            
         IF ((N_H2.GE.H2_COL(I)).AND.(N_H2.LE.H2_COL(I+1))) THEN
        
         SHIELD = LOG10(THETA(I,1,1)) + 
     *      LOG10(THETA(I+1,1,1)/THETA(I,1,1))*
     *      LOG10(N_H2/H2_COL(I))/LOG10(H2_COL(I+1)/H2_COL(I))
     
         SHIELD = 10**(SHIELD)
                         
        END IF
      
      END DO
            
C Extropolate if column density is higher than upper bound
      
      IF(N_H2.GT.H2_COL(NH2)) THEN 
      
         SHIELD = LOG10(THETA(NH2-1,1,1)) + 
     *      LOG10(THETA(NH2,1,1)/THETA(NH2-1,1,1))*
     *      LOG10(N_H2-H2_COL(NH2-1))/
     *      LOG10(H2_COL(NH2)/H2_COL(NH2-1))
              
         SHIELD = 10**(SHIELD)

      END IF
                  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     CO
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C For CO, shielding is a function of N(H2), N(CO), and temperature

      ELSE IF(SPEC.EQ.'CO') THEN
            
C Manually set temperature ranges

      NT = 5

      TRANGE(1) =    5.0
      TRANGE(2) =   20.0
      TRANGE(3) =   50.0
      TRANGE(4) =  100.0
      TRANGE(5) = 1000.0

C Counter for temperature ranges

      K = 1
      
C CO shielding factors at 5 K

      OPEN(UNIT=1,
     * FILE='Self_Shielding/CO_Shielding/CO_shielding_5K_11.2b.dat')
     
      DO I=1,3
         READ(1,*)
      END DO
      
      READ(1,'(24X,I3)') NX   
      READ(1,'(24X,I3)') NH2
      READ(1,*)
      
      READ(1,'(11X,50(1PE10.3))') (CO_COL(I),I=1,NX)
      
      I = 1
      
200   READ(1,*,END=201) H2_COL(I),(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 200
      
201   CONTINUE
      
      CLOSE(UNIT=1)
      
      K = K + 1

C CO shielding factors at 20 K
      
      OPEN(UNIT=1,
     * FILE='Self_Shielding/CO_Shielding/CO_shielding_20K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
202   READ(1,*,END=203) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 202
      
203   CONTINUE
      
      CLOSE(UNIT=1)

      K = K + 1

C CO shielding factors at 50 K
      
      OPEN(UNIT=1,
     * FILE='Self_Shielding/CO_Shielding/CO_shielding_50K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
204   READ(1,*,END=205) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 204
      
205   CONTINUE
      
      CLOSE(UNIT=1)

      K = K + 1

C CO shielding factors at 100 K
      
      OPEN(UNIT=1,
     * FILE='Self_Shielding/CO_Shielding/CO_shielding_100K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
206   READ(1,*,END=207) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 206
      
207   CONTINUE
      
      CLOSE(UNIT=1)

      K = K + 1

C CO shielding factors at 1000 K
      
      OPEN(UNIT=1,
     * FILE='Self_Shielding/CO_Shielding/CO_shielding_1000K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
208   READ(1,*,END=209) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 208
      
209   CONTINUE
      
      CLOSE(UNIT=1)
                  
C Interpolate/Extrapolate to find CO shielding factors 

      DO I=1,NH2-1            
         IF ((N_H2.GE.H2_COL(I)).AND.(N_H2.LT.H2_COL(I+1))) THEN      
            X0 = H2_COL(I)
            X1 = H2_COL(I+1)
            IL = I
            IU = I+1      
         END IF
      END DO
      
      IF(N_H2.GT.H2_COL(NH2)) THEN
         X0 = H2_COL(NH2-1)
         X1 = H2_COL(NH2)
         IL = NH2-1
         IU = NH2      
      ELSE IF(N_H2.LT.H2_COL(1)) THEN
         X0 = H2_COL(1)
         X1 = H2_COL(2)
         IL = 1
         IU = 2      
      END IF
            
      DO I=1,NX-1
         IF ((N_CO.GE.CO_COL(I)).AND.(N_CO.LT.CO_COL(I+1))) THEN      
            Y0 = CO_COL(I)
            Y1 = CO_COL(I+1)      
            JL = I
            JU = I+1      
         END IF
      END DO

      IF(N_CO.GT.CO_COL(NX)) THEN
         Y0 = CO_COL(NX-1)
         Y1 = CO_COL(NX)
         JL = NX-1
         JU = NX      
      ELSE IF(N_CO.LT.CO_COL(1)) THEN
         Y0 = CO_COL(1)
         Y1 = CO_COL(2)
         JL = 1
         JU = 2      
      END IF

      DO I=1,NT-1
         IF ((TEMP.GE.TRANGE(I)).AND.(TEMP.LT.TRANGE(I+1))) THEN      
            Z0 = TRANGE(I)
            Z1 = TRANGE(I+1)      
            KL = I
            KU = I+1      
         END IF
      END DO

      IF(TEMP.GT.TRANGE(NT)) THEN
         Z0 = TRANGE(NT-1)
         Z1 = TRANGE(NT)
         KL = NT-1
         KU = NT      
      ELSE IF(TEMP.LT.TRANGE(1)) THEN
         Z0 = TRANGE(1)
         Z1 = TRANGE(2)
         KL = 1
         KU = 2      
      END IF
      
C Trilinear interpolation (in log space) to determine shielding function

      MX = lOG10(N_H2/H2_COL(IL))/LOG10(H2_COL(IU)/H2_COL(IL))
      MY = LOG10(N_CO/CO_COL(JL))/LOG10(CO_COL(JU)/CO_COL(JL))
      MZ = LOG10(TEMP/TRANGE(KL))/LOG10(TRANGE(KU)/TRANGE(KL))      
      
      SHIELD = ((lOG10(THETA(IL,JL,KL))*(1-MX) + 
     *    LOG10(THETA(IU,JL,KL))*MX)*(1-MY) + 
     *   (LOG10(THETA(IL,JU,KL))*(1-MX) + 
     *    LOG10(THETA(IU,JU,KL))*MX)*MY)*(1-MZ) + 
     *   ((LOG10(THETA(IL,JL,KU))*(1-MX) + 
     *    LOG10(THETA(IU,JL,KU))*MX)*(1-MY) + 
     *   (LOG10(THETA(IL,JU,KU))*(1-MX) + 
     *    LOG10(THETA(IU,JU,KU))*MX)*MY)*MZ       

      SHIELD = 10**SHIELD
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     N2
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C For N2, shielding is a function of N(H2), N(N2), and temperature

      ELSE IF(SPEC.EQ.'N2') THEN

C Manually set temperature ranges

      NT = 5

      TRANGE(1) =   10.0
      TRANGE(2) =   30.0
      TRANGE(3) =   50.0
      TRANGE(4) =  100.0
      TRANGE(5) = 1000.0

C Counter for temperature ranges

      K = 1
      
C N2 shielding factors at 10 K

      OPEN(UNIT=1,
     * FILE='Self_Shielding/N2_Shielding/N2_shielding_10K_11.2b.dat')
     
      DO I=1,3
         READ(1,*)
      END DO
      
      READ(1,'(24X,I3)') NX   
      READ(1,'(24X,I3)') NH2
      READ(1,*)
      
      READ(1,'(11X,50(1PE10.3))') (N2_COL(I),I=1,NX)
      
      I = 1
      
300   READ(1,*,END=301) H2_COL(I),(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 300
      
301   CONTINUE
      
      CLOSE(UNIT=1)
      
      K = K + 1

C N2 shielding factors at 30 K
      
      OPEN(UNIT=1,
     * FILE='Self_Shielding/N2_Shielding/N2_shielding_30K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
302   READ(1,*,END=303) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 302
      
303   CONTINUE
      
      CLOSE(UNIT=1)

      K = K + 1

C N2 shielding factors at 50 K
      
      OPEN(UNIT=1,
     * FILE='Self_Shielding/N2_Shielding/N2_shielding_50K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
304   READ(1,*,END=305) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 304
      
305   CONTINUE
      
      CLOSE(UNIT=1)

      K = K + 1

C N2 shielding factors at 100 K
      
      OPEN(UNIT=1,
     *FILE='Self_Shielding/N2_Shielding/N2_shielding_100K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
306   READ(1,*,END=307) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 306
      
307   CONTINUE
      
      CLOSE(UNIT=1)

      K = K + 1

C N2 shielding factors at 1000 K
      
      OPEN(UNIT=1,
     *FILE='Self_Shielding/N2_Shielding/N2_shielding_1000K_11.2b.dat')

      DO I=1,7
         READ(1,*)
      END DO
            
      I = 1
      
308   READ(1,*,END=309) DUMMY,(THETA(I,J,K),J=1,NX)
      
      I = I + 1
      
      GO TO 308
      
309   CONTINUE
      
      CLOSE(UNIT=1)
                  
C Interpolate/Extrapolate to find N2 shielding factors 

      DO I=1,NH2-1           
         IF ((N_H2.GE.H2_COL(I)).AND.(N_H2.LT.H2_COL(I+1))) THEN      
            X0 = H2_COL(I)
            X1 = H2_COL(I+1)
            IL = I
            IU = I+1      
         END IF         
      END DO
      
      IF(N_H2.GT.H2_COL(NH2)) THEN
         X0 = H2_COL(NH2-1)
         X1 = H2_COL(NH2)
         IL = NH2-1
         IU = NH2      
      ELSE IF(N_H2.LT.H2_COL(1)) THEN
         X0 = H2_COL(1)
         X1 = H2_COL(2)
         IL = 1
         IU = 2      
      END IF
            
      DO I=1,NX-1
         IF ((N_N2.GE.N2_COL(I)).AND.(N_N2.LT.N2_COL(I+1))) THEN      
            Y0 = N2_COL(I)
            Y1 = N2_COL(I+1)      
            JL = I
            JU = I+1      
         END IF
      END DO

      IF(N_N2.GT.N2_COL(NX)) THEN
         Y0 = N2_COL(NX-1)
         Y1 = N2_COL(NX)
         JL = NX-1
         JU = NX      
      ELSE IF(N_N2.LT.N2_COL(1)) THEN
         Y0 = N2_COL(1)
         Y1 = N2_COL(2)
         JL = 1
         JU = 2      
      END IF

      DO I=1,NT-1
         IF ((TEMP.GE.TRANGE(I)).AND.(TEMP.LT.TRANGE(I+1))) THEN      
            Z0 = TRANGE(I)
            Z1 = TRANGE(I+1)      
            KL = I
            KU = I+1      
         END IF
      END DO

      IF(TEMP.GT.TRANGE(NT)) THEN
         Z0 = TRANGE(NT-1)
         Z1 = TRANGE(NT)
         KL = NT-1
         KU = NT      
      ELSE IF(TEMP.LT.TRANGE(1)) THEN
         Z0 = TRANGE(1)
         Z1 = TRANGE(2)
         KL = 1
         KU = 2      
      END IF
      
C Trilinear interpolation (in log space) to determine shielding function
      
      MX = lOG10(N_H2/H2_COL(IL))/LOG10(H2_COL(IU)/H2_COL(IL))
      MY = LOG10(N_N2/N2_COL(JL))/LOG10(N2_COL(JU)/N2_COL(JL))
      MZ = LOG10(TEMP/TRANGE(KL))/LOG10(TRANGE(KU)/TRANGE(KL))      
      
      SHIELD = ((lOG10(THETA(IL,JL,KL))*(1-MX) + 
     *    LOG10(THETA(IU,JL,KL))*MX)*(1-MY) + 
     *   (LOG10(THETA(IL,JU,KL))*(1-MX) + 
     *    LOG10(THETA(IU,JU,KL))*MX)*MY)*(1-MZ) + 
     *   ((LOG10(THETA(IL,JL,KU))*(1-MX) + 
     *    LOG10(THETA(IU,JL,KU))*MX)*(1-MY) + 
     *   (LOG10(THETA(IL,JU,KU))*(1-MX) + 
     *    LOG10(THETA(IU,JU,KU))*MX)*MY)*MZ       

      SHIELD = 10**SHIELD
      
      END IF
      
      RETURN
      
      END SUBROUTINE SHIELDING
