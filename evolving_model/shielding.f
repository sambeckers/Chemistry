      SUBROUTINE SHIELDING(SPEC,N_H2,N_CO,N_N2,TEMP,SHIELD)
      
      IMPLICIT NONE
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Declaration of variables
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
      INTEGER I,J,K
      INTEGER D,E
      INTEGER NH2_H2,NH2_CO,NH2_N2,NX_CO,NX_N2,NT
      
      PARAMETER(D=110)
      PARAMETER(E=5)
      
      DOUBLE PRECISION TEMP
      DOUBLE PRECISION N_H2,N_CO,N_N2
      DOUBLE PRECISION H2_COL_H2(D)
      DOUBLE PRECISION H2_COL_CO(D),CO_COL(D)
      DOUBLE PRECISION H2_COL_N2(D),N2_COL(D)
      DOUBLE PRECISION TRANGE(E)
      DOUBLE PRECISION THETA_H2(D,D,1)
      DOUBLE PRECISION THETA_CO(D,D,E)
      DOUBLE PRECISION THETA_N2(D,D,E)
      DOUBLE PRECISION SHIELD
      
      INTEGER IL,IU,JL,JU,KL,KU
      DOUBLE PRECISION X0,X1,Y0,Y1,Z0,Z1,MX,MY,MZ
      
      DOUBLE PRECISION DUMMY
      
      CHARACTER*8 SPEC

C Global storage to avoid repeated reopening/reading of files
      LOGICAL H2_READ, CO_READ, N2_READ
      DATA H2_READ/.FALSE./, CO_READ/.FALSE./, N2_READ/.FALSE./
      COMMON /H2SPEC/ H2_COL_H2, THETA_H2, NH2_H2
      COMMON /COSPEC/ H2_COL_CO, CO_COL, THETA_CO, NH2_CO, NX_CO
      COMMON /N2SPEC/ H2_COL_N2, N2_COL, THETA_N2, NH2_N2, NX_N2
                  
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     H2
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
C If H2, only one file necessary
C Using Lee et al. 1996 tabulated values for b = 3.0 km/s
      
      IF(SPEC.EQ.'H2') THEN
      
      IF (.NOT. H2_READ) THEN
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/H2_Shielding/H2_shielding.dat')
      
         READ(1,*)
      
         I = 1 
         NH2_H2 = 0
      
 100     READ(1,*,END=101) H2_COL_H2(I), THETA_H2(I,1,1)
      
         I = I + 1 
         NH2_H2 = NH2_H2 + 1
      
         GO TO 100

 101     CONTINUE
 
         CLOSE(UNIT=1)
         H2_READ = .TRUE.
      ENDIF
      
C Interpolate to find H2 shielding factor 
      
      DO I=1,NH2_H2-1
            
         IF ((N_H2.GE.H2_COL_H2(I)).AND.
     *       (N_H2.LE.H2_COL_H2(I+1))) THEN
        
         SHIELD = LOG10(THETA_H2(I,1,1)) + 
     *      LOG10(THETA_H2(I+1,1,1)/THETA_H2(I,1,1))*
     *      LOG10(N_H2/H2_COL_H2(I))/
     *      LOG10(H2_COL_H2(I+1)/H2_COL_H2(I))
     
         SHIELD = 10**(SHIELD)
                         
        END IF
      
      END DO
            
C Extrapolate if column density is higher than upper bound
      
      IF(N_H2.GE.H2_COL_H2(NH2_H2)) THEN 
      
         SHIELD = LOG10(THETA_H2(NH2_H2-1,1,1)) + 
     *      LOG10(THETA_H2(NH2_H2,1,1)/THETA_H2(NH2_H2-1,1,1))*
     *      LOG10(N_H2-H2_COL_H2(NH2_H2-1))/
     *      LOG10(H2_COL_H2(NH2_H2)/H2_COL_H2(NH2_H2-1))
              
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

      IF (.NOT. CO_READ) THEN

         K = 1
      
C CO shielding factors at 5 K

         OPEN(UNIT=1,
     *    FILE='Self_Shielding/CO_Shielding/CO_shielding_5K_11.2b.dat')
     
         DO I=1,3
            READ(1,*)
         END DO
      
         READ(1,'(24X,I3)') NX_CO
         READ(1,'(24X,I3)') NH2_CO
         READ(1,*)
      
         READ(1,'(11X,50(1PE10.3))') (CO_COL(I),I=1,NX_CO)
      
         I = 1
      
200      READ(1,*,END=201) H2_COL_CO(I),(THETA_CO(I,J,K),J=1,NX_CO)
      
         I = I + 1
      
         GO TO 200
      
201      CONTINUE
      
         CLOSE(UNIT=1)
      
         K = K + 1

C CO shielding factors at 20 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/CO_Shielding/'//
     *             'CO_shielding_20K_11.2b.dat')

         DO I=1,7
            READ(1,*) 
         END DO
            
         I = 1
      
202      READ(1,*,END=203) DUMMY,(THETA_CO(I,J,K),J=1,NX_CO)
      
         I = I + 1
      
         GO TO 202
      
203      CONTINUE
      
         CLOSE(UNIT=1)
      
         K = K + 1

C CO shielding factors at 50 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/CO_Shielding/'//
     *             'CO_shielding_50K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
204      READ(1,*,END=205) DUMMY,(THETA_CO(I,J,K),J=1,NX_CO)
      
         I = I + 1
      
         GO TO 204
      
205      CONTINUE
      
         CLOSE(UNIT=1)

         K = K + 1

C CO shielding factors at 100 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/CO_Shielding/'//
     *             'CO_shielding_100K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
206      READ(1,*,END=207) DUMMY,(THETA_CO(I,J,K),J=1,NX_CO)
      
         I = I + 1
      
         GO TO 206
      
207      CONTINUE
      
         CLOSE(UNIT=1)

         K = K + 1

C CO shielding factors at 1000 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/CO_Shielding/'//
     *             'CO_shielding_1000K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
208      READ(1,*,END=209) DUMMY,(THETA_CO(I,J,K),J=1,NX_CO)
      
         I = I + 1
      
         GO TO 208
      
209      CONTINUE
      
         CLOSE(UNIT=1)

         CO_READ = .TRUE.

      ENDIF
                  
C Interpolate/Extrapolate to find CO shielding factors 

      DO I=1,NH2_CO-1            
         IF ((N_H2.GE.H2_COL_CO(I)).AND.
     *       (N_H2.LT.H2_COL_CO(I+1))) THEN      
            X0 = H2_COL_CO(I)
            X1 = H2_COL_CO(I+1)
            IL = I
            IU = I+1      
         END IF
      END DO
      
      IF(N_H2.GE.H2_COL_CO(NH2_CO)) THEN
         X0 = H2_COL_CO(NH2_CO-1)
         X1 = H2_COL_CO(NH2_CO)
         IL = NH2_CO-1
         IU = NH2_CO      
      ELSE IF(N_H2.LT.H2_COL_CO(1)) THEN
         X0 = H2_COL_CO(1)
         X1 = H2_COL_CO(2)
         IL = 1
         IU = 2      
      END IF
            
      DO I=1,NX_CO-1
         IF ((N_CO.GE.CO_COL(I)).AND.(N_CO.LT.CO_COL(I+1))) THEN      
            Y0 = CO_COL(I)
            Y1 = CO_COL(I+1)      
            JL = I
            JU = I+1      
         END IF
      END DO

      IF(N_CO.GE.CO_COL(NX_CO)) THEN
         Y0 = CO_COL(NX_CO-1)
         Y1 = CO_COL(NX_CO)
         JL = NX_CO-1
         JU = NX_CO      
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

      IF(TEMP.GE.TRANGE(NT)) THEN
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

      MX = LOG10(N_H2/H2_COL_CO(IL))/
     *     LOG10(H2_COL_CO(IU)/H2_COL_CO(IL))
      MY = LOG10(N_CO/CO_COL(JL))/LOG10(CO_COL(JU)/CO_COL(JL))
      MZ = LOG10(TEMP/TRANGE(KL))/LOG10(TRANGE(KU)/TRANGE(KL))      
      
      SHIELD = ((LOG10(THETA_CO(IL,JL,KL))*(1-MX) + 
     *    LOG10(THETA_CO(IU,JL,KL))*MX)*(1-MY) + 
     *   (LOG10(THETA_CO(IL,JU,KL))*(1-MX) + 
     *    LOG10(THETA_CO(IU,JU,KL))*MX)*MY)*(1-MZ) + 
     *   ((LOG10(THETA_CO(IL,JL,KU))*(1-MX) + 
     *    LOG10(THETA_CO(IU,JL,KU))*MX)*(1-MY) + 
     *   (LOG10(THETA_CO(IL,JU,KU))*(1-MX) + 
     *    LOG10(THETA_CO(IU,JU,KU))*MX)*MY)*MZ       

      SHIELD = 10**SHIELD
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     N2
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      ELSE IF(SPEC.EQ.'N2') THEN

      NT = 5

      TRANGE(1) =   10.0
      TRANGE(2) =   30.0
      TRANGE(3) =   50.0
      TRANGE(4) =  100.0
      TRANGE(5) = 1000.0

      IF (.NOT. N2_READ) THEN

         K = 1
      
C N2 shielding factors at 10 K

         OPEN(UNIT=1,
     *        FILE='Self_Shielding/N2_Shielding/'//
     *             'N2_shielding_10K_11.2b.dat')
     
         DO I=1,3
            READ(1,*)
         END DO
      
         READ(1,'(24X,I3)') NX_N2
         READ(1,'(24X,I3)') NH2_N2
         READ(1,*)
      
         READ(1,'(11X,50(1PE10.3))') (N2_COL(I),I=1,NX_N2)
      
         I = 1
      
300      READ(1,*,END=301) H2_COL_N2(I),(THETA_N2(I,J,K),J=1,NX_N2)
      
         I = I + 1
      
         GO TO 300
      
301      CONTINUE
      
         CLOSE(UNIT=1)
      
         K = K + 1

C N2 shielding factors at 30 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/N2_Shielding/'//
     *             'N2_shielding_30K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
302      READ(1,*,END=303) DUMMY,(THETA_N2(I,J,K),J=1,NX_N2)
      
         I = I + 1
      
         GO TO 302
      
303      CONTINUE
      
         CLOSE(UNIT=1)

         K = K + 1

C N2 shielding factors at 50 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/N2_Shielding/'//
     *             'N2_shielding_50K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
304      READ(1,*,END=305) DUMMY,(THETA_N2(I,J,K),J=1,NX_N2)
      
         I = I + 1
      
         GO TO 304
      
305      CONTINUE
      
         CLOSE(UNIT=1)

         K = K + 1

C N2 shielding factors at 100 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/N2_Shielding/'//
     *             'N2_shielding_100K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
306      READ(1,*,END=307) DUMMY,(THETA_N2(I,J,K),J=1,NX_N2)
      
         I = I + 1
      
         GO TO 306
      
307      CONTINUE
      
         CLOSE(UNIT=1)

         K = K + 1

C N2 shielding factors at 1000 K
      
         OPEN(UNIT=1,
     *        FILE='Self_Shielding/N2_Shielding/'//
     *             'N2_shielding_1000K_11.2b.dat')

         DO I=1,7
            READ(1,*)
         END DO
            
         I = 1
      
308      READ(1,*,END=309) DUMMY,(THETA_N2(I,J,K),J=1,NX_N2)
      
         I = I + 1
      
         GO TO 308
      
309      CONTINUE
      
         CLOSE(UNIT=1)

         N2_READ = .TRUE.

      ENDIF
                  
C Interpolate/Extrapolate to find N2 shielding factors 

      DO I=1,NH2_N2-1           
         IF ((N_H2.GE.H2_COL_N2(I)).AND.
     *       (N_H2.LT.H2_COL_N2(I+1))) THEN      
            X0 = H2_COL_N2(I)
            X1 = H2_COL_N2(I+1)
            IL = I
            IU = I+1      
         END IF         
      END DO
      
      IF(N_H2.GE.H2_COL_N2(NH2_N2)) THEN
         X0 = H2_COL_N2(NH2_N2-1)
         X1 = H2_COL_N2(NH2_N2)
         IL = NH2_N2-1
         IU = NH2_N2      
      ELSE IF(N_H2.LT.H2_COL_N2(1)) THEN
         X0 = H2_COL_N2(1)
         X1 = H2_COL_N2(2)
         IL = 1
         IU = 2      
      END IF
            
      DO I=1,NX_N2-1
         IF ((N_N2.GE.N2_COL(I)).AND.(N_N2.LT.N2_COL(I+1))) THEN      
            Y0 = N2_COL(I)
            Y1 = N2_COL(I+1)      
            JL = I
            JU = I+1      
         END IF
      END DO

      IF(N_N2.GE.N2_COL(NX_N2)) THEN
         Y0 = N2_COL(NX_N2-1)
         Y1 = N2_COL(NX_N2)
         JL = NX_N2-1
         JU = NX_N2      
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

      IF(TEMP.GE.TRANGE(NT)) THEN
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
      
      MX = LOG10(N_H2/H2_COL_N2(IL))/
     *     LOG10(H2_COL_N2(IU)/H2_COL_N2(IL))
      MY = LOG10(N_N2/N2_COL(JL))/LOG10(N2_COL(JU)/N2_COL(JL))
      MZ = LOG10(TEMP/TRANGE(KL))/LOG10(TRANGE(KU)/TRANGE(KL))      
      
      SHIELD = ((LOG10(THETA_N2(IL,JL,KL))*(1-MX) + 
     *    LOG10(THETA_N2(IU,JL,KL))*MX)*(1-MY) + 
     *   (LOG10(THETA_N2(IL,JU,KL))*(1-MX) + 
     *    LOG10(THETA_N2(IU,JU,KL))*MX)*MY)*(1-MZ) + 
     *   ((LOG10(THETA_N2(IL,JL,KU))*(1-MX) + 
     *    LOG10(THETA_N2(IU,JL,KU))*MX)*(1-MY) + 
     *   (LOG10(THETA_N2(IL,JU,KU))*(1-MX) + 
     *    LOG10(THETA_N2(IU,JU,KU))*MX)*MY)*MZ       

      SHIELD = 10**SHIELD
      
      END IF
      
      RETURN
      
      END