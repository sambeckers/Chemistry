      program scaling
      
      implicit none
      
      integer i,j
      integer d,e
      
      parameter (d=100)
      parameter (e=100)
      
      integer nh2, nn2
            
      real h2col(d),n2col(d)      
      real f100(d,d),f1000(d,d),fco(d,d)
      real scale(d,d)
      real dummy
      
      open(unit=1,file='N2_shielding_100K_11.2b.dat')
      
      read(1,*)
      read(1,*)
      read(1,*)      
      read(1,'(24X,I3)') nn2
      read(1,'(24X,I3)') nh2      
      read(1,*)
      
      read(1,'(11X,50(1PE10.3))') (n2col(i),i=1,nn2)
      
      i=1
      
100   read(1,*,end=101) h2col(i),(f100(i,j),j=1,nn2)      
      i=i+1      
      go to 100      
101   continue
      
      close(unit=1)
      
      open(unit=1,file='N2_shielding_1000K_11.2b.dat')
      
      do i=1,7
         read(1,*)
      end do  
      
      i=1
      
102   read(1,*,end=103) dummy,(f1000(i,j),j=1,nn2)      
      i=i+1      
      go to 102      
103   continue
      
      close(unit=1)
            
      do i=1,nh2

         do j=1,nn2      
            scale(i,j) = f1000(i,j)/f100(i,j)               
         end do
         
      end do
      
      open(unit=1,
     *   file='../CO_Shielding/CO_shielding_100K_11.2b.dat')
     
      do i=1,9
         read(1,*)
      end do
      
      i=1
      
104   read(1,*,end=105) dummy,(fco(i,j),j=1,nn2)
      i=i+1      
      go to 104      
105   continue
      
      close(unit=1)
      
      do i=1,nh2
      
         do j=1,nn2
      
         scale(i,j) = scale(i,j)*fco(i,j)
      
         end do

         write(8,'(11X,50(1PE10.3))') (scale(i,j),j=1,nn2) 

      end do
      
      
      
      end program scaling
