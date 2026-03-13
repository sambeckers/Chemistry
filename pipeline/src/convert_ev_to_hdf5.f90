module convert_ev_to_hdf5_mod
  implicit none
  private
  public :: append_ev_output

contains

  subroutine append_ev_output(batch_file, ev_output_file, chemistry_type, slot, particle_id)
    character(len=*), intent(in) :: batch_file
    character(len=*), intent(in) :: ev_output_file
    character(len=*), intent(in) :: chemistry_type
    integer, intent(in) :: slot
    integer(8), intent(in) :: particle_id

    write(*,'(a)') 'convert_ev_to_hdf5.f90 is a scaffold for the HDF5 Fortran implementation.'
    write(*,'(a)') 'Expected contract:'
    write(*,'(a)') '  batch_file      = persistent batch_XXXXX.h5 output'
    write(*,'(a)') '  ev_output_file  = ev_output.dat for one particle and chemistry type'
    write(*,'(a)') '  chemistry_type  = Crich or Orich'
    write(*,'(a)') '  slot            = zero-based particle slot within the batch file'
    write(*,'(a)') '  particle_id     = PHANTOM iorig identifier'
    write(*,'(a)') 'The current runnable pipeline uses scripts/append_ev_to_hdf5.py until an HDF5 Fortran toolchain is available.'
    write(*,'(a)') 'Build this program with h5fc or an equivalent HDF5-enabled Fortran wrapper once available.'
    write(*,'(a,a)') 'Batch file: ', trim(batch_file)
    write(*,'(a,a)') 'ev_output: ', trim(ev_output_file)
    write(*,'(a,a)') 'Chemistry: ', trim(chemistry_type)
    write(*,'(a,i0)') 'Slot: ', slot
    write(*,'(a,i0)') 'Particle ID: ', particle_id
  end subroutine append_ev_output

end module convert_ev_to_hdf5_mod


program convert_ev_to_hdf5
  use convert_ev_to_hdf5_mod, only: append_ev_output
  implicit none

  character(len=1024) :: batch_file
  character(len=1024) :: ev_output_file
  character(len=256)  :: chemistry_type
  character(len=64)   :: slot_arg
  character(len=64)   :: particle_arg
  integer :: slot
  integer(8) :: particle_id

  if (command_argument_count() /= 5) then
    write(*,'(a)') 'Usage: convert_ev_to_hdf5 <batch_file> <ev_output_file> <chemistry_type> <slot> <particle_id>'
    stop 1
  end if

  call get_command_argument(1, batch_file)
  call get_command_argument(2, ev_output_file)
  call get_command_argument(3, chemistry_type)
  call get_command_argument(4, slot_arg)
  call get_command_argument(5, particle_arg)

  read(slot_arg, *) slot
  read(particle_arg, *) particle_id

  call append_ev_output(trim(batch_file), trim(ev_output_file), trim(chemistry_type), slot, particle_id)
end program convert_ev_to_hdf5