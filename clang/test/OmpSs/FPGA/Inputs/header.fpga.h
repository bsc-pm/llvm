#pragma once

// Used in testing, not actual correct functions
int OMPIF_Comm_rank();
int OMPIF_Comm_size();

void OMPIF_Send(const void* data, int count);
void nanos6_fpga_memcpy_wideport_in(int *ptr);