/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "parameters.cuh"
#include "utilities/error.cuh"
#include <cmath>

Parameters::Parameters(char* input_dir)
{
  print_line_1();
  printf("Started reading potential.in.\n");
  print_line_2();

  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/nep.in");
  FILE* fid = my_fopen(file, "r");
  char name[20];

  int count = fscanf(fid, "%s%f", name, &rc);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("cutoff = %g A.\n", rc);
  if (rc < 3.0f) {
    PRINT_INPUT_ERROR("cutoff should >= 3 A.");
  } else if (rc > 10.0f) {
    PRINT_INPUT_ERROR("cutoff should <= 10 A.");
  }

  count = fscanf(fid, "%s%d", name, &n_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("n_max = %d.\n", n_max);
  if (n_max < 0) {
    PRINT_INPUT_ERROR("n_max should >= 0.");
  } else if (n_max > 12) {
    PRINT_INPUT_ERROR("n_max should <= 12.");
  }

  count = fscanf(fid, "%s%d", name, &L_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("l_max = %d.\n", L_max);
  if (L_max < 0) {
    PRINT_INPUT_ERROR("l_max should >= 0.");
  } else if (L_max > 6) {
    PRINT_INPUT_ERROR("l_max should <= 6.");
  }

  int dim = (n_max + 1) * (L_max + 1);
  printf("number of descriptor components = %d.\n", dim);

  count = fscanf(fid, "%s%d", name, &num_neurons1);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("num_neurons1 = %d.\n", num_neurons1);
  if (num_neurons1 < 1) {
    PRINT_INPUT_ERROR("num_neurons1 should >= 1.");
  } else if (num_neurons1 > 100) {
    PRINT_INPUT_ERROR("num_neurons1 should <= 100.");
  }

  count = fscanf(fid, "%s%d", name, &num_neurons2);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("num_neurons2 = %d.\n", num_neurons2);
  if (num_neurons2 < 0) {
    PRINT_INPUT_ERROR("num_neurons2 should >= 0.");
  } else if (num_neurons2 > 50) {
    PRINT_INPUT_ERROR("num_neurons2 should <= 50.");
  }

  if (num_neurons2 > 0) {
    if (num_neurons1 > 50) {
      PRINT_INPUT_ERROR("num_neurons1 should <= 50 when num_neurons1 > 0.");
    }
  }

  number_of_variables = (dim + 1) * num_neurons1;
  number_of_variables += (num_neurons1 + 1) * num_neurons2;
  number_of_variables += (num_neurons2 == 0 ? num_neurons1 : num_neurons2) + 1;
  printf("number of parameters to be optimized = %d.\n", number_of_variables);

  count = fscanf(fid, "%s%d", name, &batch_size);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("batch_size = %d.\n", batch_size);
  if (batch_size < 1) {
    PRINT_INPUT_ERROR("num_neurons2 should >= 1.");
  }

  population_size = 4 + int(std::floor(3.0f * std::log(number_of_variables * 1.0f)));
  printf("population_size = %d.\n", population_size);

  count = fscanf(fid, "%s%d", name, &maximum_generation);
  PRINT_SCANF_ERROR(count, 2, "reading error for potential.in.");
  printf("maximum_generation = %d.\n", maximum_generation);
  if (maximum_generation < 100) {
    PRINT_INPUT_ERROR("maximum_generation should >= 100.");
  } else if (maximum_generation > 1000000) {
    PRINT_INPUT_ERROR("maximum_generation should <= 1000000.");
  }

  fclose(fid);
}