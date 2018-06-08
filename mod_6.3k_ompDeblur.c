//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Vivian Doan
 * UCLA ID: 904914594
 * Email id: doanpvivian@gmail.com
 * Input: Old files
 */

// 6.3k

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
  // commented out pow function and replaced with x * x;
  return x*x;
  //return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
  	// changed (2*stepCount) to stepCount + stepCount
  	//double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
  	double lambda = (Ksigma * Ksigma) / (double)(stepCount + stepCount);
	double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	int x, y, z, step;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, (double)((stepCount << 2) - stepCount));

	for(step = 0; step < stepCount; step++)
	{
		int a, b;
		int blockSize = 8;
		
		/*
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				u[Index(0, y, z)] *= boundryScale;
			}
		}
		*/
		#pragma omp parallel
		for(z = 0; z < zMax; z += blockSize)
		{
			for(y = 0; y < yMax; y += blockSize)
			{
				#pragma omp parallel
				for (a = z; a < blockSize; a++)
				{
					#pragma omp parallel
					for (b = y; b < blockSize; b++)
					{
						// strength reduction
						double temp = u[Index(0, b, a)];
						int c;
						for (c = 0; c < boundryScale; c++)
						{
							temp += temp;
						}
						u[Index(0, b, a)] = temp;
					}
				}
				
			}
		}

		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 1; x < xMax; x++)
				{
					int xyzInd = Index(x, y, z);
					u[xyzInd] += u[xyzInd - 1] * nu;
				}
			}
		}
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				u[Index(0, y, z)] *= boundryScale;
			}
		}
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = xMax - 2; x >= 0; x--)
				{
					int xyzInd = Index(x, y , z);
					u[xyzInd] += u[xyzInd + 1] * nu;
				}
			}
		}
		for(z = 0; z < zMax; z++)
		{
			for(x = 0; x < xMax; x++)
			{
				u[Index(x, 0, z)] *= boundryScale;
			}
		}
		for(z = 0; z < zMax; z++)
		{
			for(y = 1; y < yMax; y++)
			{
				for(x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
				}
			}
		}
		for(z = 0; z < zMax; z++)
		{
			for(x = 0; x < xMax; x++)
			{
				u[Index(x, yMax - 1, z)] *= boundryScale;
			}
		}
		for(z = 0; z < zMax; z++)
		{
			for(y = yMax - 2; y >= 0; y--)
			{
				for(x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] += u[Index(x, y + 1, z)] * nu;
				}
			}
		}
		for(x = 0; x < xMax; x++)
		{
			for(y = 0; y < yMax; y++)
			{
				u[Index(x, y, 0)] *= boundryScale;
			}
		}
		for(z = 1; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] = u[Index(x, y, z - 1)] * nu;
				}
			}
		}
		for(x = 0; x < xMax; x++)
		{	
			for(y = 0; y < yMax; y++)
			{
				u[Index(x, y, zMax - 1)] *= boundryScale;
			}
		}
		for(z = zMax - 2; z >= 0; z--)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 0; x < xMax; x++)
				{
					u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
				}
			}
		}
	}
	for(z = 0; z < zMax; z++)
	{
		for(y = 0; y < yMax; y++)
		{
			for(x = 0; x < xMax; x++)
			{
				u[Index(x, y, z)] *= postScale;
			}
		}
	}
}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		for(z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for(x = 1; x < xMax - 1; x++)
				{
					// code motion to avoid calculating xyzInd on every iteration
					int xyzInd = Index(x, y, z);
					double u_xyzInd = u[xyzInd];
					g[xyzInd] = 1.0 / sqrt(epsilon + 
						SQR(u_xyzInd - u[xyzInd + 1]) + 
						SQR(u_xyzInd - u[xyzInd - 1]) + 
						SQR(u_xyzInd - u[Index(x, y + 1, z)]) + 
						SQR(u_xyzInd - u[Index(x, y - 1, z)]) + 
						SQR(u_xyzInd - u[Index(x, y, z + 1)]) + 
						SQR(u_xyzInd - u[Index(x, y, z - 1)]));
				}
			}
		}
		memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
		OMP_GaussianBlur(conv, Ksigma, 3);
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 0; x < xMax; x++)
				{
					// more code motion, avoid calculating index of xyz 
					int xyzInd = Index(x, y, z);
					double r = conv[xyzInd] * f[xyzInd] / sigma2;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[xyzInd] -= f[xyzInd] * r;
				}
			}
		}
		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;
		for(z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for(x = 1; x < xMax - 1; x++)
				{
					// more code motion
					int xyzInd = Index(x, y, z);
					int xyz_xmin = xyzInd - 1;
					int xyz_xplu = xyzInd + 1;
					int xyz_ymin = Index(x, y - 1, z);
					int xyz_yplu = Index(x, y + 1, z);
					int xyz_zmin = Index(x, y, z - 1);
					int xyz_zplu = Index(x, y, z + 1);

					double g_xmin = g[xyz_xmin];
					double g_xplu = g[xyz_xplu];
					double g_ymin = g[xyz_ymin];
					double g_yplu = g[xyz_yplu];
					double g_zmin = g[xyz_zmin];
					double g_zplu = g[xyz_zplu];

					double oldVal = u[xyzInd];
					double newVal = (u[xyzInd] + dt * ( 
						u[xyz_xmin] * g_xmin + 
						u[xyz_xplu] * g_xplu + 
						u[xyz_ymin] * g_ymin + 
						u[xyz_yplu] * g_yplu + 
						u[xyz_zmin] * g_zmin + 
						u[xyz_zplu] * g_zplu - gamma * conv[xyzInd])) /
						(1.0 + dt * (g_xplu + g_xmin + g_yplu + g_ymin + g_zplu + g_zmin));
					if(fabs(oldVal - newVal) < epsilon)
					{
						converged++;
					}
					u[xyzInd] = newVal;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}

