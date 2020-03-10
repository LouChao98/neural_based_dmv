#include<assert.h>
using namespace std;
#define M_PI       3.14159265358979323846

double gammaln(const double xx) //Retourne la valeur de ln(gamma(xx)) pour xx>0
{
 int j;
 double x,y,tmp,ser;
 static const double cof[6]={76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};
 
 y=x=xx;
 tmp=x+5.5;
 tmp -= (x+0.5)*log(tmp);
 ser=1.000000000190015;
 for (j=0;j<6;j++) ser += cof[j]/++y;
 return -tmp+log(2.5066282746310005*ser/x);
}

double digamma(double x) {
  double result = 0, xx, xx2, xx4;
  assert(x > 0);
  for ( ; x < 7; ++x)
    result -= 1/x;
  x -= 1.0/2.0;
  xx = 1.0/x;
  xx2 = xx*xx;
  xx4 = xx2*xx2;
  result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
  return result;
}

double gamma(double x)
{
    int i,k,m;
    double ga,gr,r,z;

    static double g[] = {
        1.0,
        0.5772156649015329,
       -0.6558780715202538,
       -0.420026350340952e-1,
        0.1665386113822915,
       -0.421977345555443e-1,
       -0.9621971527877e-2,
        0.7218943246663e-2,
       -0.11651675918591e-2,
       -0.2152416741149e-3,
        0.1280502823882e-3,
       -0.201348547807e-4,
       -0.12504934821e-5,
        0.1133027232e-5,
       -0.2056338417e-6,
        0.6116095e-8,
        0.50020075e-8,
       -0.11812746e-8,
        0.1043427e-9,
        0.77823e-11,
       -0.36968e-11,
        0.51e-12,
       -0.206e-13,
       -0.54e-14,
        0.14e-14};

    if (x > 171.0) return 1e308;    // This value is an overflow flag.
    if (x == (int)x) {
        if (x > 0.0) {
            ga = 1.0;               // use factorial
            for (i=2;i<x;i++) {
               ga *= i;
            }
         }
         else
            ga = 1e308;
     }
     else {
        if (fabs(x) > 1.0) {
            z = fabs(x);
            m = (int)z;
            r = 1.0;
            for (k=1;k<=m;k++) {
                r *= (z-k);
            }
            z -= m;
        }
        else
            z = x;
        gr = g[24];
        for (k=23;k>=0;k--) {
            gr = gr*z+g[k];
        }
        ga = 1.0/(gr*z);
        if (fabs(x) > 1.0) {
            ga *= r;
            if (x < 0.0) {
                ga = -M_PI/(x*ga*sin(M_PI*x));
            }
        }
    }
    return ga;
}

char* itoa(int value, char* result, int base) {
	// check that the base if valid
	if (base < 2 || base > 36) { *result = '\0'; return result; }

	char* ptr = result, *ptr1 = result, tmp_char;
	int tmp_value;
	
	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
	} while ( value );

	// Apply negative sign
	if (tmp_value < 0) *ptr++ = '-';
	*ptr-- = '\0';
	while(ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--= *ptr1;
		*ptr1++ = tmp_char;
	}
	return result;
}

string& trim(string &str)
{
    int i,j,start,end;

    //ltrim
    for (i=0; (str[i]!=0 && str[i]<=32); )
        i++;
    start=i;

    //rtrim
    for(i=0,j=0; str[i]!=0; i++)
        j = ((str[i]<=32)? j+1 : 0);
    end=i-j;
    str = str.substr(start,end-start);
    return str;
}

template <class T>
void alloc_arr_2D(int dim1, int dim2, T **&arr)
{
	arr = new T *[dim1];
	for(int i = 0; i < dim1; i++)
		arr[i] = new T[dim2];
}

template <class T>
void alloc_arr_3D(int dim1, int dim2, int dim3, T ***&arr)
{
	arr = new T **[dim1];
	for(int i = 0; i < dim1; i++)
		alloc_arr_2D(dim2, dim3, arr[i]);
}

template <class T>
void alloc_arr_4D(int dim1, int dim2, int dim3, int dim4, T ****&arr)
{
	arr = new T ***[dim1];
	for(int i = 0; i < dim1; i++)
		alloc_arr_3D(dim2, dim3, dim4, arr[i]);
}

template <class T>
void alloc_arr_5D(int dim1, int dim2, int dim3, int dim4, int dim5, T *****&arr)
{
	arr = new T ****[dim1];
	for(int i = 0; i < dim1; i++)
		alloc_arr_4D(dim2, dim3, dim4, dim5, arr[i]);
}

template <class T>
void alloc_arr_6D(int dim1, int dim2, int dim3, int dim4, int dim5, int dim6, T ******&arr)
{
	arr = new T *****[dim1];
	for(int i = 0; i < dim1; i++)
		alloc_arr_5D(dim2, dim3, dim4, dim5, dim6, arr[i]);
}

template <class T>
void delete_arr_2D(int dim1, T **&arr)
{
	for(int i = 0; i < dim1; i++)
		delete [] arr[i];
	delete [] arr;
}

template <class T>
void delete_arr_3D(int dim1, int dim2, T ***&arr)
{
	for(int i = 0; i < dim1; i++)
		delete_arr_2D(dim2, arr[i]);
	delete [] arr;
}

template <class T>
void delete_arr_4D(int dim1, int dim2, int dim3, T ****&arr)
{
	for(int i = 0; i < dim1; i++)
		delete_arr_3D(dim2, dim3, arr[i]);
	delete [] arr;
}

template <class T>
void delete_arr_5D(int dim1, int dim2, int dim3, int dim4, T *****&arr)
{
	for(int i = 0; i < dim1; i++)
		delete_arr_4D(dim2, dim3, dim4, arr[i]);
	delete [] arr;
}

template <class T>
void delete_arr_6D(int dim1, int dim2, int dim3, int dim4, int dim5, T ******&arr)
{
	for(int i = 0; i < dim1; i++)
		delete_arr_5D(dim2, dim3, dim4, dim5, arr[i]);
	delete [] arr;
}