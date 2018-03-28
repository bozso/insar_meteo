//
// ps_poly_orbit
//
// input asc_master.res  or  dsc_master.res and degree
// ouput asc_master.reso or  dsc_master.reso 
//
// substract the tabular orbits from master.res file
// and estimates the polynomial orbit
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { double t; double x; double y; double z; } torb; 

 int  poly_fit   ( int m, int u, torb *orb, double *X, char c, FILE *lo );
 int  plc        ( int i, int j, int n ); 
 void ATA_ATL    ( int m, int u, double *A, double *L, double *ATA,double *ATL ); 
 int  chole      ( double *q, int n );  
 void change_ext ( char *name, char *ext );

 void  pause (int i);

// ------------------------------------------------------
 
 int main(int argc, char * argv[])
{
  int i,dop,               // deegre of polinomials
        ndp;               // number of orbit records
  torb *orb;               // tabular orbit data
  double *X, t,x,y,z;

  char *out, *buf, *head;
  char *log;              // log output file

  
  FILE *in,*ou, *lo;

  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  printf("\n +                     ps_poly_orbit                     +");    
  printf("\n +    tabular orbit data are converted to polynomials    +"); 
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

  if(argc<3)
 {
  printf("\n          usage:    ps_poly_orbit asc_master.res 4"        );
  printf("\n                 or"                                       );
  printf("\n                    ps_poly_orbit dsc_master.res 4"        );
  printf("\n\n          asc_master.res or dsc_master.res - input files"); 
  printf("\n          4                                - degree     \n");               
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
  exit(1);
 }
    
//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]); 
  
  if (( out = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate OUT"); exit(1); }
  if (( log = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate LOG"); exit(1); }


  if (( buf = (char *) malloc(80 * sizeof(char)) )  == NULL)
  { printf("\nNot enough memory to allocate BUF");  exit(1); } 
  if (( head = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate HEAD"); exit(1); } 

  head="NUMBER_OF_DATAPOINTS:"; 
  sscanf(argv[2],"%d",&dop); 

//  sprintf(out,"%s%s",argv[1],"o"); 
      sprintf( out,"%s",argv[1] ); 
  change_ext ( out,"porb" );

  sprintf(log,"%s%s",out,".log"); 
      
  if( (in = fopen(argv[1],"rt")) == NULL )
  { printf("\n  1. Data file not found ! "); exit(1); }
  if( (ou = fopen(out,"w+t")) == NULL )
  { printf("\n  3. Data file not found ! "); exit(1); }
  if( (lo = fopen(log,"w+t")) == NULL )
  { printf("\n  4. Data file not found ! "); exit(1); }

   fprintf(lo,"\n %s %s %s\n",argv[0], argv[1],argv[2]); 
   fprintf(lo,"\n  input: %s",argv[1]);
   fprintf(lo,"\n output: %s",out); 
   fprintf(lo,"\n degree: %d\n",dop); 

   printf("\n  input: %s",argv[1]);
   printf("\n output: %s",out); 
   printf("\n degree: %d\n",dop); 

   while( fscanf(in,"%s",buf)>0 && strncmp(buf,head,21) !=0 ); 
   fscanf(in,"%d",&ndp);
   
   if (( orb = (torb *) malloc(ndp * sizeof(torb)) ) == NULL)
  { printf("\nNot enough memory to allocate orb"); exit(1); } 
   if (( X = (double *) calloc((dop+1), sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate X"); exit(1); } 
    
  for(i=0;i<ndp;i++)
{ 
  fscanf(in,"%lf %lf %lf %lf",&t,&x,&y,&z);
  (orb+i)->t=t;
  (orb+i)->x=x;
  (orb+i)->y=y;
  (orb+i)->z=z;  
} 
  fprintf(ou,"%3d\n",dop);
  fprintf(ou,"%13.5f\n", orb->t);
  fprintf(ou,"%13.5f\n",(orb+ndp-1)->t);    
//
      printf("\n fit of X coordinates:");
  fprintf(lo,"\n fit of X coordinates:");
  poly_fit ( ndp, dop+1,orb, X, 'x',lo );  // ***********
  for(i=0;i<(dop+1);i++) fprintf(ou," %23.15e",*(X+i));   
  fprintf(ou,"\n");  

       printf("\n\n fit of Y coordinates:");
  fprintf(lo,"\n\n fit of Y coordinates:");
  poly_fit ( ndp, dop+1,orb, X, 'y',lo );  // ***********
  for(i=0;i<(dop+1);i++) fprintf(ou," %23.15e",*(X+i));  
  fprintf(ou,"\n");
    
       printf("\n\n fit of Z coordinates:");
  fprintf(lo,"\n\n fit of Z coordinates:");
  poly_fit ( ndp, dop+1,orb, X, 'z',lo );  // ***********
  for(i=0;i<(dop+1);i++) fprintf(ou," %23.15e",*(X+i));  
  fprintf(ou,"\n\n");  

  fprintf(lo,"\n\n");

  fclose(in);
  fclose(ou);
  fclose(lo);

  printf("\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++");     
  printf("\n +             end          ps_poly_orbit                +");
  printf("\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
     
}  // end main   
// ***********************************************************

// --------------------------------------------------------------------
 int  poly_fit ( int m, int u, torb *orb, double *X, char c, FILE *lo )
{
//
// o(t) = a0 + a1*t + a2*t^2 + a3*t^3  + ... 
//    
  int i,j;         
  double t, *A, *L, *ATA, *ATL, mu0=0.0;
   
   if ((   A = (double *) malloc(u * sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate A"); exit(1); } 
   if (( ATA = (double *) calloc(u*(u+1)/2, sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate ATA"); exit(1); } 
   if (( ATL = (double *) calloc(u, sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate ATL"); exit(1); } 
   if ((   L = (double *) calloc( 1, sizeof(double)) ) == NULL)
  { printf("\nNot enough memory to allocate L"); exit(1); } 

  for(i=0;i<m;i++)
 { 
          if(c=='x') *L= (orb+i)->x;
     else if(c=='y') *L= (orb+i)->y; 
     else if(c=='z') *L= (orb+i)->z;           
     t= (orb+i)->t - orb->t;   
     for(j=0;j<u;j++) *(A+j)= pow(t,1.0*j);     
     ATA_ATL( 1, u, A, L, ATA, ATL);    
 } 
   if( chole( ATA,u ) != 0 )
   { printf("\n Error - singular normal matrix ! \n press any key"); exit(0); } 

     for(i=0;i<u;i++) // update of unknowns
     {                 *(X+i) = 0.0;
     for(j=0;j<u;j++) *(X+i) = *(X+i) + *(ATA+plc(i,j,u)) * *(ATL+j); }
    
  for(i=0;i<m;i++)
 { 
          if(c=='x') *L= -(orb+i)->x;
     else if(c=='y') *L= -(orb+i)->y; 
     else if(c=='z') *L= -(orb+i)->z; 
     t= (orb+i)->t - orb->t;
     
     for(j=0;j<u;j++) *L += *(X+j)* pow(t,1.0*j); 
     mu0 += *L * *L;
 }
   mu0=sqrt(mu0/(m-u)*1.0);

   fprintf(lo,"  mu0= %8.4lf dof= %d",mu0,m-u);
//   fprintf(lo,"\n\n         coefficients                  std\n");
  
   printf("\n\n mu0= %8.4lf",mu0);     
   printf("     dof= %d",m-u);
   printf("\n\n         coefficients                  std\n");

   for(j=0;j<u;j++) 
  { 
        printf("\n%2d %23.15e   %23.15e",j, *(X+j), mu0*sqrt( *(ATA+plc(j,j,u)) )); 
 
//  fprintf(lo,"\n%2d %23.15e   %23.15e",j, *(X+j), mu0*sqrt( *(ATA+plc(j,j,u)) ));
  
  } 
     
 return(1); 
} // end poly_fit

// -------------------------------------------------
 void change_ext ( char *name, char *ext )
{
// change the extent of name for ext

  int  i=0;
  while( *(name+i) != '.' && *(name+i) != '\0') i++;
  *(name+i)='\0';

  sprintf(name,"%s.%s",name,ext);

 } // end change_ext

  
// ---------------------------------------------------------------------
  void pause(int i)
{
 printf("\n\n Pause(%d) => Ctrl c to exit - Enter to continue !\n",i);
 getchar();
}
// **************************
 int plc(int i, int j, int n)
{
// position of i-th, j-th element  0
// in the lower triangle           1 2
// stored in vector                3 4 5 

 int k,l;
 k=(i<j) ? i+1 : j+1;
 l=(i>j) ? i+1 : j+1;
 return( (l-1)*l/2+k-1 );
} // end plc

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 void ATA_ATL( int m,int u, double *A, double *L, double *ATA, double *ATL)
{
// m - number of measurements
// u - number of unknowns

  int i,j,k,l;
  double *buf;

  if (( buf = (double *) malloc( m * sizeof(double)) ) == NULL)
  { printf("\n Not enough memory to allocate BUF \n press any key"); exit(1); }

  for(i=0;i<u;i++) // row of ATPA
  {

// buffer +++++++++++++++++++++++++++++++++
    for(k=0;k<m;k++)
   {  *(buf+k)=0.0;
    for(l=0;l<m;l++) *(buf+k)= *(buf+k) + *(A + l*u + i );
   } 
// buffer +++++++++++++++++++++++++++++++++

   for(l=0;l<m;l++)
   {
    *(ATL+i) = *(ATL+i) + *(buf+l) * *(L+l);
   }

  for(j=i;j<u;j++) // column of ATPA
  {
   k=plc(i,j,u);

   for(l=0;l<m;l++)
   {
    *(ATA+k) = *(ATA+k) + *(buf+l) * *(A+l*u+j);
   }
  }  // end - column of ATPA
  }  // end - row of ATPA
  free(buf);

} // end ATA_ATL

// *************************

 int chole(double *q, int n)
{
// Cholesky decomposition of
// symmetric positive definit
// normal matrix

 int i,j,k,l,ia,l1;
 double a,sqr,sum;

 for(i=0;i<n;i++) {
  a=*(q+plc(i,i,n));

  if( a<=0.0 ) return(1);
  sqr=sqrt(a);
  for(k=i;k<n;k++) *(q+plc(i,k,n))/=sqr;
  l=i+1; if(l==n) goto end1;
  for(j=l;j<n;j++) for(k=j;k<n;k++) {

   *(q+plc(j,k,n))-= *(q+plc(i,j,n)) * *(q+plc(i,k,n)); }
 }
 end1:
 for(i=0;i<n;i++) {

  ia=plc(i,i,n); *(q+ia)=1.0/ *(q+ia);
  l1=i+1; if(l1==n) goto end2;
  for(j=l1;j<n;j++) {

   sum=0.0;
   for(k=i;k<j;k++) sum+= *(q+plc(i,k,n)) * *(q+plc(k,j,n));
   *(q+plc(i,j,n))=-sum/ *(q+plc(j,j,n)); }
 }
 end2:
 for(i=0;i<n;i++) for(k=i;k<n;k++) { sum=0.0;

  for(j=k;j<n;j++) {
   sum+= *(q+plc(i,j,n)) * *(q+plc(k,j,n)); }
  *(q+plc(i,k,n))=sum; }

 return(0);
 } // end chole
