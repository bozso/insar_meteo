// ps_data_select
//
// Input files: (StaMPS ps_data.xy ->) asc_data.xy and des_data.xy
// adjacent ascending and descending PSs are selected.
//
// One PS is selected if there is at last one PS in the 
// other series inside the approximate separation distance.
//
// The PS"la1,fi1" is selected if the first PS"la2,fi2"
// is closer than the sepration distance "dam"
//
// The "dam", "la1,fi1" and "la2,fi1" are interpreted
// on spherical Earth with radius 6372000 m 

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define R 6372000      // radius of Earth
#define C 57.295779513 // 180/pi

typedef struct { float la; float fi;  } psxy;

void  pause (int i);
 int  selectp(float dam, FILE *in1, psxy *in2,int ni, FILE *ou1);

// -------------------------------------------------------------
 
 int main(int argc, char *argv [])
{
   int i,n,ni1,ni2;
   psxy *indata;
   
   char *inp1;                      // ASC input file
   char *inp2;                      // DSC input file     
   char *out1;                      // output file
   char *out2;                      // output file  

   char *logf="data_select.log";    // log output file
 
   FILE *in1,*in2,*ou1, *ou2, *log;    

   float dam;
   float la,fi,v,he,dhe;

  if (( out1 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\n Not enough memory to allocate OUT1");  exit(1); }
  if (( out2 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\n Not enough memory to allocate OUT2");  exit(1); }

    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++");
    printf("\n +                  ps_data_select                    +");    
    printf("\n + adjacent ascending and descending PSs are selected +");
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

    if(argc<4)
   {
    printf("\n   usage:  ps_data_select asc_data.xy dsc_data.xy 100  \n");
    printf("\n           asc_data.xy  - (1st) ascending  data file");    
    printf("\n           dsc_data.xy  - (2nd) descending data file");    
    printf("\n           100          - (3rd) PSs separation (m)\n");     
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
    exit(1);
   }

    if( (log = fopen(logf,"w+t")) == NULL )
   { printf("\n  LOG file not found ! "); exit(1); }

    fprintf(log,"\n %s %s %s %s\n",argv[0],argv[1],argv[2],argv[3]);


    sprintf(out1,"%s%s",argv[1],"s");
    sprintf(out2,"%s%s",argv[2],"s");   
         printf("\n  input: %s\n output: %s\n",argv[1],out1);
         printf("\n  input: %s\n output: %s\n",argv[2],out2); 

    fprintf(log,"\n  input: %s\n output: %s\n",argv[1],out1);
    fprintf(log,"\n  input: %s\n output: %s\n",argv[2],out2); 
      
  if( (in1 = fopen(argv[1],"rt")) == NULL )
  { printf("\n  ASC Data file not found ! "); exit(1); }
  if( (in2 = fopen(argv[2],"rt")) == NULL )
  { printf("\n  DSC Data file not found ! "); exit(1); }

  if( (ou1 = fopen(out1,"w+t")) == NULL )
  { printf("\n  OU1 Data file not found ! "); exit(1); }
  if( (ou2 = fopen(out2,"w+t")) == NULL )
  { printf("\n  OU2 Data file not found ! "); exit(1); }

//---------------------------------------------------------------
   sscanf(argv[3],"%f",&dam);

        printf("\n Appr. PSs separation %5.1f (m)\n",dam); 
   fprintf(log,"\n Appr. PSs separation %5.1f (m)",  dam); 
//----------------------------------------------------------------

   ni1=0;
   while(fscanf(in1,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe)>0) ni1++;
   rewind(in1); 
 
   ni2=0;
   while(fscanf(in2,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe)>0) ni2++;
   rewind(in2); 

//  Copy data to memory 
    
   if (( indata = (psxy *) malloc(ni2 * sizeof(psxy)) ) == NULL)
   { printf("\nNot enough memory to allocate indata 1");  exit(1); }

   for(i=0;i<ni2;i++)
   {
     fscanf(in2,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe);
     (indata+i)->la=la;
     (indata+i)->fi=fi;               
   }

//-------------------------------------------------------------------  

        printf("\n\n %s  PSs %d\n",argv[1],ni1);
   fprintf(log,"\n\n %s  PSs %d",  argv[1],ni1);
 
   printf("\n Select PSs ...\n");  
   n=selectp(dam, in1, indata,ni2, ou1);   // **************
   rewind(ou1);
   rewind(in1);
   rewind(in2);

        printf("\n\n %s PSs %d\n",out1,n);
     fprintf(log,"\n %s PSs %d"  ,out1,n);

//-------------------------------------------------------------------

        printf("\n\n %s  PSs %d\n",argv[2],ni2);
   fprintf(log,"\n\n %s  PSs %d",  argv[2],ni2);

// Copy data to memory 
 
   free(indata);
   if (( indata = (psxy *) malloc(n * sizeof(psxy)) ) == NULL)
   { printf("\nNot enough memory to allocate indata 2");  exit(1); }
   for(i=0;i<n;i++)
   {
     fscanf(ou1,"%e %e %e %e %e",&la,&fi,&v,&he,&dhe);
     (indata+i)->la=la;
     (indata+i)->fi=fi;               
   }
//--------------------------------------------------------------------------

   printf("\n Select PSs ...\n");                                            
   n=selectp(dam, in2, indata,n, ou2);      // **************

      printf("\n\n %s PSs %d\n" ,out2,n);
   fprintf(log,"\n %s PSs %d\n\n",out2,n);

printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++");
printf("\n +                end   ps_data_select                +");
printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");

return(1);
      
}  // end main   
// *************************************************************

// -------------------------------------------------------------
  int selectp(float dam, FILE *in1, psxy *in2, int ni, FILE *ou1)
 { 
// The PS"la1,fi1" is selected if the first PS"la2,fi1"
// is closer than the sepration distance "dam" 
// The "dam", "la1,fi1" and "la2,fi1" are interpreted
// on spherical Earth with radius 6372000 m 
//
  int n=0, // no of selected PSs
      m=0, // no of data in in1
      ef;
  float fi1,la1,v1,he1,dhe1, fi2,la2,v2, da;

  float dm=dam/R*C*dam/R*C; // faster run
  
   while( fscanf(in1,"%e %e %e %e %e",&la1,&fi1,&v1,&he1,&dhe1) > 0)
  { 
    ef=0;                 
    do {         
          la2= (in2+ef)->la;
          fi2= (in2+ef)->fi; 

//        da= R * acos( sin(fi1/C)*sin(fi2/C)+cos(fi1/C)*cos(fi2/C)*cos((la1-la2)/C) ); 
//        da= da-dam;

          da= (fi1-fi2)*(fi1-fi2)+(la1-la2)*(la1-la2); // faster run
          da= da-dm;                                   // rewer  PSs

          ef++;  
       } 
    while( (da > 0.0) && ( (ef-1) < ni) ); 

    if( (ef-1) < ni ) {
                         fprintf(ou1,"%16.7e %16.7e %16.7e %16.7e %16.7e\n",la1,fi1,v1,he1,dhe1);
                        n++;
                      }
     m++;  if((m%10000)==0) printf("\n %6d ...",m);               
   }  
   return(n);
 } // end selectp

// ---------------------------------------------------------------------
void pause(int i)
{
 printf("\n\n Pause(%1d) => Ctrl c to exit - Enter to continue !\n",i);
 getchar(); 
} // end pause
