//
//  ds_zero_select
//
//  select dominant DSs with nearly zero LOS changes
//
//  DSs are separated into two output files:
//  one, where the velocity is smaller than zero criteria
//  and the others.
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void change_ext  ( char *name, char *ext );
void  pause (int i);

// ------------------------------------------------------
 
 int main(int argc, char *argv[])
{
   int n=0,nz=0,nt=0;
   
   char *inp;    // input file
   char *out1;   // output file
   char *out2;   // output file

   char *buf;                       
   char *log="zero_select.log";  

   FILE *in, *ou1, *ou2, *lo;     

   float zch;
   float la,fi,he, ve,vu;
   
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++");
    printf("\n +                   ds_zero_select                   +");    
    printf("\n +  select integrated DSs with nearly zero velocity   +");
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

//  printf("argc: %d\n",argc);  
//  printf("%s\n",argv[0]);
//  printf("%s\n",argv[1]);  
//  printf("%s\n",argv[2]);    

    if(argc<3)
   {
    printf("\n     usage:   ds_zero_select integrate.xyi 0.6         \n");
    printf("\n            integrate.xyi  -  integrated data file");    
    printf("\n            0.6 (mm/year)  -  zero data criteria\n");        
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    
    exit(1);
   }

  if (( inp = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate INP");  exit(1); }
  if (( out1 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate OUT");  exit(1); }
  if (( out2 = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate OUT");  exit(1); }
  if (( buf = (char *) malloc(80 * sizeof(char)) ) == NULL)
  { printf("\nNot enough memory to allocate buf");  exit(1); }
  
//   sprintf(out1,"%s%s",argv[1],".0ref");  // DSs smaller than criteria
//   sprintf(out2,"%s%s",argv[1],".0trg");  // DSs target

      sprintf(out1,"%s",argv[1]);  // DSs smaller than criteria
      sprintf(out2,"%s",argv[1]);  // DSs target
   change_ext(out1,"0ref");
   change_ext(out2,"0trg");

   sscanf(argv[2],"%f",&zch);

  if( (lo = fopen(log,"w+t")) == NULL )
  { printf("\n LOG data file not found ! "); exit(1); }

   fprintf(lo,"\n %s %s %s\n",argv[0],argv[1],argv[2]);
   
   printf("\n    input: %s"  ,argv[1]);
   printf("\n   output: %s\n           %s\n",out1,out2);      
   printf("\n   zero DSs < |%3.1f| mm/year\n",zch); 

   fprintf(lo,"\n    input: %s"  ,argv[1]);
   fprintf(lo,"\n   output: %s\n           %s\n",out1,out2);      
   fprintf(lo,"\n   zero DSs < |%3.1f| mm/year\n",zch); 

  if( (in = fopen(argv[1],"rt")) == NULL )
  { printf("\n %s data file not found ! ",argv[1]); exit(1); }
  if( (ou1 = fopen(out1,"w+t")) == NULL )
  { printf("\n OUT1 data file not found ! "); exit(1); }
  if( (ou2 = fopen(out2,"w+t")) == NULL )
  { printf("\n OUT2 data file not found ! "); exit(1); }
    
//---------------------------------------------------------------
     
   while( fscanf(in,"%f %f %f %f %f",&la,&fi,&he,&ve,&vu) >0 )
  { 

      if( sqrt(ve*ve+vu*vu) <= zch )
     {
        fprintf(ou1,"%16.7e %15.7e %9.3f %7.3f %7.3f\n",la,fi,he,ve,vu);        
        nz++;

  
      }
      else 
     {
       fprintf(ou2,"%16.7e %15.7e %9.3f %7.3f %7.3f\n",la,fi,he,ve,vu);  
       nt++;
     }
      n++; if((n%1000)==0) printf("\n %6d ...",n); 
  } 
      printf("\n %6d\n",n); 
  
         printf("\n   zero dominant DSs %6d\n        target   DSs %6d\n",nz,nt);

      printf("\n Records of output files:\n"); 
      printf("\n longitude latitude  height  ew_v   up_v"); 
      printf("\n (     degree          m       mm/year )\n");  

     fprintf(lo,"\n   zero dominant DSs %6d\n        target   DSs %6d\n",nz,nt); 

     fprintf(lo,"\n Records of output files:\n"); 
     fprintf(lo,"\n longitude latitude  height  ew_v   up_v"); 
     fprintf(lo,"\n (     degree          m       mm/year )\n\n");  


//----------------------------------------------------------------------
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++"); 
    printf("\n +              end    ds_zero_select                 +");
    printf("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n");    

return(1);

}  // end main 
  
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
 printf("\n\n Pause(%1d) => Ctrl c to exit - Enter to continue !\n",i);
 getchar();
}
