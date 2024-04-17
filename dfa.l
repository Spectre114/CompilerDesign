%{
    #include<stdio.h>
%}
%s A B D
%%
<INITIAL>0 BEGIN INITIAL;
<INITIAL>1 BEGIN A;
<INITIAL>\n BEGIN INITIAL; {printf("NOT ACCEPTED\n");}
<INITIAL>[^01\n] BEGIN D; {printf("INVALID INPUT\n");}
<A>0 BEGIN B;
<A>1 BEGIN A;
<A>\n BEGIN INITIAL; {printf("NOT ACCEPTED\n");}
<A>[^01\n] BEGIN D; {printf("INVALID INPUT\n");}
<B>0 BEGIN INITIAL;
<B>1 BEGIN A;
<B>\n BEGIN INITIAL; {printf("ACCEPTED\n");}
<B>[^01\n] BEGIN D; {printf("INVALID INPUT\n");}
<D>0 BEGIN D; 
<D>1 BEGIN D; 
<D>[^01\n] BEGIN D;
<D>\n BEGIN INITIAL; {printf("Not Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}