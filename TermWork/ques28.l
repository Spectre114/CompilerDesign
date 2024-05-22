%{
    #include<stdio.h>
%}
%s A B C D 
%%
<INITIAL>0 BEGIN A;
<INITIAL>1 BEGIN C;
<INITIAL>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<A>0 BEGIN A;
<A>1 BEGIN B;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>1 BEGIN B;
<B>0 BEGIN A;
<B>\n BEGIN INITIAL; {printf("Accepted\n");}
<C>0 BEGIN D;
<C>1 BEGIN C;
<C>\n BEGIN INITIAL; {printf("Accepted\n");}
<D>0 BEGIN D;
<D>1 BEGIN C;
<D>\n BEGIN INITIAL; {printf("Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}
