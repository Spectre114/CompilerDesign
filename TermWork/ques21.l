%{
    #include<stdio.h>
%}
%s A B C
%%
<INITIAL>0 BEGIN B;
<INITIAL>1 BEGIN A;
<INITIAL>\n BEGIN INITIAL; {printf("Accepted\n");}
<A>0 BEGIN C;
<A>1 BEGIN INITIAL;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>1 BEGIN C;
<B>0 BEGIN INITIAL;
<B>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<C>0 BEGIN A;
<C>1 BEGIN B;
<C>\n BEGIN INITIAL; {printf("Not Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}
