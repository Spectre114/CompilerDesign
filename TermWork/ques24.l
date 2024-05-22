%{
    #include<stdio.h>
%}
%s A B C
%%
<INITIAL>a BEGIN B;
<INITIAL>b BEGIN A;
<INITIAL>\n BEGIN INITIAL; {printf("Accepted\n");}
<A>a BEGIN C;
<A>b BEGIN INITIAL;
<A>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<B>b BEGIN C;
<B>a BEGIN INITIAL;
<B>\n BEGIN INITIAL; {printf("Not Accepted\n");}
<C>a BEGIN A;
<C>b BEGIN B;
<C>\n BEGIN INITIAL; {printf("Not Accepted\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}
