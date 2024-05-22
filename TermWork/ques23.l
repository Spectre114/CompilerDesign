%{
    #include<stdio.h>
%}
%s A B C
%%
<INITIAL>[a-zA-Z] BEGIN A;
<INITIAL>[0-9] BEGIN B;
<A>[a-zA-Z0-9] BEGIN A;
<A>\n BEGIN INITIAL; {printf("Identifier\n");}
<B>[0-9] BEGIN B;
<B>[.] BEGIN C;
<B>\n BEGIN INITIAL; {printf("Integer\n");}
<C>[0-9] BEGIN C;
<C>\n BEGIN INITIAL; {printf("Float\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}