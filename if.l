%{
    #include<stdio.h>
%}
%%
[0-9]+ {printf("%s is an Integer\n",yytext);}
[0-9]+"."[0-9]* {printf("%s is Float\n",yytext);}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: ");
    yylex();
}