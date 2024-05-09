%{
#include<stdio.h>
%}
%%
[0-9]+ {printf("%s is an integer no\n",yytext);}
[0-9]+"."[0-9]+ {printf("%s is float no\n",yytext);}
%%
int yywrap(){return 1;}
int main(){
	printf("Enter number\n");
	yylex();
}