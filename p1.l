%{
#include<stdio.h>
int c = 0;
%}
%%
[a-zA-Z] {c++;}
%%
int yywrap() {return 1;}
int main()
{
	printf("Enter input:");
	yylex();
	printf("Total number of characters: %d\n",c);
}