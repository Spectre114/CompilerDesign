%{
    #include<stdio.h>
%}
%%
[a-zA-Z][a-zA-Z0-9]* {printf("%s is an identifier\n",yytext);}
.* {printf("%s is not an identifier\n",yytext);}
%%
int yywrap(){return 1;}
int main(){
	printf("Enter identifier\n");
	yylex();
}