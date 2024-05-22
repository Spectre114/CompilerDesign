%{
    #include<stdio.h>
%}
%%
[0-9]+ {printf("%s is an Integer\n",yytext);}
[0-9]*"."[0-9]+ {printf("%s is a Float\n",yytext);}
"if"|"else"|"for"|"while"|"int"|"float" {printf("%s is a Keyword\n",yytext);}
"+"|"-"|"*"|"/"|"="|"<"|">"|"++"|"--" {printf("%s is an Operator\n",yytext);}
","|";" {printf("%s is A Seperator\n",yytext);}
[a-zA-Z][a-zA-Z0-9]* {printf("%s is an Identifier\n",yytext);}
[ ] {printf("");} 
. {printf("Invalid Input\n");}
%%
int yywrap(){return 1;}
int main(){
    printf("Enter the input: \n");
    yylex();
}
