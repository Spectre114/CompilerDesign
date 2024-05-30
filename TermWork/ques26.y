%{
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

void yyerror(const char *s);
int yylex(void);

%}
%token A B
%%
S:
    /*empty*/ {printf("Valid String\n");}
    | A S B
    ;
%%
int main(void) {
    return yyparse();
}

void yyerror(const char *s) {
    fprintf(stderr, "Error: %s\n", s);
}
